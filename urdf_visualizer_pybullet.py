import pybullet as p
import time
import os
import math
import sys
import keyboard
import trimesh
import numpy as np
import random
import xml.etree.ElementTree as ET
import tempfile  # 임시 파일 관리를 위한 모듈 추가
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (for 3D plotting)
import csv
import hashlib
import json


# PLY 처리 모듈 import 추가
from ply_processor import load_pipe_from_ply, load_ply_as_pybullet_body


# 사용자 지정 데이터 경로 함수
def get_data_path():
    return os.path.dirname(os.path.abspath(__file__))

def get_file_hash(file_path):
    """파일의 MD5 해시를 계산하여 캐시 키로 사용"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def check_cache_valid(ply_path, obj_path, vhacd_path):
    """캐시 파일이 유효한지 확인"""
    cache_info_path = obj_path + ".cache_info"
    
    if not all(os.path.exists(f) for f in [obj_path, vhacd_path, cache_info_path]):
        return False
    
    try:
        with open(cache_info_path, 'r') as f:
            cache_info = json.load(f)
        
        current_hash = get_file_hash(ply_path)
        return cache_info.get('source_hash') == current_hash
    except:
        return False

def save_cache_info(ply_path, obj_path):
    """캐시 정보 저장"""
    cache_info_path = obj_path + ".cache_info"
    cache_info = {
        'source_hash': get_file_hash(ply_path),
        'created_time': time.time(),
        'source_file': ply_path
    }
    
    with open(cache_info_path, 'w') as f:
        json.dump(cache_info, f)

def convert_ply_to_obj(ply_path, obj_path, timeout_seconds=60):
    """PLY를 OBJ로 변환 (타임아웃과 폴백 메커니즘 포함)"""
    start_time = time.time()
    
    try:
        # PLY 파일 로드
        mesh = trimesh.load(ply_path)
        
        # 포인트 클라우드인지 확인
        if isinstance(mesh, trimesh.points.PointCloud):
            original_count = len(mesh.vertices)
            print(f"포인트 클라우드 감지: {original_count:,}개 포인트")
            
            # 극도로 큰 포인트 클라우드는 거부
            if original_count > 1000000:  # 100만개 이상
                print(f"경고: 너무 큰 포인트 클라우드 ({original_count:,}개)")
                print("기본 파이프로 폴백합니다.")
                return None
            
            # 대용량 포인트 클라우드 다운샘플링
            max_points = 100000  # 최대 100K 포인트로 제한
            if original_count > max_points:
                print(f"대용량 포인트 클라우드 다운샘플링: {original_count:,} → {max_points:,}")
                
                # 균등한 간격으로 샘플링
                step = original_count // max_points
                indices = np.arange(0, original_count, step)[:max_points]
                mesh.vertices = mesh.vertices[indices]
                
                print(f"다운샘플링 완료: {len(mesh.vertices):,}개 포인트")
            
            # 타임아웃 체크
            if time.time() - start_time > timeout_seconds:
                print("타임아웃 - 기본 파이프로 폴백")
                return None
            
            print("Convex hull로 메시 변환 중...")
            
            # Convex hull을 사용하여 메시 생성
            try:
                mesh = mesh.convex_hull
                print(f"Convex hull 생성 완료: {len(mesh.vertices):,}개 vertices, {len(mesh.faces):,}개 faces")
            except Exception as e:
                print(f"Convex hull 생성 실패: {e}")
                print("기본 파이프로 폴백합니다.")
                return None
                
        # 최종 타임아웃 체크
        if time.time() - start_time > timeout_seconds:
            print("타임아웃 - 기본 파이프로 폴백")
            return None
        
        # OBJ 파일로 저장
        mesh.export(obj_path)
        print(f"변환 완료: {time.time() - start_time:.2f}초 소요")
        
        return obj_path
        
    except Exception as e:
        print(f"PLY 변환 중 오류 발생: {e}")
        print("기본 파이프로 폴백합니다.")
        return None

# VHACD를 사용하여 컨벡스 분해 수행
def perform_convex_decomposition(mesh_path, fast_mode=False):
    print(f"메시 파일 '{mesh_path}'에 대해 컨벡스 분해 수행 중...")
    
    if fast_mode:
        print("고속 모드 활성화 - 성능 우선 파라미터 사용")
        # 고속 처리를 위한 파라미터
        concavity = 0.05         # 더 높은 값으로 단순화
        resolution = 50000       # 해상도 절반으로 감소
        depth = 6               # 분해 깊이 대폭 감소
        maxVertices = 32        # 정점 수 절반으로 감소
    else:
        print("일반 모드 - 품질과 성능 균형 파라미터 사용")
        # 기본 파라미터 (기존보다 약간 최적화)
        concavity = 0.025       # 약간 증가
        resolution = 75000      # 적당히 감소
        depth = 8              # 약간 감소
        maxVertices = 48       # 약간 감소
    
    # VHACD 파라미터 설정
    p.vhacd(
        mesh_path,               # 입력 OBJ 파일
        mesh_path + "_vhacd.obj",   # 출력 OBJ 파일
        "log.txt",               # 로그 파일
        concavity=concavity,     # 동적 concavity 값
        alpha=0.04,              # alpha 값
        beta=0.05,               # beta 값
        gamma=0.01,              # gamma 값 증가
        minVolumePerCH=0.002,    # 최소 볼륨 증가
        resolution=resolution,   # 동적 해상도
        maxNumVerticesPerCH=maxVertices,  # 동적 정점 수
        depth=depth,             # 동적 분해 깊이
        planeDownsampling=6,     # 다운샘플링 증가
        convexhullDownsampling=6,# 볼록 헐 다운샘플링 증가
        pca=0,                   # PCA 활성화 여부
        mode=0,                  # 볼륨 기반 분해 모드
        convexhullApproximation=1 # 볼록 헐 근사 활성화
    )
    
    print(f"컨벡스 분해 완료: {mesh_path}_vhacd.obj")
    return mesh_path + "_vhacd.obj"

# URDF 파일의 mesh 경로를 수정하는 함수 추가
def modify_urdf_mesh_paths(urdf_file, output_file):
    # XML 파싱
    tree = ET.parse(urdf_file)
    root = tree.getroot()
    
    # 현재 작업 디렉토리 절대 경로
    current_dir = get_data_path()
    
    # 모든 mesh 요소 검색 및 경로 수정
    for mesh in root.findall(".//mesh"):
        if 'filename' in mesh.attrib:
            # 상대 경로를 절대 경로로 변환
            rel_path = mesh.attrib['filename']
            abs_path = os.path.abspath(os.path.join(current_dir, rel_path))
            mesh.attrib['filename'] = abs_path
    
    # 수정된 URDF 저장
    tree.write(output_file)
    return output_file

# 충돌 검사 함수 추가 (멀티 로봇 지원)
def check_collision(robot_id, joint_positions, pipe_id=None, other_robots=None):
    """로봇의 현재 관절 위치에서 충돌이 있는지 확인하고 충돌된 링크를 반환 (멀티 로봇 지원)"""
    # 관절 위치 설정
    for i, pos in enumerate(joint_positions):
        p.resetJointState(robot_id, i, pos)
    
    # 로봇의 모든 충돌 검사 (파이프 및 다른 로봇 포함)
    all_contact_points = p.getContactPoints(robot_id)
    collision_links = set()
    pipe_collision = False
    robot_collision = False
    
    # 충돌이 있는지 확인
    has_collision = len(all_contact_points) > 0
    
    # 모든 접촉점 디버깅 (로봇과 모든 객체 사이의 충돌)
    if has_collision:
        print(f"로봇 {robot_id} 충돌 감지: 접촉점 수={len(all_contact_points)}")
        for i, contact in enumerate(all_contact_points):
            print(f"  일반 접촉점 {i+1}: bodyA={contact[1]}, bodyB={contact[2]}, 링크A={contact[3]}, 링크B={contact[4]}")
    
    # 로봇의 충돌 링크 수집
    for contact in all_contact_points:
        # contact[1]과 contact[2]는 충돌한 두 객체의 ID
        # contact[3]과 contact[4]는 충돌한 링크의 인덱스
        bodyA = contact[1]
        bodyB = contact[2]
        linkA = contact[3]
        
        if bodyA == robot_id:
            collision_links.add(linkA)
            # 로봇과 파이프의 충돌 확인
            if pipe_id is not None and bodyB == pipe_id:
                pipe_collision = True
                print(f"  -> 로봇 {robot_id}이 파이프와 충돌! (로봇 링크 {linkA} - 파이프)")
            # 로봇과 다른 로봇의 충돌 확인
            elif other_robots is not None and bodyB in other_robots:
                robot_collision = True
                print(f"  -> 로봇 {robot_id}이 로봇 {bodyB}와 충돌! (링크 {linkA} - 링크 {contact[4]})")
        elif bodyB == robot_id:  # 충돌 순서가 반대인 경우도 확인
            collision_links.add(contact[4])  # 로봇 링크 인덱스
            if pipe_id is not None and bodyA == pipe_id:
                pipe_collision = True
                print(f"  -> 파이프가 로봇 {robot_id}와 충돌! (파이프 - 로봇 링크 {contact[4]})")
            elif other_robots is not None and bodyA in other_robots:
                robot_collision = True
                print(f"  -> 로봇 {bodyA}이 로봇 {robot_id}와 충돌! (링크 {contact[3]} - 링크 {contact[4]})")
    
    # 파이프와의 충돌 확인 (파이프 ID가 제공된 경우)
    if pipe_id is not None:
        # 파이프와 로봇 사이의 접촉점만 직접 확인
        pipe_contacts = p.getContactPoints(robot_id, pipe_id)
        
        if len(pipe_contacts) > 0:
            pipe_collision = True
            print(f"로봇 {robot_id} 파이프 충돌 접촉점 수: {len(pipe_contacts)}")  # 디버깅용
            
            # 추가 디버그 정보
            for i, contact in enumerate(pipe_contacts):
                print(f"  접촉점 {i+1}: 로봇 링크 {contact[3]}, 거리: {contact[8]}")
                
                # 충돌 지점에 디버그 라인 추가
                p.addUserDebugLine(
                    contact[5],  # 접촉점 위치 A
                    contact[6],  # 접촉점 위치 B
                    [1, 0, 0],   # 빨간색
                    lineWidth=3.0,
                    lifeTime=0.5  # 0.5초 동안 표시
                )
    
    # 다른 로봇들과의 충돌 확인 (다른 로봇 ID가 제공된 경우)
    if other_robots is not None:
        for other_robot_id in other_robots:
            if other_robot_id != robot_id:
                robot_contacts = p.getContactPoints(robot_id, other_robot_id)
                if len(robot_contacts) > 0:
                    robot_collision = True
                    print(f"로봇 {robot_id}과 로봇 {other_robot_id} 충돌 접촉점 수: {len(robot_contacts)}")
                    
                    # 로봇 간 충돌 지점에 디버그 라인 추가
                    for i, contact in enumerate(robot_contacts):
                        print(f"  로봇간 접촉점 {i+1}: 로봇1 링크 {contact[3]}, 로봇2 링크 {contact[4]}, 거리: {contact[8]}")
                        
                        p.addUserDebugLine(
                            contact[5],  # 접촉점 위치 A
                            contact[6],  # 접촉점 위치 B
                            [1, 0.5, 0],   # 주황색 (로봇 간 충돌)
                            lineWidth=3.0,
                            lifeTime=0.5
                        )
    
    return has_collision, collision_links, pipe_collision, robot_collision

def get_random_configuration(robot_id):
    """로봇의 무작위 관절 위치 생성"""
    num_joints = p.getNumJoints(robot_id)
    joint_limits = []
    
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        if joint_info[2] == p.JOINT_REVOLUTE:
            joint_limits.append((joint_info[8], joint_info[9]))  # lower, upper limits
    
    return [random.uniform(low, high) for low, high in joint_limits]

def distance_between_configurations(q1, q2):
    """두 관절 위치 간의 거리 계산"""
    return np.linalg.norm(np.array(q1) - np.array(q2))

def find_nearest_node(tree, q_rand):
    """트리에서 가장 가까운 노드 찾기"""
    min_dist = float('inf')
    nearest_node = None
    
    for node in tree:
        dist = distance_between_configurations(node['config'], q_rand)
        if dist < min_dist:
            min_dist = dist
            nearest_node = node
            
    return nearest_node

def steer(q_near, q_rand, step_size):
    """q_near에서 q_rand 방향으로 step_size만큼 이동"""
    direction = np.array(q_rand) - np.array(q_near)
    distance = np.linalg.norm(direction)
    
    if distance <= step_size:
        return q_rand
    
    return q_near + (direction / distance) * step_size

# === RRT* 관련 함수 추가 ===
def get_cost(q1, q2):
    return distance_between_configurations(q1, q2)

def find_near_nodes(tree, q_new, radius):
    near_nodes = []
    for node in tree:
        if distance_between_configurations(node['config'], q_new) <= radius:
            near_nodes.append(node)
    return near_nodes

def choose_parent(near_nodes, q_new):
    min_cost = float('inf')
    best_parent = None
    for node in near_nodes:
        cost = node['cost'] + get_cost(node['config'], q_new)
        if cost < min_cost:
            min_cost = cost
            best_parent = node
    return best_parent, min_cost

def rewire(tree, near_nodes, new_node, robot_id, pipe_id=None, other_robots=None):
    for node in near_nodes:
        new_cost = new_node['cost'] + get_cost(new_node['config'], node['config'])
        if new_cost < node['cost']:
            collision_result = check_collision(robot_id, node['config'], pipe_id, other_robots)
            is_collision = collision_result[0]  # has_collision
            if not is_collision:
                node['parent'] = new_node
                node['cost'] = new_cost

def rrt_star_plan(robot_id, start_config, goal_config, max_iterations=5000, step_size=0.05, radius=0.5, pipe_id=None, other_robots=None):
    tree = [{'config': start_config, 'parent': None, 'cost': 0}]
    for iteration in range(max_iterations):
        q_rand = goal_config if random.random() < 0.1 else get_random_configuration(robot_id)
        nearest_node = find_nearest_node(tree, q_rand)
        q_near = nearest_node['config']
        q_new = steer(q_near, q_rand, step_size)
        collision_result = check_collision(robot_id, q_new, pipe_id, other_robots)
        is_collision = collision_result[0]  # has_collision
        if not is_collision:
            near_nodes = find_near_nodes(tree, q_new, radius)
            best_parent, min_cost = choose_parent(near_nodes, q_new)
            new_node = {'config': q_new, 'parent': best_parent, 'cost': min_cost}
            tree.append(new_node)
            rewire(tree, near_nodes, new_node, robot_id, pipe_id, other_robots)
            if distance_between_configurations(q_new, goal_config) < step_size:
                path = []
                current = new_node
                while current is not None:
                    path.append(current['config'])
                    current = current['parent']
                return list(reversed(path))
        if iteration % 100 == 0:
            print(f"로봇 {robot_id} RRT* 진행 중... {iteration}/{max_iterations}")
    return None

def get_link_position(robot_id, joint_positions):
    """주어진 관절 위치에서 로봇의 끝점(TCP) 위치를 계산"""
    # 관절 상태 설정
    for i, pos in enumerate(joint_positions):
        p.resetJointState(robot_id, i, pos)
    
    # TCP 링크 찾기 - DDA와 일반 URDF 모두 대응
    num_joints = p.getNumJoints(robot_id)
    tcp_link_index = num_joints - 1  # 기본적으로 마지막 링크
    
    # 링크 이름 확인으로 TCP 링크 정확히 찾기
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        link_name = joint_info[12].decode('utf-8')  # child link name
        if 'tcp' in link_name.lower():
            tcp_link_index = i
            break
    
    # TCP 링크의 상태 가져오기
    tcp_link_state = p.getLinkState(robot_id, tcp_link_index)
    return tcp_link_state[0]  # 위치 반환

def visualize_path(robot_id, path, color=[0, 1, 0]):
    """경로를 시각화 (멀티 로봇 지원, 로봇별 다른 색상)"""
    # 경로의 각 지점에서 TCP 위치 계산
    points = []
    for config in path:
        tcp_pos = get_link_position(robot_id, config)
        points.append(tcp_pos)
    
    # 경로를 선으로 연결
    for i in range(len(points)-1):
        p.addUserDebugLine(
            points[i],
            points[i+1],
            lineColorRGB=color,  # 로봇별 다른 색상
            lineWidth=2.0,
            lifeTime=0  # 0 = 영구적으로 표시
        )

def draw_coordinate_axes(object_id, scale=0.1):
    """객체의 원점에 XYZ 좌표축 화살표를 그리는 함수"""
    if object_id == -1:  # 월드 원점
        position = [0, 0, 0]
        orientation = [0, 0, 0, 1]
    else:
        # 객체의 위치와 방향 가져오기
        position, orientation = p.getBasePositionAndOrientation(object_id)
    
    # 회전 행렬로 변환하여 축 방향 계산
    rotation_matrix = p.getMatrixFromQuaternion(orientation)
    
    # X, Y, Z 축 방향 벡터 계산
    x_axis = [rotation_matrix[0] * scale, rotation_matrix[3] * scale, rotation_matrix[6] * scale]
    y_axis = [rotation_matrix[1] * scale, rotation_matrix[4] * scale, rotation_matrix[7] * scale]
    z_axis = [rotation_matrix[2] * scale, rotation_matrix[5] * scale, rotation_matrix[8] * scale]
    
    # 축 끝점 위치 계산
    x_end = [position[0] + x_axis[0], position[1] + x_axis[1], position[2] + x_axis[2]]
    y_end = [position[0] + y_axis[0], position[1] + y_axis[1], position[2] + y_axis[2]]
    z_end = [position[0] + z_axis[0], position[1] + z_axis[1], position[2] + z_axis[2]]
    
    # X축 화살표 (빨간색)
    p.addUserDebugLine(
        position, x_end, 
        lineColorRGB=[1, 0, 0], 
        lineWidth=3.0, 
        lifeTime=0
    )
    
    # Y축 화살표 (초록색)  
    p.addUserDebugLine(
        position, y_end, 
        lineColorRGB=[0, 1, 0], 
        lineWidth=3.0, 
        lifeTime=0
    )
    
    # Z축 화살표 (파란색)
    p.addUserDebugLine(
        position, z_end, 
        lineColorRGB=[0, 0, 1], 
        lineWidth=3.0, 
        lifeTime=0
    )
    
    # 축 라벨 텍스트 추가
    p.addUserDebugText("X", x_end, textColorRGB=[1, 0, 0], textSize=1.0, lifeTime=0)
    p.addUserDebugText("Y", y_end, textColorRGB=[0, 1, 0], textSize=1.0, lifeTime=0)
    p.addUserDebugText("Z", z_end, textColorRGB=[0, 0, 1], textSize=1.0, lifeTime=0)

def draw_link_coordinate_axes(robot_id, scale=0.08):
    """로봇의 모든 링크에 좌표축을 표시하는 함수"""
    num_joints = p.getNumJoints(robot_id)
    
    for link_index in range(num_joints):
        # 링크 상태 가져오기
        link_state = p.getLinkState(robot_id, link_index)
        position = link_state[0]  # 링크 위치
        orientation = link_state[1]  # 링크 방향
        
        # 링크 이름 가져오기
        joint_info = p.getJointInfo(robot_id, link_index)
        link_name = joint_info[12].decode('utf-8')
        
        # 회전 행렬로 변환하여 축 방향 계산
        rotation_matrix = p.getMatrixFromQuaternion(orientation)
        
        # X, Y, Z 축 방향 벡터 계산
        x_axis = [rotation_matrix[0] * scale, rotation_matrix[3] * scale, rotation_matrix[6] * scale]
        y_axis = [rotation_matrix[1] * scale, rotation_matrix[4] * scale, rotation_matrix[7] * scale]
        z_axis = [rotation_matrix[2] * scale, rotation_matrix[5] * scale, rotation_matrix[8] * scale]
        
        # 축 끝점 위치 계산
        x_end = [position[0] + x_axis[0], position[1] + x_axis[1], position[2] + x_axis[2]]
        y_end = [position[0] + y_axis[0], position[1] + y_axis[1], position[2] + y_axis[2]]
        z_end = [position[0] + z_axis[0], position[1] + z_axis[1], position[2] + z_axis[2]]
        
        # 링크별로 다른 색상 강도 사용 (베이스 객체보다 연한 색)
        alpha = 0.7  # 투명도
        
        # X축 화살표 (연한 빨간색)
        p.addUserDebugLine(
            position, x_end, 
            lineColorRGB=[1, 0.3, 0.3], 
            lineWidth=2.0, 
            lifeTime=0
        )
        
        # Y축 화살표 (연한 초록색)  
        p.addUserDebugLine(
            position, y_end, 
            lineColorRGB=[0.3, 1, 0.3], 
            lineWidth=2.0, 
            lifeTime=0
        )
        
        # Z축 화살표 (연한 파란색)
        p.addUserDebugLine(
            position, z_end, 
            lineColorRGB=[0.3, 0.3, 1], 
            lineWidth=2.0, 
            lifeTime=0
        )
        
        # TCP 링크이면 특별히 표시
        if 'tcp' in link_name.lower():
            # TCP 링크는 더 큰 텍스트로 표시
            p.addUserDebugText(
                f"TCP", 
                [position[0], position[1], position[2] + 0.05], 
                textColorRGB=[1, 1, 0], 
                textSize=1.2, 
                lifeTime=0
            )
        else:
            # 일반 링크는 작은 텍스트로 링크 이름 표시
            p.addUserDebugText(
                f"L{link_index}", 
                [position[0], position[1], position[2] + 0.03], 
                textColorRGB=[0.7, 0.7, 0.7], 
                textSize=0.8, 
                lifeTime=0
            )

def reset_simulation(robots, pipe_id, robot_simulation_states):
    """모든 로봇의 시뮬레이션을 초기 상태로 리셋"""
    # 이전에 그려진 경로 제거
    p.removeAllUserDebugItems()
    
    # 모든 로봇의 관절을 초기 위치로 리셋
    for robot_id in robots:
        for i in range(p.getNumJoints(robot_id)):
            p.resetJointState(robot_id, i, 0.0)
        
        # 로봇 색상 초기화
        for i in range(p.getNumJoints(robot_id)):
            p.changeVisualShape(robot_id, i, rgbaColor=[1, 1, 1, 1])
        
        # 로봇 시뮬레이션 상태 초기화
        robot_simulation_states[robot_id] = {
            'simulation_started': False,
            'path_index': 0,
            'collision_detected': False
        }
    
    # 파이프 위치 리셋
    p.resetBasePositionAndOrientation(pipe_id, [0, 0.5, -0.50], [0, 0, 0, 1])
    
    # 파이프 색상 초기화
    p.changeVisualShape(pipe_id, -1, rgbaColor=[1, 1, 1, 1])
    
    # 좌표축 다시 그리기
    draw_coordinate_axes(-1, scale=0.3)  # 월드 원점
    if pipe_id is not None:
        draw_coordinate_axes(pipe_id, scale=0.2)  # 파이프
    for robot_id in robots:
        draw_coordinate_axes(robot_id, scale=0.15)  # 각 로봇
    
    return True

def highlight_collision_links(robot_id, collision_links, pipe_id=None, pipe_collision=False, robot_collision=False):
    """충돌한 링크와 파이프를 색상으로 강조 표시 (멀티 로봇 지원)"""
    # 모든 링크의 색상을 원래대로 복원
    for i in range(p.getNumJoints(robot_id)):
        p.changeVisualShape(robot_id, i, rgbaColor=[1, 1, 1, 1])
    
    # 충돌한 링크를 색상으로 표시
    for link_index in collision_links:
        if robot_collision:
            # 로봇 간 충돌은 주황색으로 표시
            p.changeVisualShape(robot_id, link_index, rgbaColor=[1, 0.5, 0, 1])
        else:
            # 파이프 충돌은 빨간색으로 표시
            p.changeVisualShape(robot_id, link_index, rgbaColor=[1, 0, 0, 1])
    
    # 파이프와 충돌 시 파이프 색상 변경
    if pipe_id is not None:
        if pipe_collision:
            p.changeVisualShape(pipe_id, -1, rgbaColor=[1, 0, 0, 1])  # 빨간색
        else:
            p.changeVisualShape(pipe_id, -1, rgbaColor=[1, 1, 1, 1])  # 흰색

def plot_path_3d(robot_id, path):
    xs, ys, zs = [], [], []
    for config in path:
        tcp_pos = get_link_position(robot_id, config)
        xs.append(tcp_pos[0])
        ys.append(tcp_pos[1])
        zs.append(tcp_pos[2])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs, ys, zs, marker='o', color='g', label='RRT* Path')
    ax.scatter(xs[0], ys[0], zs[0], color='b', s=50, label='Start')
    ax.scatter(xs[-1], ys[-1], zs[-1], color='r', s=50, label='Goal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('RRT* Path (TCP Trajectory)')
    ax.legend()
    plt.show()

def save_joint_trajectory_to_csv(path, filename="joint_trajectory.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 헤더: 시간, 관절1, 관절2, ...
        num_joints = len(path[0])
        header = ['time'] + [f'joint_{i+1}' for i in range(num_joints)]
        writer.writerow(header)
        # 각 행: 시간, 각 관절 각도
        for idx, config in enumerate(path):
            # 시간은 idx * dt (dt는 시뮬레이션 타임스텝, 예: 1/240초)
            dt = 1.0 / 240.0  # 시뮬레이션 타임스텝
            time_val = idx * dt
            writer.writerow([time_val] + list(config))
    print(f"Joint trajectory saved to {filename}")







def main():
    physicsClient = None
    try:
        # PyBullet 초기화 시도 (GUI 모드)
        try:
            # 기본 GUI 모드로 연결
            physicsClient = p.connect(p.GUI)
            if physicsClient < 0:
                raise Exception("Physics 서버 연결 실패")
        except p.error as e:
            print(f"연결 시도 중 오류: {e}")
            return 1
            
        # 멀티 로봇 데이터 구조 초기화
        robots = {}  # 로봇 ID를 키로 하는 로봇 데이터 딕셔너리
        robot_paths = {}  # 각 로봇의 경로
        robot_simulation_states = {}  # 각 로봇의 시뮬레이션 상태
            
        # 데이터 경로 설정 (pybullet_data 사용하지 않음)
        p.setGravity(0, 0, -9.81)  # 중력을 0으로 설정
        p.setRealTimeSimulation(0)  # 실시간 시뮬레이션 비활성화
        
        # 디버그 시각화 도구 설정
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)  # GUI 활성화
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)  # 마우스 픽킹 활성화
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)  # 그림자 활성화
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)  # 렌더링 활성화
        
        # Synthetic Camera 창들 비활성화
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        
        # 와이어프레임 모드 비활성화 (솔리드 렌더링)
        p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
        
        # 카메라 컨트롤 GUI 제거 - 마우스로만 제어

        # 초기 카메라 위치 설정
        p.resetDebugVisualizerCamera(
            cameraDistance=2.0,
            cameraYaw=0,
            cameraPitch=0,
            cameraTargetPosition=[0, 0, 0]
        )
        
        # 카메라 제어 설명
        print("\n카메라 제어 방법:")
        print("- 마우스 드래그: 카메라 회전")
        print("- 마우스 스크롤: 줌 인/아웃")
        print("- 마우스 중간버튼 드래그: 카메라 이동")

        # 고정값 설정 (로봇 1번)
        # 로봇 1 시작 위치
        robot1_start_x_val = 0.0
        robot1_start_y_val = 0.0
        robot1_start_z_val = 0.0
        
        # 로봇 1 시작 자세 (롤, 피치, 요)
        robot1_start_roll_val = 0.0
        robot1_start_pitch_val = 0.0
        robot1_start_yaw_val = 0.0
        
        # 로봇 1 종료 위치
        robot1_end_x_val = 0.0
        robot1_end_y_val = 0.2
        robot1_end_z_val = -0.3
        
        # 로봇 1 종료 자세 (롤, 피치, 요)
        robot1_end_roll_val = 0.0
        robot1_end_pitch_val = 0.0
        robot1_end_yaw_val = 0.0
        
        # 고정값 설정 (로봇 2번)
        # 로봇 2 시작 위치
        robot2_start_x_val = 0.5
        robot2_start_y_val = 0.0
        robot2_start_z_val = 0.0
        
        # 로봇 2 시작 자세 (롤, 피치, 요)
        robot2_start_roll_val = 0.0
        robot2_start_pitch_val = 0.0
        robot2_start_yaw_val = 0.0
        
        # 로봇 2 종료 위치
        robot2_end_x_val = 0.5
        robot2_end_y_val = 0.2
        robot2_end_z_val = -0.3
        
        # 로봇 2 종료 자세 (롤, 피치, 요)
        robot2_end_roll_val = 0.0
        robot2_end_pitch_val = 0.0
        robot2_end_yaw_val = 0.0
        
        # 파이프 위치 고정값
        pipe_x_val = 0.0
        pipe_y_val = 1.65
        pipe_z_val = 1.10
        
        # 파이프 자세 고정값 (롤, 피치, 요)
        pipe_roll_val = 1.65  # X축 90도 회전
        pipe_pitch_val = 0.13
        pipe_yaw_val = 0.13
        
        # 로봇 원점 위치 및 자세 고정값
        robot1_base_x_val = 0.0
        robot1_base_y_val = 0.0
        robot1_base_z_val = 0.0
        robot1_base_roll_val = 0.0
        robot1_base_pitch_val = 0.0
        robot1_base_yaw_val = 3.14  # 180도
        
        robot2_base_x_val = 1.0
        robot2_base_y_val = 0.0
        robot2_base_z_val = 0.0
        robot2_base_roll_val = 0.0
        robot2_base_pitch_val = 0.0
        robot2_base_yaw_val = 3.14  # 180도
        
        # Apply 버튼 추가
        apply_robot_base_button = p.addUserDebugParameter("Apply Robot Base", 0, 1, 0)
        
        # 로봇별 시작 버튼
        robot1_start_button = p.addUserDebugParameter("Robot1 Start", 0, 1, 0)
        robot2_start_button = p.addUserDebugParameter("Robot2 Start", 0, 1, 0)
        
        # 전체 초기화 버튼
        reset_button = p.addUserDebugParameter("Reset All", 0, 1, 0)
        
        # 이전 버튼 상태 저장
        previous_robot1_button_state = p.readUserDebugParameter(robot1_start_button)
        previous_robot2_button_state = p.readUserDebugParameter(robot2_start_button)
        previous_reset_state = p.readUserDebugParameter(reset_button)

        # 현재 디렉토리 가져오기
        current_dir = get_data_path()
        print(f"현재 디렉토리: {current_dir}")

        pipe_id = load_pipe_from_ply("pipe.ply", fallback_to_default=True)

        if pipe_id is None:
            print("파이프 로드에 실패했습니다.")
            return 1

        print(f"파이프 객체 ID: {pipe_id}")



        # 키 입력 정보 추가
        p.addUserDebugText(
            "W 키: 와이어프레임 모드 전환",
            [0, 0, 0.5],
            textColorRGB=[0, 1, 0],
            textSize=1.5
        )

        # 로봇 로드 (절대 경로 사용) - 2대 로봇 로드 (서로 다른 URDF 사용)
        # 로봇 1용 URDF (DDA 버전 - 메시 파일 참조 수정 완료)
        robot1_path = os.path.join(current_dir, "rb10_1300e_DDA.urdf")
        if not os.path.exists(robot1_path):
            raise Exception(f"로봇 1 URDF 파일을 찾을 수 없습니다: {robot1_path}")
        
        # 로봇 2용 URDF (일반 버전)
        robot2_path = os.path.join(current_dir, "rb10_1300e_RT.urdf")
        if not os.path.exists(robot2_path):
            raise Exception(f"로봇 2 URDF 파일을 찾을 수 없습니다: {robot2_path}")
                
        # 로봇 1 로드 (DDA 버전)
        robot1_orientation = p.getQuaternionFromEuler([0, 0, math.pi])  # 180도 회전
        robot1_position = [0, 0, 0]  # 원점에 배치
        
        try:
            robot1_id = p.loadURDF(
                robot1_path,
                basePosition=robot1_position,
                baseOrientation=robot1_orientation,
                useFixedBase=True
            )
            if robot1_id < 0:
                raise Exception("로봇 1 (DDA) URDF 로드 실패")
            print("로봇 1 (DDA 버전)이 성공적으로 로드되었습니다!")
            # 로봇 1 좌표축 표시
            draw_coordinate_axes(robot1_id, scale=0.15)
        except p.error as e:
            print(f"로봇 1 (DDA) 로드 중 오류 발생: {e}")
            raise
        
        # 로봇 2 로드 (일반 버전, 다른 위치에 배치)
        robot2_orientation = p.getQuaternionFromEuler([0, 0, math.pi])  # 180도 회전
        robot2_position = [1.0, 0, 0]  # X축으로 1m 이동하여 배치
        
        try:
            robot2_id = p.loadURDF(
                robot2_path,
                basePosition=robot2_position,
                baseOrientation=robot2_orientation,
                useFixedBase=True
            )
            if robot2_id < 0:
                raise Exception("로봇 2 (일반) URDF 로드 실패")
            print("로봇 2 (일반 버전)가 성공적으로 로드되었습니다!")
            # 로봇 2 좌표축 표시
            draw_coordinate_axes(robot2_id, scale=0.15)
        except p.error as e:
            print(f"로봇 2 (일반) 로드 중 오류 발생: {e}")
            raise

        # 로봇들을 데이터 구조에 저장
        robots[robot1_id] = {
            'id': robot1_id,
            'name': 'Robot1',
            'joints': [],
            'position': robot1_position,
            'orientation': robot1_orientation
        }
        
        robots[robot2_id] = {
            'id': robot2_id,
            'name': 'Robot2',
            'joints': [],
            'position': robot2_position,
            'orientation': robot2_orientation
        }
        
        # 각 로봇의 관절 정보 저장
        for robot_id in robots:
            robot_joints = []
            for i in range(p.getNumJoints(robot_id)):
                joint_info = p.getJointInfo(robot_id, i)
                if joint_info[2] == p.JOINT_REVOLUTE:
                    robot_joints.append({
                        'index': i,
                        'name': joint_info[1].decode('utf-8'),
                        'position': 0.0,
                        'target': 0.0,
                        'velocity': 0.0
                    })
            robots[robot_id]['joints'] = robot_joints
        
        # 각 로봇의 시뮬레이션 상태 초기화
        robot_simulation_states[robot1_id] = {
            'simulation_started': False,
            'path_index': 0,
            'collision_detected': False
        }
        
        robot_simulation_states[robot2_id] = {
            'simulation_started': False,
            'path_index': 0,
            'collision_detected': False
        }

        print("\n멀티 로봇 시뮬레이션 준비가 완료되었습니다.")
        print(f"로드된 로봇: {len(robots)}대")
        for robot_id in robots:
            print(f"  - {robots[robot_id]['name']} (ID: {robot_id})")
        print("1. 각 로봇의 시작 위치와 종료 위치를 설정하세요.")
        print("2. 파이프 위치(Pipe X/Y/Z)를 조절하여 원하는 위치로 이동하세요.")
        print("3. 'Robot1 Start' 또는 'Robot2 Start' 버튼을 클릭하여 각 로봇의 시뮬레이션을 시작하세요.")
        print("4. 'Reset All' 버튼으로 모든 로봇을 초기화할 수 있습니다.")

        print("6. 충돌이 발생하면 해당 부위가 빨간색으로 표시됩니다.")
        print("7. 로봇 간 충돌은 주황색으로 표시됩니다.")
        print("8. 마우스로 카메라를 자유롭게 조작할 수 있습니다.")
        print("9. Ctrl+C를 눌러 종료할 수 있습니다.")
        print("10. 로봇 원점 위치/자세를 변경한 후 'Apply Robot Base' 버튼을 눌러 적용하세요.\n")

        # 이전 Apply 버튼 상태 저장
        previous_apply_state = p.readUserDebugParameter(apply_robot_base_button)

        # 로봇 재로드 함수
        def reload_robots_with_new_base():
            nonlocal robot1_id, robot2_id, robots, robot1_path, robot2_path, robot_simulation_states
            
            # 새로운 로봇 원점 위치 및 자세 (고정값 사용)
            new_robot1_pos = [robot1_base_x_val, robot1_base_y_val, robot1_base_z_val]
            new_robot1_rpy = [robot1_base_roll_val, robot1_base_pitch_val, robot1_base_yaw_val]
            new_robot1_orientation = p.getQuaternionFromEuler(new_robot1_rpy)
            
            new_robot2_pos = [robot2_base_x_val, robot2_base_y_val, robot2_base_z_val]
            new_robot2_rpy = [robot2_base_roll_val, robot2_base_pitch_val, robot2_base_yaw_val]
            new_robot2_orientation = p.getQuaternionFromEuler(new_robot2_rpy)
            
            # PyBullet 연결 상태 확인
            if not p.isConnected():
                print("오류: PyBullet 연결이 끊어졌습니다.")
                return False
            
            # 기존 로봇 ID 백업 (복원용)
            old_robot1_id = robot1_id
            old_robot2_id = robot2_id
            
            # 기존 딕셔너리 엔트리 정리
            old_robots = robots.copy()
            robots.clear()
            
            try:
                # 기존 로봇 제거
                if old_robot1_id is not None:
                    p.removeBody(old_robot1_id)
                    print(f"로봇 1 (ID: {old_robot1_id}) 제거됨")
                if old_robot2_id is not None:
                    p.removeBody(old_robot2_id)
                    print(f"로봇 2 (ID: {old_robot2_id}) 제거됨")
                
                # 새로운 위치와 자세로 로봇 재로드
                print("새로운 위치로 로봇들을 로드 중...")
                
                robot1_id = p.loadURDF(
                    robot1_path,
                    basePosition=new_robot1_pos,
                    baseOrientation=new_robot1_orientation,
                    useFixedBase=True
                )
                
                if robot1_id < 0:
                    raise Exception("로봇 1 로드 실패")
                    
                robot2_id = p.loadURDF(
                    robot2_path,
                    basePosition=new_robot2_pos,
                    baseOrientation=new_robot2_orientation,
                    useFixedBase=True
                )
                
                if robot2_id < 0:
                    raise Exception("로봇 2 로드 실패")
                
                # 좌표축 표시
                draw_coordinate_axes(robot1_id, scale=0.15)
                draw_coordinate_axes(robot2_id, scale=0.15)
                
                # 새로운 로봇 데이터로 딕셔너리 업데이트
                robots[robot1_id] = {
                    'name': 'Robot1_DDA',
                    'joints': [],
                    'position': new_robot1_pos,
                    'orientation': new_robot1_orientation
                }
                robots[robot2_id] = {
                    'name': 'Robot2_RT', 
                    'joints': [],
                    'position': new_robot2_pos,
                    'orientation': new_robot2_orientation
                }
                
                # 관절 정보 재설정
                for robot_id in [robot1_id, robot2_id]:
                    robot_joints = []
                    joint_count = p.getNumJoints(robot_id)
                    
                    if joint_count < 0:
                        raise Exception(f"로봇 {robot_id}의 관절 정보 읽기 실패")
                    
                    for i in range(joint_count):
                        joint_info = p.getJointInfo(robot_id, i)
                        if joint_info[2] == p.JOINT_REVOLUTE:
                            robot_joints.append({
                                'index': i,
                                'name': joint_info[1].decode('utf-8'),
                                'position': 0.0,
                                'target': 0.0,
                                'velocity': 0.0
                            })
                    robots[robot_id]['joints'] = robot_joints
                
                # 시뮬레이션 상태 재초기화
                robot_simulation_states.clear()
                robot_simulation_states[robot1_id] = {
                    'simulation_started': False,
                    'path_index': 0,
                    'collision_detected': False
                }
                robot_simulation_states[robot2_id] = {
                    'simulation_started': False,
                    'path_index': 0,
                    'collision_detected': False
                }
                
                print("✅ 로봇들이 새로운 위치와 자세로 성공적으로 재로드되었습니다!")
                print(f"Robot1 (ID: {robot1_id}) 위치: {new_robot1_pos}, RPY: {new_robot1_rpy}")
                print(f"Robot2 (ID: {robot2_id}) 위치: {new_robot2_pos}, RPY: {new_robot2_rpy}")
                return True
                
            except Exception as e:
                print(f"❌ 로봇 재로드 중 오류 발생: {e}")
                print("기존 설정으로 복원을 시도합니다...")
                
                # 실패 시 기존 딕셔너리 복원
                robots.clear()
                robots.update(old_robots)
                
                # 기존 로봇 ID 복원
                robot1_id = old_robot1_id
                robot2_id = old_robot2_id
                
                print("기존 설정으로 복원되었습니다.")
                return False

        # 시뮬레이션 루프
        while p.isConnected():
            try:
                # 카메라 컨트롤 코드 제거 - 마우스만 사용
                
                # 버튼 상태 확인
                current_robot1_button_state = p.readUserDebugParameter(robot1_start_button)
                current_robot2_button_state = p.readUserDebugParameter(robot2_start_button)
                current_reset_state = p.readUserDebugParameter(reset_button)
                current_apply_state = p.readUserDebugParameter(apply_robot_base_button)
                
                # Apply Robot Base 버튼이 눌렸는지 확인
                if current_apply_state != previous_apply_state:
                    print("Apply Robot Base 버튼이 눌렸습니다. 로봇들을 재로드합니다...")
                    
                    # GUI 파라미터 유효성 검증
                    try:
                        new_pos1 = [robot1_base_x_val, robot1_base_y_val, robot1_base_z_val]
                        new_pos2 = [robot2_base_x_val, robot2_base_y_val, robot2_base_z_val]
                        
                        # 위치 값 유효성 검사
                        for pos in [new_pos1, new_pos2]:
                            for val in pos:
                                if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                                    raise ValueError("유효하지 않은 위치 값")
                        
                        # 로봇 재로드 실행
                        success = reload_robots_with_new_base()
                        if success:
                            print("로봇 베이스 위치가 성공적으로 적용되었습니다.")
                        else:
                            print("로봇 베이스 위치 적용에 실패했습니다.")
                            
                    except Exception as e:
                        print(f"Apply Robot Base 처리 중 오류: {e}")
                        print("GUI 파라미터를 확인하세요.")
                    
                    previous_apply_state = current_apply_state
                
                # 파이프 위치 업데이트 (고정값 사용)
                pipe_pos = [pipe_x_val, pipe_y_val, pipe_z_val]
                
                # 파이프 자세 업데이트 (고정값 사용)
                pipe_rpy = [pipe_roll_val, pipe_pitch_val, pipe_yaw_val]
                pipe_orientation = p.getQuaternionFromEuler(pipe_rpy)
                
                p.resetBasePositionAndOrientation(pipe_id, pipe_pos, pipe_orientation)
                
                # 로봇들 간의 지속적인 충돌 검사 (디버깅용)
                robot_ids = list(robots.keys())
                for i, robot_id in enumerate(robot_ids):
                    # 파이프와의 충돌 검사
                    debug_contacts = p.getContactPoints(robot_id, pipe_id)
                    if len(debug_contacts) > 0:
                        print(f"[디버그] 파이프-로봇{i+1} 접촉점: {len(debug_contacts)}")
                    
                    # 다른 로봇들과의 충돌 검사
                    for j, other_robot_id in enumerate(robot_ids):
                        if i < j:  # 중복 검사 방지
                            robot_contacts = p.getContactPoints(robot_id, other_robot_id)
                            if len(robot_contacts) > 0:
                                print(f"[디버그] 로봇{i+1}-로봇{j+1} 접촉점: {len(robot_contacts)}")
                
                # 초기화 버튼이 눌렸을 때
                if current_reset_state != previous_reset_state:
                    previous_reset_state = current_reset_state
                    if reset_simulation(robots, pipe_id, robot_simulation_states):
                        print("\n모든 로봇 시뮬레이션이 초기화되었습니다.")
                        print("새로운 시작 위치를 설정하고 파이프 위치를 조절하세요.")
                        robot_paths.clear()  # 모든 경로 지우기
                        continue
                
                # 로봇 1 시작 버튼이 눌렸을 때
                if current_robot1_button_state != previous_robot1_button_state:
                    previous_robot1_button_state = current_robot1_button_state
                    robot1_id = list(robots.keys())[0]  # 첫 번째 로봇 ID
                    
                    if not robot_simulation_states[robot1_id]['simulation_started']:
                        print("\n로봇 1 경로 생성 중...")
                        
                        # 로봇 1 시작 위치 (고정값 사용)
                        start_pos = [robot1_start_x_val, robot1_start_y_val, robot1_start_z_val]
                        
                        # 로봇 1 시작 자세 (고정값 사용)
                        start_rpy = [robot1_start_roll_val, robot1_start_pitch_val, robot1_start_yaw_val]
                        start_orientation = p.getQuaternionFromEuler(start_rpy)
                        
                        # 로봇 1 종료 위치 (고정값 사용)
                        end_pos = [robot1_end_x_val, robot1_end_y_val, robot1_end_z_val]
                        
                        # 로봇 1 종료 자세 (고정값 사용)
                        end_rpy = [robot1_end_roll_val, robot1_end_pitch_val, robot1_end_yaw_val]
                        end_orientation = p.getQuaternionFromEuler(end_rpy)
                        
                        print(f"로봇 1 시작 위치: {start_pos}, 자세(RPY): {start_rpy}")
                        print(f"로봇 1 종료 위치: {end_pos}, 자세(RPY): {end_rpy}")
                        
                        # 역기구학으로 관절 각도 계산
                        # TCP 링크 인덱스 찾기 (로봇 1)
                        tcp_link_index = p.getNumJoints(robot1_id) - 1
                        for i in range(p.getNumJoints(robot1_id)):
                            joint_info = p.getJointInfo(robot1_id, i)
                            link_name = joint_info[12].decode('utf-8')  # child link name
                            if 'tcp' in link_name.lower():
                                tcp_link_index = i
                                break
                        
                        start_joints = p.calculateInverseKinematics(
                            robot1_id, 
                            tcp_link_index, 
                            start_pos,
                            targetOrientation=start_orientation
                        )
                        
                        end_joints = p.calculateInverseKinematics(
                            robot1_id, 
                            tcp_link_index, 
                            end_pos,
                            targetOrientation=end_orientation
                        )
                        
                        # 다른 로봇들 ID 리스트 준비
                        other_robot_ids = [rid for rid in robots.keys() if rid != robot1_id]
                        
                        # RRT*로 경로 생성 (멀티 로봇 충돌 고려)
                        path = rrt_star_plan(robot1_id, start_joints, end_joints, pipe_id=pipe_id, other_robots=other_robot_ids)
                        
                        if path is None:
                            print("로봇 1 경로를 찾을 수 없습니다. 다른 위치를 시도해보세요.")
                            continue
                        
                        print(f"로봇 1 경로가 생성되었습니다. 경로 길이: {len(path)}")
                        
                        # 경로 시각화 (초록색)
                        visualize_path(robot1_id, path, color=[0, 1, 0])
                        
                        # 경로 저장
                        robot_paths[robot1_id] = path
                        save_joint_trajectory_to_csv(path, "robot1_trajectory.csv")
                        
                        # 로봇 1 시뮬레이션 상태 업데이트
                        robot_simulation_states[robot1_id]['simulation_started'] = True
                        robot_simulation_states[robot1_id]['path_index'] = 0
                        robot_simulation_states[robot1_id]['collision_detected'] = False
                
                # 로봇 2 시작 버튼이 눌렸을 때
                if current_robot2_button_state != previous_robot2_button_state:
                    previous_robot2_button_state = current_robot2_button_state
                    robot2_id = list(robots.keys())[1]  # 두 번째 로봇 ID
                    
                    if not robot_simulation_states[robot2_id]['simulation_started']:
                        print("\n로봇 2 경로 생성 중...")
                        
                        # 로봇 2 시작 위치 (고정값 사용)
                        start_pos = [robot2_start_x_val, robot2_start_y_val, robot2_start_z_val]
                        
                        # 로봇 2 시작 자세 (고정값 사용)
                        start_rpy = [robot2_start_roll_val, robot2_start_pitch_val, robot2_start_yaw_val]
                        start_orientation = p.getQuaternionFromEuler(start_rpy)
                        
                        # 로봇 2 종료 위치 (고정값 사용)
                        end_pos = [robot2_end_x_val, robot2_end_y_val, robot2_end_z_val]
                        
                        # 로봇 2 종료 자세 (고정값 사용)
                        end_rpy = [robot2_end_roll_val, robot2_end_pitch_val, robot2_end_yaw_val]
                        end_orientation = p.getQuaternionFromEuler(end_rpy)
                        
                        print(f"로봇 2 시작 위치: {start_pos}, 자세(RPY): {start_rpy}")
                        print(f"로봇 2 종료 위치: {end_pos}, 자세(RPY): {end_rpy}")
                        
                        # 역기구학으로 관절 각도 계산
                        # TCP 링크 인덱스 찾기 (로봇 2)
                        tcp_link_index = p.getNumJoints(robot2_id) - 1
                        for i in range(p.getNumJoints(robot2_id)):
                            joint_info = p.getJointInfo(robot2_id, i)
                            link_name = joint_info[12].decode('utf-8')  # child link name
                            if 'tcp' in link_name.lower():
                                tcp_link_index = i
                                break
                        
                        start_joints = p.calculateInverseKinematics(
                            robot2_id, 
                            tcp_link_index, 
                            start_pos,
                            targetOrientation=start_orientation
                        )
                        
                        end_joints = p.calculateInverseKinematics(
                            robot2_id, 
                            tcp_link_index, 
                            end_pos,
                            targetOrientation=end_orientation
                        )
                        
                        # 다른 로봇들 ID 리스트 준비
                        other_robot_ids = [rid for rid in robots.keys() if rid != robot2_id]
                        
                        # RRT*로 경로 생성 (멀티 로봇 충돌 고려)
                        path = rrt_star_plan(robot2_id, start_joints, end_joints, pipe_id=pipe_id, other_robots=other_robot_ids)
                        
                        if path is None:
                            print("로봇 2 경로를 찾을 수 없습니다. 다른 위치를 시도해보세요.")
                            continue
                        
                        print(f"로봇 2 경로가 생성되었습니다. 경로 길이: {len(path)}")
                        
                        # 경로 시각화 (파란색)
                        visualize_path(robot2_id, path, color=[0, 0, 1])
                        
                        # 경로 저장
                        robot_paths[robot2_id] = path
                        save_joint_trajectory_to_csv(path, "robot2_trajectory.csv")
                        
                        # 로봇 2 시뮬레이션 상태 업데이트
                        robot_simulation_states[robot2_id]['simulation_started'] = True
                        robot_simulation_states[robot2_id]['path_index'] = 0
                        robot_simulation_states[robot2_id]['collision_detected'] = False
                

                # 모든 로봇의 시뮬레이션 실행   
                for robot_id in robots:
                    robot_state = robot_simulation_states[robot_id]
                    
                    if (robot_state['simulation_started'] and 
                        robot_id in robot_paths and 
                        robot_state['path_index'] < len(robot_paths[robot_id]) and 
                        not robot_state['collision_detected']):
                        
                        # 현재 경로의 관절 위치로 이동
                        current_config = robot_paths[robot_id][robot_state['path_index']]
                        
                        # 로봇 관절 제어
                        for i, joint in enumerate(robots[robot_id]['joints']):
                            p.setJointMotorControl2(
                                bodyIndex=robot_id,
                                jointIndex=joint['index'],
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=current_config[i],
                                force=1000,
                                positionGain=1.0,
                                velocityGain=0.5,
                                maxVelocity=0.5
                            )
                        
                        # 다른 로봇들 ID 리스트 준비
                        other_robot_ids = [rid for rid in robots.keys() if rid != robot_id]
                        
                        # 충돌 검사 (파이프 및 다른 로봇과의 충돌 확인 포함)
                        collision_result = check_collision(robot_id, current_config, pipe_id, other_robot_ids)
                        has_collision, collision_links, pipe_collision, robot_collision = collision_result
                        
                        if has_collision:
                            robot_name = robots[robot_id]['name']
                            print(f"\n{robot_name} 충돌이 감지되었습니다!")
                            print(f"충돌한 링크: {collision_links}")
                            if pipe_collision:
                                print(f"{robot_name}이 파이프와 충돌했습니다!")
                            if robot_collision:
                                print(f"{robot_name}이 다른 로봇과 충뎼했습니다!")
                            
                            highlight_collision_links(robot_id, collision_links, pipe_id, pipe_collision, robot_collision)
                            robot_state['collision_detected'] = True
                        else:
                            # 충뎼이 없으면 다음 경로 지점으로 이동
                            robot_state['path_index'] += 1
                            
                            # 마지막 경로 지점에 도달한 경우
                            if robot_state['path_index'] >= len(robot_paths[robot_id]):
                                robot_name = robots[robot_id]['name']
                                print(f"\n{robot_name} 경로 실행이 완료되었습니다!")
                
                # 파이프 색상 업데이트 (어떤 로봇이라도 충돌 시 빨간색)
                any_pipe_collision = False
                for robot_id in robots:
                    if robot_id in robot_paths and robot_simulation_states[robot_id]['simulation_started']:
                        other_robot_ids = [rid for rid in robots.keys() if rid != robot_id]
                        if robot_simulation_states[robot_id]['path_index'] < len(robot_paths[robot_id]):
                            current_config = robot_paths[robot_id][robot_simulation_states[robot_id]['path_index']]
                            collision_result = check_collision(robot_id, current_config, pipe_id, other_robot_ids)
                            if len(collision_result) >= 3 and collision_result[2]:  # pipe_collision
                                any_pipe_collision = True
                                break
                
                if not any_pipe_collision:
                    p.changeVisualShape(pipe_id, -1, rgbaColor=[1, 1, 1, 1])  # 파이프를 흰색으로
                
                # 시뮬레이션 스텝 진행
                p.stepSimulation()                
                time.sleep(1./10.)  # 240Hz로 시뮬레이션 실행
                

            except KeyboardInterrupt:
                print("\n사용자가 시뮬레이션을 종료했습니다.")
                break
            except Exception as e:
                print(f"\n시뮬레이션 중 오류 발생: {e}")
                break

    except Exception as e:
        print(f"\n초기화 중 오류 발생: {e}")
        return 1
    finally:
        if physicsClient is not None and p.isConnected():
            try:
                p.disconnect()
                print("\n시뮬레이션이 종료되었습니다.")
            except:
                pass
    return 0

if __name__ == "__main__":
    sys.exit(main())