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

# tnwjd 

# 사용자 지정 데이터 경로 함수
def get_data_path():
    return os.path.dirname(os.path.abspath(__file__))

def convert_ply_to_obj(ply_path, obj_path):
    # PLY 파일 로드
    mesh = trimesh.load(ply_path)
    
    # OBJ 파일로 저장
    mesh.export(obj_path)
    
    return obj_path

# VHACD를 사용하여 컨벡스 분해 수행
def perform_convex_decomposition(mesh_path):
    print(f"메시 파일 '{mesh_path}'에 대해 컨벡스 분해 수행 중...")
    
    # VHACD 파라미터 설정 - 처리 시간 단축을 위해 값 조정
    p.vhacd(
        mesh_path,               # 입력 OBJ 파일
        mesh_path + "_vhacd.obj",   # 출력 OBJ 파일
        "log.txt",               # 로그 파일
        concavity=0.01,          # concavity 값 증가 (낮을수록 더 정확하지만 느림)
        alpha=0.04,              # alpha 값
        beta=0.05,               # beta 값
        gamma=0.005,             # gamma 값 증가
        minVolumePerCH=0.001,    # 최소 볼륨 증가
        resolution=100000,       # 해상도 감소 (높을수록 더 정확하지만 느림)
        maxNumVerticesPerCH=64,  # 볼록 헐당 최대 정점 수 감소
        depth=10,                # 분해 깊이 감소
        planeDownsampling=4,     # 평면 다운샘플링
        convexhullDownsampling=4,# 볼록 헐 다운샘플링
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

# 충돌 검사 함수 추가 
def check_collision(robot_id, joint_positions, pipe_id=None):
    """로봇의 현재 관절 위치에서 충돌이 있는지 확인하고 충돌된 링크를 반환"""
    # 관절 위치 설정
    for i, pos in enumerate(joint_positions):
        p.resetJointState(robot_id, i, pos)
    
    # 로봇의 모든 충돌 검사 (파이프 포함)
    all_contact_points = p.getContactPoints(robot_id)
    collision_links = set()
    pipe_collision = False
    
    # 충돌이 있는지 확인
    has_collision = len(all_contact_points) > 0
    
    # 모든 접촉점 디버깅 (로봇과 모든 객체 사이의 충돌)
    if has_collision:
        print(f"로봇 충돌 감지: 접촉점 수={len(all_contact_points)}")
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
                print(f"  -> 파이프와 충돌! (로봇 링크 {linkA} - 파이프)")
        elif bodyB == robot_id:  # 충돌 순서가 반대인 경우도 확인
            collision_links.add(contact[4])  # 로봇 링크 인덱스
            if pipe_id is not None and bodyA == pipe_id:
                pipe_collision = True
                print(f"  -> 파이프와 충돌! (파이프 - 로봇 링크 {contact[4]})")
    
    # 파이프와의 충돌 확인 (파이프 ID가 제공된 경우)
    if pipe_id is not None:
        # 파이프와 로봇 사이의 접촉점만 직접 확인
        pipe_contacts = p.getContactPoints(robot_id, pipe_id)
        
        if len(pipe_contacts) > 0:
            pipe_collision = True
            print(f"파이프 충돌 접촉점 수: {len(pipe_contacts)}")  # 디버깅용
            
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
    
    return has_collision, collision_links, pipe_collision

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

def rewire(tree, near_nodes, new_node, robot_id):
    for node in near_nodes:
        new_cost = new_node['cost'] + get_cost(new_node['config'], node['config'])
        if new_cost < node['cost']:
            is_collision, _, _ = check_collision(robot_id, node['config'])
            if not is_collision:
                node['parent'] = new_node
                node['cost'] = new_cost

def rrt_star_plan(robot_id, start_config, goal_config, max_iterations=5000, step_size=0.05, radius=0.5):
    tree = [{'config': start_config, 'parent': None, 'cost': 0}]
    for iteration in range(max_iterations):
        q_rand = goal_config if random.random() < 0.1 else get_random_configuration(robot_id)
        nearest_node = find_nearest_node(tree, q_rand)
        q_near = nearest_node['config']
        q_new = steer(q_near, q_rand, step_size)
        is_collision, _, _ = check_collision(robot_id, q_new)
        if not is_collision:
            near_nodes = find_near_nodes(tree, q_new, radius)
            best_parent, min_cost = choose_parent(near_nodes, q_new)
            new_node = {'config': q_new, 'parent': best_parent, 'cost': min_cost}
            tree.append(new_node)
            rewire(tree, near_nodes, new_node, robot_id)
            if distance_between_configurations(q_new, goal_config) < step_size:
                path = []
                current = new_node
                while current is not None:
                    path.append(current['config'])
                    current = current['parent']
                return list(reversed(path))
        if iteration % 100 == 0:
            print(f"RRT* 진행 중... {iteration}/{max_iterations}")
    return None

def get_link_position(robot_id, joint_positions):
    """주어진 관절 위치에서 로봇의 끝점(TCP) 위치를 계산"""
    # 관절 상태 설정
    for i, pos in enumerate(joint_positions):
        p.resetJointState(robot_id, i, pos)
    
    # TCP 링크의 상태 가져오기
    tcp_link_state = p.getLinkState(robot_id, p.getNumJoints(robot_id)-1)
    return tcp_link_state[0]  # 위치 반환

def visualize_path(robot_id, path):
    """경로를 시각화"""
    # 이전에 그려진 라인 제거
    p.removeAllUserDebugItems()
    
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
            lineColorRGB=[0, 1, 0],  # 초록색
            lineWidth=2.0,
            lifeTime=0  # 0 = 영구적으로 표시
        )

def reset_simulation(robot_id, pipe_id):
    """시뮬레이션을 초기 상태로 리셋"""
    # 이전에 그려진 경로 제거
    p.removeAllUserDebugItems()
    
    # 로봇의 모든 관절을 초기 위치로 리셋
    for i in range(p.getNumJoints(robot_id)):
        p.resetJointState(robot_id, i, 0.0)
    
    # 파이프 위치 리셋
    p.resetBasePositionAndOrientation(pipe_id, [0, 0.5, -0.50], [0, 0, 0, 1])
    
    # 파이프 색상 초기화
    p.changeVisualShape(pipe_id, -1, rgbaColor=[1, 1, 1, 1])
    
    return True

def highlight_collision_links(robot_id, collision_links, pipe_id=None, pipe_collision=False):
    """충돌한 링크와 파이프를 빨간색으로 강조 표시"""
    # 모든 링크의 색상을 원래대로 복원
    for i in range(p.getNumJoints(robot_id)):
        p.changeVisualShape(robot_id, i, rgbaColor=[1, 1, 1, 1])
    
    # 충돌한 링크를 빨간색으로 표시
    for link_index in collision_links:
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
            
        # 데이터 경로 설정 (pybullet_data 사용하지 않음)
        p.setGravity(0, 0, -9.81)  # 중력을 0으로 설정
        p.setRealTimeSimulation(0)  # 실시간 시뮬레이션 비활성화
        
        # 디버그 시각화 도구 설정
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)  # GUI 활성화
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)  # 마우스 픽킹 활성화
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)  # 그림자 활성화
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)  # 렌더링 활성화
        
        # 사용자 정의 디버그 매개변수 추가
        camera_distance_slider = p.addUserDebugParameter("Camera Distance", 0.1, 5.0, 2.0)
        camera_yaw_slider = p.addUserDebugParameter("Camera Yaw", -180, 180, 0)
        camera_pitch_slider = p.addUserDebugParameter("Camera Pitch", -89, 89, 0)
        reset_camera_button = p.addUserDebugParameter("Reset Camera", 1, 0, 0)
        prev_reset_camera_state = p.readUserDebugParameter(reset_camera_button)

        # 카메라 초기 설정
        camera_distance = 2.0
        camera_yaw = 0
        camera_pitch = 0
        camera_target = [0, 0, 0]
        
        # 초기 카메라 위치 설정
        p.resetDebugVisualizerCamera(
            cameraDistance=camera_distance,
            cameraYaw=camera_yaw,
            cameraPitch=camera_pitch,
            cameraTargetPosition=camera_target
        )
        
        # 카메라 제어 설명
        print("\n카메라 제어 방법:")
        print("- Camera Distance/Yaw/Pitch 슬라이더: 카메라 위치 조절")
        print("- Reset Camera 버튼: 카메라 초기 위치로 리셋")
        print("- 마우스 제어: PyBullet 내장 카메라 제어 사용")

        # GUI 요소 추가
        # 시작 위치 입력
        start_x = p.addUserDebugParameter("Start X", -1.0, 1.0, 0.0)
        start_y = p.addUserDebugParameter("Start Y", -1.0, 1.0, 0.0)
        start_z = p.addUserDebugParameter("Start Z", -1.0, 1.0, 0.0)
        
        # 시작 자세 입력 (롤, 피치, 요)
        start_roll = p.addUserDebugParameter("Start Roll", -3.14, 3.14, 0.0)
        start_pitch = p.addUserDebugParameter("Start Pitch", -3.14, 3.14, 0.0)
        start_yaw = p.addUserDebugParameter("Start Yaw", -3.14, 3.14, 0.0)
        
        # 종료 위치 입력 (독립적인 종료점)
        end_x = p.addUserDebugParameter("End X", -1.0, 1.0, 0.0)
        end_y = p.addUserDebugParameter("End Y", -1.0, 1.0, 0.2)
        end_z = p.addUserDebugParameter("End Z", -1.0, 1.0, -0.3)
        
        # 종료 자세 입력 (롤, 피치, 요)
        end_roll = p.addUserDebugParameter("End Roll", -3.14, 3.14, 0.0)
        end_pitch = p.addUserDebugParameter("End Pitch", -3.14, 3.14, 0.0)
        end_yaw = p.addUserDebugParameter("End Yaw", -3.14, 3.14, 0.0)
        
        # 파이프 위치 조절
        pipe_x = p.addUserDebugParameter("Pipe X", -5.0, 5.0, 0.0)
        pipe_y = p.addUserDebugParameter("Pipe Y", -5.0, 5.0, 1.65)
        pipe_z = p.addUserDebugParameter("Pipe Z", -5.0, 5.0, 1.10)
        
        # 파이프 자세 조절 (롤, 피치, 요)
        pipe_roll = p.addUserDebugParameter("Pipe Roll", -3.14, 3.14, 1.65)  # 기본값 X축 90도 회전
        pipe_pitch = p.addUserDebugParameter("Pipe Pitch", -3.14, 3.14, 0.13)
        pipe_yaw = p.addUserDebugParameter("Pipe Yaw", -3.14, 3.14, 0.13)
        
        # 시뮬레이션 속도 조절
        speed_slider = p.addUserDebugParameter("Simulation Speed", 0.1, 2.0, 1.0)
        
        # 시작 버튼 (0 = 시작 안 됨, 1 = 시작)
        start_button = p.addUserDebugParameter("Start Simulation", 1, 0, 0)
        
        # 초기화 버튼 (0 = 초기화 안 됨, 1 = 초기화)
        reset_button = p.addUserDebugParameter("Reset Simulation", 1, 0, 0)
        
        # 이전 버튼 상태 저장
        previous_button_state = p.readUserDebugParameter(start_button)
        previous_reset_state = p.readUserDebugParameter(reset_button)

        # 현재 디렉토리 가져오기
        current_dir = get_data_path()
        print(f"현재 디렉토리: {current_dir}")

        # PLY 파일을 OBJ로 변환하고 컨벡스 분해 수행
        ply_path = os.path.join(current_dir, "pipe.ply")
        pipe_id = None
        
        if not os.path.exists(ply_path):
            print(f"PLY 파일을 찾을 수 없습니다: {ply_path}")
            print("기본 원통형 파이프를 생성합니다.")
            
            # 기본 원통형 파이프 생성
            pipe_radius = 0.03  # 3cm 반지름
            pipe_length = 0.3   # 30cm 길이
            
            visual_shape_id = p.createVisualShape(
                shapeType=p.GEOM_CYLINDER,
                radius=pipe_radius,
                length=pipe_length,
                rgbaColor=[1, 1, 1, 1],  # 흰색
                specularColor=[0.4, 0.4, 0.4],
                visualFramePosition=[0, 0, 0]
            )
            
            collision_shape_id = p.createCollisionShape(
                shapeType=p.GEOM_CYLINDER,
                radius=pipe_radius,
                height=pipe_length,
                collisionFramePosition=[0, 0, 0]
            )
            
            print(f"원통형 파이프 생성: 반지름={pipe_radius}m, 길이={pipe_length}m")
            
            # 파이프의 위치 설정
            pipe_position = [0, 0.5, -0.50]  # X: 0mm, Y: 500mm, Z: -500mm
            
            # 파이프 회전 - X축 방향으로 90도 회전 (가로로 눕힘)
            pipe_orientation = p.getQuaternionFromEuler([math.pi/2, 0, 0])
            
            pipe_id = p.createMultiBody(
                baseMass=0.5,  # 충분한 질량 부여
                baseVisualShapeIndex=visual_shape_id,
                baseCollisionShapeIndex=collision_shape_id,
                basePosition=pipe_position,
                baseOrientation=pipe_orientation
            )
        else:
            # OBJ 파일 경로 설정
            obj_path = os.path.join(current_dir, "pipe.obj")
            convert_ply_to_obj(ply_path, obj_path)
            
            # 메시 스케일 설정
            mesh_scale = [0.001, 0.001, 0.001]  # mm 단위를 m 단위로 변환
            
            # 컨벡스 분해 수행
            decomposed_obj_path = perform_convex_decomposition(obj_path)
            
            print(f"컨벡스 분해된 OBJ 파일: {decomposed_obj_path}")
            
            # OBJ 파일 로드 - 고품질 시각적 형상
            visual_shape_id = p.createVisualShape(
                shapeType=p.GEOM_MESH,
                fileName=obj_path,  # 원본 메시를 시각적 형상으로 사용
                rgbaColor=[1, 1, 1, 1],
                specularColor=[0.4, 0.4, 0.4],
                visualFramePosition=[0, 0, 0],
                meshScale=mesh_scale
            )
            
            # 컨벡스 분해된 충돌 형상 생성
            collision_shape_id = p.createCollisionShape(
                shapeType=p.GEOM_MESH,
                fileName=decomposed_obj_path,  # 컨벡스 분해된 메시를 충돌 형상으로 사용
                collisionFramePosition=[0, 0, 0],
                meshScale=mesh_scale
            )
            
            print(f"파이프 시각적 형상: 원본 OBJ 메시, 충돌 형상: 컨벡스 분해 메시 (스케일={mesh_scale[0]})")
            
            # 파이프의 위치 설정
            pipe_position = [0, 0.5, -0.50]
            
            # 파이프 회전 - X축 방향으로 90도 회전 (가로로 눕힘)
            pipe_orientation = p.getQuaternionFromEuler([math.pi/2, 0, 0])
            
            # 파이프의 충돌 플래그 설정
            collision_flags = p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION
            
            pipe_id = p.createMultiBody(
                baseMass=0.5,  # 충분한 질량 부여
                baseVisualShapeIndex=visual_shape_id,
                baseCollisionShapeIndex=collision_shape_id,
                basePosition=pipe_position,
                baseOrientation=pipe_orientation,
                flags=collision_flags
            )
        
        # 파이프의 충돌 속성 설정
        p.changeDynamics(
            pipe_id, 
            -1,  # 베이스 링크
            contactStiffness=50000.0,
            contactDamping=1000.0,
            restitution=0.01,
            lateralFriction=0.5,
            collisionMargin=0.0001  # 충돌 마진 최소화
        )
        
        print(f"파이프 객체 ID: {pipe_id}")
        
        # 고품질 와이어프레임 모드 설정
        p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 1)
        
        # 추가적인 디버그 옵션 활성화
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)  # 그림자 비활성화
        
        # 컨벡스 분해된 파이프 충돌 형상 표시 디버그 텍스트
        p.addUserDebugText(
            "컨벡스 분해 파이프 충돌 형상",
            [pipe_position[0], pipe_position[1], pipe_position[2] + 0.1],
            textColorRGB=[1, 0, 0],
            textSize=1.5
        )
        
        # 컨벡스 헐 시각화 활성화
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 1)
        
        # 키 입력 정보 추가
        p.addUserDebugText(
            "W 키: 와이어프레임 모드 전환",
            [0, 0, 0.5],
            textColorRGB=[0, 1, 0],
            textSize=1.5
        )

        # 로봇 로드 (절대 경로 사용)
        robot_path = os.path.join(current_dir, "rb10_1300e.urdf")
        if not os.path.exists(robot_path):
            raise Exception(f"Robot URDF 파일을 찾을 수 없습니다: {robot_path}")
        
        # URDF 파일의 메시 경로 수정
        print("URDF 파일의 메시 경로를 절대 경로로 수정 중...")
        modified_urdf_path = os.path.join(current_dir, "rb10_1300e_modified.urdf")
        modified_urdf_path = modify_urdf_mesh_paths(robot_path, modified_urdf_path)
        print(f"수정된 URDF 파일 경로: {modified_urdf_path}")
        
        # 로봇을 거꾸로 배치하기 위한 회전 각도 (180도 회전)
        robot_orientation = p.getQuaternionFromEuler([0, 0, math.pi])
        
        # 로봇의 초기 위치를 좌표계 원점으로 설정
        robot_position = [0, 0, 0]  # X: 0mm, Y: 0mm, Z: 0mm
        
        # 절대 경로로 로봇 로드
        try:
            robot_id = p.loadURDF(
                modified_urdf_path,
                basePosition=robot_position,
                baseOrientation=robot_orientation,
                useFixedBase=True
            )
            if robot_id < 0:
                raise Exception("Robot URDF 로드 실패")
            print("로봇이 성공적으로 로드되었습니다!")
        except p.error as e:
            print(f"로봇 로드 중 오류 발생: {e}")
            raise

        # 로봇의 관절 정보 저장
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

        print("\n시뮬레이션 준비가 완료되었습니다.")
        print("1. 시작 위치(Start X/Y/Z)를 설정하세요.")        
        print("2. 파이프 위치(Pipe X/Y/Z)를 조절하여 원하는 위치로 이동하세요.")
        print("3. 'Start Simulation' 버튼을 클릭하여 시뮬레이션을 시작하세요.")
        print("4. 'Reset Simulation' 버튼을 클릭하여 시뮬레이션을 초기화할 수 있습니다.")
        print("5. 시뮬레이션 속도는 'Simulation Speed' 슬라이더로 조절할 수 있습니다.")
        print("6. 충돌이 발생하면 해당 부위가 빨간색으로 표시됩니다.")
        print("7. 마우스 중간 버튼을 누른 상태에서 마우스를 움직이면 카메라가 회전합니다.")
        print("8. Ctrl+C를 눌러 종료할 수 있습니다.\n")

        path = None
        simulation_started = False
        collision_detected = False

        # 시뮬레이션 루프
        while p.isConnected():
            try:
                # 카메라 리셋 버튼 확인
                current_reset_camera_state = p.readUserDebugParameter(reset_camera_button)
                if current_reset_camera_state != prev_reset_camera_state:
                    prev_reset_camera_state = current_reset_camera_state
                    p.resetDebugVisualizerCamera(
                        cameraDistance=camera_distance,
                        cameraYaw=camera_yaw,
                        cameraPitch=camera_pitch,
                        cameraTargetPosition=camera_target
                    )
                    print("카메라 위치가 초기화되었습니다.")
                
                # 카메라 슬라이더 값 읽기 및 적용
                current_distance = p.readUserDebugParameter(camera_distance_slider)
                current_yaw = p.readUserDebugParameter(camera_yaw_slider)
                current_pitch = p.readUserDebugParameter(camera_pitch_slider)
                
                # 카메라 위치 업데이트 (슬라이더 값이 변경된 경우에만)
                if (current_distance != camera_distance or 
                    current_yaw != camera_yaw or 
                    current_pitch != camera_pitch):
                    
                    camera_distance = current_distance
                    camera_yaw = current_yaw
                    camera_pitch = current_pitch
                    
                    p.resetDebugVisualizerCamera(
                        cameraDistance=camera_distance,
                        cameraYaw=camera_yaw,
                        cameraPitch=camera_pitch,
                        cameraTargetPosition=camera_target
                    )
                
                # 키보드 입력 처리 (스페이스 키로 카메라 초기화)
                if keyboard.is_pressed(' '):
                    p.resetDebugVisualizerCamera(
                        cameraDistance=camera_distance,
                        cameraYaw=camera_yaw,
                        cameraPitch=camera_pitch,
                        cameraTargetPosition=camera_target
                    )
                    print("카메라 위치가 초기화되었습니다.")
                    # 키 중복 인식 방지를 위한 짧은 대기
                    time.sleep(0.2)
                
                # 버튼 상태 확인
                current_button_state = p.readUserDebugParameter(start_button)
                current_reset_state = p.readUserDebugParameter(reset_button)
                
                # 파이프 위치 업데이트
                pipe_pos = [
                    p.readUserDebugParameter(pipe_x),
                    p.readUserDebugParameter(pipe_y),
                    p.readUserDebugParameter(pipe_z)
                ]
                
                # 파이프 자세 업데이트 (RPY -> 쿼터니언)
                pipe_rpy = [
                    p.readUserDebugParameter(pipe_roll),
                    p.readUserDebugParameter(pipe_pitch),
                    p.readUserDebugParameter(pipe_yaw)
                ]
                pipe_orientation = p.getQuaternionFromEuler(pipe_rpy)
                
                p.resetBasePositionAndOrientation(pipe_id, pipe_pos, pipe_orientation)
                
                # 지속적인 충돌 검사 (디버깅용)
                debug_contacts = p.getContactPoints(robot_id, pipe_id)
                if len(debug_contacts) > 0:
                    print(f"[디버그] 파이프-로봇 접촉점: {len(debug_contacts)}")
                
                # 초기화 버튼이 눌렸을 때
                if current_reset_state != previous_reset_state:
                    previous_reset_state = current_reset_state
                    if reset_simulation(robot_id, pipe_id):
                        print("\n시뮬레이션이 초기화되었습니다.")
                        print("새로운 시작 위치를 설정하고 파이프 위치를 조절하세요.")
                        simulation_started = False
                        path = None
                        collision_detected = False
                        # 모든 링크의 색상을 원래대로 복원
                        for i in range(p.getNumJoints(robot_id)):
                            p.changeVisualShape(robot_id, i, rgbaColor=[1, 1, 1, 1])
                        continue
                
                # 시작 버튼이 눌렸을 때
                if current_button_state != previous_button_state:
                    previous_button_state = current_button_state
                    
                    if not simulation_started:
                        # 시작 위치 읽기
                        start_pos = [
                            p.readUserDebugParameter(start_x),
                            p.readUserDebugParameter(start_y),
                            p.readUserDebugParameter(start_z)
                        ]
                        
                        # 시작 자세 읽기 (RPY -> 쿼터니언)
                        start_rpy = [
                            p.readUserDebugParameter(start_roll),
                            p.readUserDebugParameter(start_pitch),
                            p.readUserDebugParameter(start_yaw)
                        ]
                        start_orientation = p.getQuaternionFromEuler(start_rpy)
                        
                        # 종료 위치 읽기
                        end_pos = [
                            p.readUserDebugParameter(end_x),
                            p.readUserDebugParameter(end_y),
                            p.readUserDebugParameter(end_z)
                        ]
                        
                        # 종료 자세 읽기 (RPY -> 쿼터니언)
                        end_rpy = [
                            p.readUserDebugParameter(end_roll),
                            p.readUserDebugParameter(end_pitch),
                            p.readUserDebugParameter(end_yaw)
                        ]
                        end_orientation = p.getQuaternionFromEuler(end_rpy)
                        
                        # 종료 위치를 시각화
                        visualize_position_id = p.addUserDebugLine(
                            end_pos,
                            [end_pos[0], end_pos[1], end_pos[2] + 0.1],
                            lineColorRGB=[0, 0, 1],
                            lineWidth=2.0,
                            lifeTime=0
                        )
                        
                        print("\n경로 생성 중...")
                        print(f"시작 위치: {start_pos}, 자세(RPY): {start_rpy}")
                        print(f"종료 위치: {end_pos}, 자세(RPY): {end_rpy}")
                        print(f"파이프 위치: {pipe_pos}, 자세(RPY): {pipe_rpy}")
                        
                        # 시작/종료 위치를 관절 각도로 변환 (역기구학) - 자세 포함
                        start_joints = p.calculateInverseKinematics(
                            robot_id, 
                            p.getNumJoints(robot_id)-1, 
                            start_pos,
                            targetOrientation=start_orientation
                        )
                        
                        end_joints = p.calculateInverseKinematics(
                            robot_id, 
                            p.getNumJoints(robot_id)-1, 
                            end_pos,  # 파이프 위치 대신 종료 위치 사용
                            targetOrientation=end_orientation  # 파이프 방향 대신 종료 방향 사용
                        )
                        
                        # RRT*로 경로 생성 (입력은 시작과 종료시 로봇의 조인트공간 위치)
                        path = rrt_star_plan(robot_id, start_joints, end_joints)
                        
                        if path is None:
                            print("경로를 찾을 수 없습니다. 다른 위치를 시도해보세요.")
                            continue
                            
                        print(f"경로가 생성되었습니다. 경로 길이: {len(path)}")
                        

                        # 생성된 경로 시각화
                        print("End Effector 경로를 시각화합니다...")
                        visualize_path(robot_id, path)
                        

                        # 경로를 CSV 파일로 저장
                        save_joint_trajectory_to_csv(path, "joint_trajectory.csv")
                        

                        simulation_started = True
                        path_index = 0
                        collision_detected = False
                

                # 최종 결과 가시화   
                if simulation_started and path is not None and path_index < len(path) and not collision_detected:
                    # 시뮬레이션 속도 읽기
                    speed = p.readUserDebugParameter(speed_slider)
                    
                    # 현재 경로의 관절 위치로 이동
                    current_config = path[path_index]
                    
                    # 로봇 관절 제어
                    for i, joint in enumerate(robot_joints):
                        p.setJointMotorControl2(
                            bodyIndex=robot_id,
                            jointIndex=joint['index'],
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=current_config[i],
                            force=1000,
                            positionGain=1.0,
                            velocityGain=0.5,
                            maxVelocity=0.5 * speed
                        )
                    
                    # 충돌 검사 (파이프 충돌 확인 포함)
                    has_collision, collision_links, pipe_collision = check_collision(robot_id, current_config, pipe_id)
                    if has_collision:
                        print("\n충돌이 감지되었습니다!")
                        print(f"충돌한 링크: {collision_links}")
                        if pipe_collision:
                            print("파이프와 충돌했습니다!")
                        highlight_collision_links(robot_id, collision_links, pipe_id, pipe_collision)
                        collision_detected = True
                    else:
                        # 충돌이 없으면 파이프 색상 원래대로
                        p.changeVisualShape(pipe_id, -1, rgbaColor=[1, 1, 1, 1])
                        path_index += 1
                
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