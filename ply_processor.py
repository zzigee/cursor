"""
PLY 파일 처리를 위한 전용 모듈
PyBullet과 trimesh를 사용하여 PLY 파일을 로드하고 처리합니다.
"""

import os
import math
import tempfile
import pybullet as p
import trimesh


def get_data_path():
    """현재 작업 디렉토리 반환"""
    return os.getcwd()


def convert_ply_to_obj(ply_path, obj_path):
    """
    PLY 파일을 OBJ 파일로 변환
    
    Args:
        ply_path (str): 입력 PLY 파일 경로
        obj_path (str): 출력 OBJ 파일 경로
        
    Returns:
        str: 변환된 OBJ 파일 경로
        
    Raises:
        Exception: PLY 파일 로드 또는 변환 실패시
    """
    try:
        # PLY 파일 로드
        mesh = trimesh.load(ply_path)
        
        # OBJ 파일로 저장
        mesh.export(obj_path)
        
        print(f"PLY → OBJ 변환 완료: {ply_path} → {obj_path}")
        return obj_path
    except Exception as e:
        raise Exception(f"PLY to OBJ 변환 실패: {e}")


def perform_convex_decomposition(mesh_path):
    """
    VHACD를 사용하여 메시의 컨벡스 분해 수행
    
    Args:
        mesh_path (str): 입력 메시 파일 경로 (OBJ)
        
    Returns:
        str: 컨벡스 분해된 메시 파일 경로
    """
    print(f"메시 파일 '{mesh_path}'에 대해 컨벡스 분해 수행 중...")
    
    output_path = mesh_path + "_vhacd.obj"
    
    # VHACD 파라미터 설정 - 처리 시간 단축을 위해 값 조정
    p.vhacd(
        mesh_path,               # 입력 OBJ 파일
        output_path,             # 출력 OBJ 파일
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
    
    print(f"컨벡스 분해 완료: {output_path}")
    return output_path


def load_ply_mesh_as_body(ply_file, scale=1.0, base_pos=[0, 0, 0], base_ori=[0, 0, 0, 1]):
    """
    PLY 파일을 메모리 기반으로 PyBullet body로 로드 (teset.py 스타일)
    임시 OBJ 파일을 생성하여 사용 후 삭제
    
    Args:
        ply_file (str): PLY 파일 경로
        scale (float): 스케일 팩터
        base_pos (list): 위치 [x, y, z]
        base_ori (list): 방향 [x, y, z, w] (quaternion)
        
    Returns:
        int: PyBullet body ID
        
    Raises:
        FileNotFoundError: PLY 파일이 존재하지 않을 때
        Exception: 메시 로드 또는 처리 실패시
    """
    if not os.path.exists(ply_file):
        raise FileNotFoundError(f"PLY 파일 없음: {ply_file}")

    # trimesh 로드
    mesh = trimesh.load(ply_file, force='mesh')

    # watertight 보정
    if not mesh.is_watertight:
        mesh = mesh.convex_hull
    if not mesh.is_winding_consistent:
        mesh = mesh.convex_hull

    # OBJ export 시 자동 삼각형화 옵션
    obj_bytes = mesh.export(file_type="obj")

    # 임시 파일 생성
    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as tmp:
        tmp.write(obj_bytes.encode("utf-8"))
        tmp.flush()
        obj_path = tmp.name

    # PyBullet 충돌/시각화 shape 생성
    collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=obj_path,
        meshScale=[scale, scale, scale]
    )
    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=obj_path,
        meshScale=[scale, scale, scale]
    )

    body_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=base_pos,
        baseOrientation=base_ori
    )

    # 임시 파일 삭제
    os.remove(obj_path)
    return body_id


def load_ply_as_pybullet_body(ply_path, position=[0, 0, 0], orientation=[0, 0, 0, 1], 
                              scale=0.001, mass=0.5, use_convex_decomposition=True,
                              color=[1, 1, 1, 1]):
    """
    PLY 파일을 PyBullet body로 로드하는 통합 함수
    
    Args:
        ply_path (str): PLY 파일 경로
        position (list): 위치 [x, y, z]
        orientation (list): 방향 [x, y, z, w] (quaternion)
        scale (float): 스케일 (기본값: 0.001 - mm을 m로 변환)
        mass (float): 질량 (기본값: 0.5)
        use_convex_decomposition (bool): VHACD 컨벡스 분해 사용 여부
        color (list): RGBA 색상 [r, g, b, a]
        
    Returns:
        int: PyBullet body ID, 실패시 None
    """
    try:
        # PLY 파일 존재 확인
        if not os.path.exists(ply_path):
            print(f"PLY 파일을 찾을 수 없습니다: {ply_path}")
            return None
        
        print(f"PLY 파일 로드 중: {ply_path}")
        
        # PLY → OBJ 변환
        base_name = os.path.splitext(os.path.basename(ply_path))[0]
        current_dir = os.path.dirname(ply_path) if os.path.dirname(ply_path) else get_data_path()
        obj_path = os.path.join(current_dir, f"{base_name}.obj")
        
        # PLY를 OBJ로 변환
        convert_ply_to_obj(ply_path, obj_path)
        
        # 메시 스케일 설정
        mesh_scale = [scale, scale, scale]
        
        # 컨벡스 분해 수행 (옵션)
        collision_obj_path = obj_path
        if use_convex_decomposition:
            try:
                collision_obj_path = perform_convex_decomposition(obj_path)
                print(f"컨벡스 분해 완료: {collision_obj_path}")
            except Exception as e:
                print(f"컨벡스 분해 실패, 원본 메시 사용: {e}")
                collision_obj_path = obj_path
        
        # 시각적 형상 생성 (원본 메시 사용)
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=obj_path,
            rgbaColor=color,
            specularColor=[0.4, 0.4, 0.4],
            visualFramePosition=[0, 0, 0],
            meshScale=mesh_scale
        )
        
        # 충돌 형상 생성 (컨벡스 분해된 메시 또는 원본 메시)
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=collision_obj_path,
            collisionFramePosition=[0, 0, 0],
            meshScale=mesh_scale
        )
        
        # 충돌 플래그 설정
        collision_flags = p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION
        
        # MultiBody 생성
        body_id = p.createMultiBody(
            baseMass=mass,
            baseVisualShapeIndex=visual_shape_id,
            baseCollisionShapeIndex=collision_shape_id,
            basePosition=position,
            baseOrientation=orientation,
            flags=collision_flags
        )
        
        # 충돌 속성 설정
        p.changeDynamics(
            body_id, 
            -1,  # 베이스 링크
            contactStiffness=50000.0,
            contactDamping=1000.0,
            restitution=0.01,
            lateralFriction=0.5,
            collisionMargin=0.0001
        )
        
        print(f"PLY 파일 로드 완료: body_id={body_id}, 위치={position}, 스케일={scale}")
        return body_id
        
    except Exception as e:
        print(f"PLY 파일 로드 실패 ({ply_path}): {e}")
        return None


def create_default_pipe(position=[0, 0.5, -0.50], orientation=None, radius=0.03, length=0.3, 
                       mass=0.5, color=[1, 1, 1, 1]):
    """
    PLY 파일이 없을 때 기본 원통형 파이프 생성
    
    Args:
        position (list): 위치 [x, y, z]
        orientation (list): 방향 (None시 X축 90도 회전)
        radius (float): 반지름 (기본값: 0.03m)
        length (float): 길이 (기본값: 0.3m)
        mass (float): 질량 (기본값: 0.5)
        color (list): RGBA 색상 [r, g, b, a]
        
    Returns:
        int: PyBullet body ID
    """
    if orientation is None:
        orientation = p.getQuaternionFromEuler([math.pi/2, 0, 0])  # X축 90도 회전
    
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_CYLINDER,
        radius=radius,
        length=length,
        rgbaColor=color,
        specularColor=[0.4, 0.4, 0.4],
        visualFramePosition=[0, 0, 0]
    )
    
    collision_shape_id = p.createCollisionShape(
        shapeType=p.GEOM_CYLINDER,
        radius=radius,
        height=length,
        collisionFramePosition=[0, 0, 0]
    )
    
    body_id = p.createMultiBody(
        baseMass=mass,
        baseVisualShapeIndex=visual_shape_id,
        baseCollisionShapeIndex=collision_shape_id,
        basePosition=position,
        baseOrientation=orientation
    )
    
    print(f"기본 원통형 파이프 생성: 반지름={radius}m, 길이={length}m, 위치={position}")
    return body_id


def load_pipe_from_ply(ply_filename="pipe.ply", fallback_to_default=True, 
                      position=[0, 0.5, -0.50], scale=0.001, mass=0.5):
    """
    PLY 파일에서 파이프를 로드하거나 기본 파이프 생성
    
    Args:
        ply_filename (str): PLY 파일명
        fallback_to_default (bool): PLY 실패시 기본 파이프 생성 여부
        position (list): 위치 [x, y, z]
        scale (float): 스케일 (mm → m 변환용)
        mass (float): 질량
        
    Returns:
        int: PyBullet body ID, 실패시 None
    """
    current_dir = get_data_path()
    ply_path = os.path.join(current_dir, ply_filename)
    
    # PLY 파일로 시도
    pipe_id = load_ply_as_pybullet_body(
        ply_path=ply_path,
        position=position,
        orientation=p.getQuaternionFromEuler([math.pi/2, 0, 0]),
        scale=scale,
        mass=mass,
        use_convex_decomposition=True
    )
    
    # PLY 로드 실패시 기본 파이프 생성
    if pipe_id is None and fallback_to_default:
        print("PLY 파일 로드 실패, 기본 원통형 파이프를 생성합니다.")
        pipe_id = create_default_pipe(position=position, mass=mass)
    
    return pipe_id


def load_multiple_ply_objects(ply_configs):
    """
    여러 PLY 객체를 한번에 로드
    
    Args:
        ply_configs (list): PLY 설정 딕셔너리 리스트
            각 딕셔너리는 다음 키를 포함:
            - 'path' (str): PLY 파일 경로
            - 'position' (list, optional): 위치 [x, y, z]
            - 'orientation' (list, optional): 방향 [x, y, z, w]
            - 'scale' (float, optional): 스케일
            - 'mass' (float, optional): 질량
            - 'color' (list, optional): 색상 [r, g, b, a]
            
    Returns:
        list: PyBullet body ID 리스트
    """
    body_ids = []
    
    for config in ply_configs:
        ply_path = config.get('path')
        position = config.get('position', [0, 0, 0])
        orientation = config.get('orientation', [0, 0, 0, 1])
        scale = config.get('scale', 0.001)
        mass = config.get('mass', 0.5)
        color = config.get('color', [1, 1, 1, 1])
        
        body_id = load_ply_as_pybullet_body(
            ply_path=ply_path,
            position=position,
            orientation=orientation,
            scale=scale,
            mass=mass,
            color=color
        )
        
        if body_id is not None:
            body_ids.append(body_id)
        else:
            print(f"PLY 객체 로드 실패: {ply_path}")
    
    return body_ids


# 편의 함수들
def load_ply_simple(ply_path, position=None, scale=1.0):
    """간단한 PLY 로드 (메모리 기반, 임시 파일 사용)"""
    if position is None:
        position = [0, 0, 0]
    return load_ply_mesh_as_body(ply_path, scale=scale, base_pos=position)


def load_ply_advanced(ply_path, position=None, scale=0.001, use_vhacd=True):
    """고급 PLY 로드 (파일 기반, VHACD 옵션)"""
    if position is None:
        position = [0, 0, 0]
    return load_ply_as_pybullet_body(
        ply_path, position=position, scale=scale, 
        use_convex_decomposition=use_vhacd
    )
