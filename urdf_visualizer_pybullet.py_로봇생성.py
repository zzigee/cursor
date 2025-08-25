import pybullet as p
import pybullet_data
import time
import os

# PyBullet 초기화
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 현재 디렉토리 가져오기
current_dir = os.path.dirname(os.path.abspath(__file__))

# URDF 로드
robot_id = p.loadURDF(os.path.join(current_dir, "rb10_1300e.urdf"), useFixedBase=True)

# 카메라 설정
p.resetDebugVisualizerCamera(
    cameraDistance=2.0,
    cameraYaw=45,
    cameraPitch=-30,
    cameraTargetPosition=[0, 0, 0.5]
)

# 중력 설정
p.setGravity(0, 0, -9.81)

# 시뮬레이션 루프
try:
    while True:
        # 시뮬레이션 스텝 진행
        p.stepSimulation()
        
        # 각 관절의 상태 출력
        for joint in range(p.getNumJoints(robot_id)):
            joint_info = p.getJointInfo(robot_id, joint)
            joint_state = p.getJointState(robot_id, joint)
            print(f"Joint {joint_info[1]}: Position = {joint_state[0]:.2f}, Velocity = {joint_state[1]:.2f}")
        
        # 충돌 체크
        contact_points = p.getContactPoints(bodyA=robot_id)
        if contact_points:
            print("충돌 발생!")
        
        time.sleep(1./240.)  # 240Hz로 시뮬레이션 실행

except KeyboardInterrupt:
    print("\n시뮬레이션 종료...")
finally:
    p.disconnect()