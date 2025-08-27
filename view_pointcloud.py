#!/usr/bin/env python3
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def view_pointcloud(ply_file):
    """PLY 파일을 포인트 클라우드로 시각화"""
    print(f"Loading {ply_file}...")
    
    try:
        # PLY 파일 로드
        mesh = trimesh.load(ply_file)
        
        if isinstance(mesh, trimesh.points.PointCloud):
            print(f"포인트 클라우드: {len(mesh.vertices):,}개 포인트")
            points = mesh.vertices
        elif hasattr(mesh, 'vertices'):
            print(f"메시에서 포인트 추출: {len(mesh.vertices):,}개 vertices")
            points = mesh.vertices
        else:
            print("포인트 데이터를 찾을 수 없습니다.")
            return
        
        # 포인트 좌표 정보
        print(f"X 범위: {points[:, 0].min():.6f} ~ {points[:, 0].max():.6f}")
        print(f"Y 범위: {points[:, 1].min():.6f} ~ {points[:, 1].max():.6f}")
        print(f"Z 범위: {points[:, 2].min():.6f} ~ {points[:, 2].max():.6f}")
        
        # 바운딩 박스 크기
        bbox_size = points.max(axis=0) - points.min(axis=0)
        print(f"바운딩 박스: {bbox_size}")
        
        # 3D 시각화
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 포인트가 너무 많으면 샘플링
        if len(points) > 50000:
            print(f"포인트 다운샘플링: {len(points):,} → 50,000")
            indices = np.random.choice(len(points), 50000, replace=False)
            points_display = points[indices]
        else:
            points_display = points
        
        # 포인트 클라우드 그리기
        scatter = ax.scatter(
            points_display[:, 0], 
            points_display[:, 1], 
            points_display[:, 2],
            c=points_display[:, 2],  # Z값으로 색상 매핑
            cmap='viridis',
            s=1,  # 포인트 크기
            alpha=0.6
        )
        
        # 축 레이블
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Point Cloud: {ply_file}')
        
        # 컬러바
        plt.colorbar(scatter, shrink=0.5, aspect=20)
        
        # 축 비율 맞추기
        ax.set_box_aspect([1,1,1])
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=== PIPE.PLY 포인트 클라우드 ===")
    view_pointcloud('pipe.ply')
    
    print("\n=== PIPE_1.PLY 포인트 클라우드 ===")
    view_pointcloud('pipe_1.ply')