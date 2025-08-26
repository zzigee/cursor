#!/usr/bin/env python3
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('TkAgg')  # TkAgg 백엔드 사용
import os

def show_pipe2_gui():
    """pipe_2.ply를 GUI로 표시"""
    
    filename = 'pipe_2.ply'
    
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return
    
    print(f"Loading {filename}...")
    
    try:
        # PLY 파일 로드
        mesh = trimesh.load(filename)
        
        if isinstance(mesh, trimesh.points.PointCloud):
            points = mesh.vertices
            print(f"* Point cloud: {len(points):,} points")
        elif hasattr(mesh, 'vertices'):
            points = mesh.vertices
            print(f"* Mesh: {len(points):,} vertices")
            if hasattr(mesh, 'faces'):
                print(f"  + {len(mesh.faces):,} faces")
        else:
            print("X No point data found")
            return
            
        # 기본 정보 출력
        print(f"\n=== Point Cloud Info ===")
        print(f"X range: {points[:,0].min():8.6f} ~ {points[:,0].max():8.6f}")
        print(f"Y range: {points[:,1].min():8.6f} ~ {points[:,1].max():8.6f}")  
        print(f"Z range: {points[:,2].min():8.6f} ~ {points[:,2].max():8.6f}")
        
        size = points.max(axis=0) - points.min(axis=0)
        center = (points.max(axis=0) + points.min(axis=0)) / 2
        print(f"Size: {size[0]:.6f} × {size[1]:.6f} × {size[2]:.6f}")
        print(f"Center: ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})")
        
        # 다운샘플링 (GUI 성능을 위해)
        display_points = points
        if len(points) > 30000:
            print(f"\nDownsampling for display: {len(points):,} -> 30,000 points")
            indices = np.random.choice(len(points), 30000, replace=False)
            display_points = points[indices]
        
        # 멀티뷰 3D 플롯 생성
        fig = plt.figure(figsize=(16, 12))
        
        # 1. 메인 3D 뷰 (Z축으로 색상)
        ax1 = fig.add_subplot(231, projection='3d')
        scatter1 = ax1.scatter(display_points[:, 0], display_points[:, 1], display_points[:, 2], 
                             c=display_points[:, 2], cmap='viridis', s=1, alpha=0.8)
        ax1.set_title('3D View (Z-colored)', fontsize=12)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')  
        ax1.set_zlabel('Z (m)')
        
        # 2. 다른 각도의 3D 뷰 (Y축으로 색상)
        ax2 = fig.add_subplot(232, projection='3d')
        scatter2 = ax2.scatter(display_points[:, 0], display_points[:, 1], display_points[:, 2], 
                             c=display_points[:, 1], cmap='plasma', s=1, alpha=0.8)
        ax2.set_title('3D View (Y-colored)', fontsize=12)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')  
        ax2.set_zlabel('Z (m)')
        ax2.view_init(elev=30, azim=60)  # 다른 각도
        
        # 3. XY 평면 투영
        ax3 = fig.add_subplot(233)
        scatter3 = ax3.scatter(display_points[:, 0], display_points[:, 1], 
                             c=display_points[:, 2], cmap='viridis', s=1, alpha=0.7)
        ax3.set_title('XY Projection (Z-colored)', fontsize=12)
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)
        
        # 4. XZ 평면 투영  
        ax4 = fig.add_subplot(234)
        scatter4 = ax4.scatter(display_points[:, 0], display_points[:, 2], 
                             c=display_points[:, 1], cmap='plasma', s=1, alpha=0.7)
        ax4.set_title('XZ Projection (Y-colored)', fontsize=12)
        ax4.set_xlabel('X (m)')
        ax4.set_ylabel('Z (m)')
        ax4.set_aspect('equal')
        ax4.grid(True, alpha=0.3)
        
        # 5. YZ 평면 투영
        ax5 = fig.add_subplot(235)
        scatter5 = ax5.scatter(display_points[:, 1], display_points[:, 2], 
                             c=display_points[:, 0], cmap='coolwarm', s=1, alpha=0.7)
        ax5.set_title('YZ Projection (X-colored)', fontsize=12)
        ax5.set_xlabel('Y (m)')
        ax5.set_ylabel('Z (m)')
        ax5.set_aspect('equal')
        ax5.grid(True, alpha=0.3)
        
        # 6. 포인트 밀도 히스토그램
        ax6 = fig.add_subplot(236)
        ax6.hist2d(display_points[:, 0], display_points[:, 1], bins=50, cmap='hot')
        ax6.set_title('XY Density Map', fontsize=12)
        ax6.set_xlabel('X (m)')
        ax6.set_ylabel('Y (m)')
        ax6.set_aspect('equal')
        
        # 컬러바들
        plt.colorbar(scatter1, ax=ax1, shrink=0.6, label='Z (m)')
        plt.colorbar(scatter3, ax=ax3, shrink=0.8, label='Z (m)')
        plt.colorbar(scatter4, ax=ax4, shrink=0.8, label='Y (m)')
        
        # 전체 제목
        fig.suptitle(f'pipe_2.ply Point Cloud Visualization\n'
                    f'{len(display_points):,} points displayed '
                    f'(original: {len(points):,})', 
                    fontsize=16, y=0.95)
        
        # 레이아웃 조정
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        
        # 사용법 출력
        print(f"\n=== GUI Controls ===")
        print("- Mouse: Rotate 3D views")
        print("- Mouse wheel: Zoom in/out")
        print("- Right click + drag: Pan view")
        print("- Close window when done viewing")
        print("\n=== Displaying GUI ===")
        print("Look for the matplotlib window...")
        
        # 윈도우 표시
        plt.show(block=False)  # 논블로킹으로 표시
        
        # 사용자 입력 대기 (간단한 방법)
        try:
            import time
            print("\nGUI window opened. Press Ctrl+C to close or wait 60 seconds...")
            time.sleep(60)  # 60초 대기
        except KeyboardInterrupt:
            print("\nClosing GUI...")
        
        plt.close('all')
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    show_pipe2_gui()