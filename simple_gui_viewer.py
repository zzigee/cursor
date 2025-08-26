#!/usr/bin/env python3
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('TkAgg')  # TkAgg 백엔드 사용

def plot_pointcloud_interactive(filename):
    """대화형 3D 포인트 클라우드 뷰어"""
    
    print(f"Loading {filename}...")
    
    try:
        # PLY 파일 로드
        mesh = trimesh.load(filename)
        
        if isinstance(mesh, trimesh.points.PointCloud):
            points = mesh.vertices
            print(f"Point cloud: {len(points):,} points")
        elif hasattr(mesh, 'vertices'):
            points = mesh.vertices
            print(f"Mesh: {len(points):,} vertices")
        else:
            print("No point data found")
            return
            
        # 다운샘플링 (50K 포인트로 제한)
        if len(points) > 50000:
            print(f"Downsampling: {len(points):,} -> 50,000 points")
            indices = np.random.choice(len(points), 50000, replace=False)
            points = points[indices]
        
        # 기본 정보 출력
        print(f"X range: {points[:,0].min():.6f} ~ {points[:,0].max():.6f}")
        print(f"Y range: {points[:,1].min():.6f} ~ {points[:,1].max():.6f}")  
        print(f"Z range: {points[:,2].min():.6f} ~ {points[:,2].max():.6f}")
        
        size = points.max(axis=0) - points.min(axis=0)
        print(f"Size: {size[0]:.6f} x {size[1]:.6f} x {size[2]:.6f}")
        
        # 3D 플롯 생성
        fig = plt.figure(figsize=(14, 10))
        
        # 메인 3D 뷰
        ax1 = fig.add_subplot(221, projection='3d')
        scatter1 = ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
                             c=points[:, 2], cmap='viridis', s=0.5, alpha=0.7)
        ax1.set_title('3D View (Z-colored)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')  
        ax1.set_zlabel('Z')
        
        # XY 투영
        ax2 = fig.add_subplot(222)
        ax2.scatter(points[:, 0], points[:, 1], c=points[:, 2], cmap='viridis', s=0.5, alpha=0.7)
        ax2.set_title('XY Projection')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_aspect('equal')
        
        # XZ 투영  
        ax3 = fig.add_subplot(223)
        ax3.scatter(points[:, 0], points[:, 2], c=points[:, 1], cmap='plasma', s=0.5, alpha=0.7)
        ax3.set_title('XZ Projection')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Z')
        ax3.set_aspect('equal')
        
        # YZ 투영
        ax4 = fig.add_subplot(224)
        ax4.scatter(points[:, 1], points[:, 2], c=points[:, 0], cmap='coolwarm', s=0.5, alpha=0.7)
        ax4.set_title('YZ Projection')
        ax4.set_xlabel('Y')
        ax4.set_ylabel('Z')
        ax4.set_aspect('equal')
        
        # 컬러바
        plt.colorbar(scatter1, ax=ax1, shrink=0.6)
        
        # 전체 제목
        fig.suptitle(f'Point Cloud: {filename} ({len(points):,} points)', fontsize=14)
        
        # 레이아웃 조정
        plt.tight_layout()
        
        # 사용법 출력
        print("\n=== GUI Controls ===")
        print("- Mouse: Rotate 3D view")
        print("- Mouse wheel: Zoom")
        print("- Right click + drag: Pan")
        print("- Close window to continue")
        
        # 표시
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """메인 함수 - 파일 선택"""
    import os
    
    # PLY 파일 목록
    ply_files = [f for f in os.listdir('.') if f.endswith('.ply')]
    
    if not ply_files:
        print("No PLY files found in current directory")
        return
        
    print("Available PLY files:")
    for i, f in enumerate(ply_files):
        file_size = os.path.getsize(f) / 1024 / 1024  # MB
        print(f"{i+1}: {f} ({file_size:.1f} MB)")
        
    while True:
        try:
            choice = input(f"\nSelect file (1-{len(ply_files)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                break
                
            idx = int(choice) - 1
            if 0 <= idx < len(ply_files):
                filename = ply_files[idx]
                print(f"\n{'='*50}")
                plot_pointcloud_interactive(filename)
                print(f"{'='*50}")
            else:
                print("Invalid selection")
                
        except (ValueError, KeyboardInterrupt):
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()