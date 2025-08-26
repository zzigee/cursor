#!/usr/bin/env python3
import trimesh
import numpy as np
import os

def simple_analyze(filename):
    """간단한 포인트 클라우드 분석"""
    if not os.path.exists(filename):
        print(f"{filename} 파일이 존재하지 않습니다.")
        return
        
    print(f"\n{'='*50}")
    print(f"파일: {filename}")
    print('='*50)
    
    try:
        mesh = trimesh.load(filename)
        
        if isinstance(mesh, trimesh.points.PointCloud):
            points = mesh.vertices
            print(f"* 포인트 클라우드: {len(points):,}개 포인트")
            
        elif hasattr(mesh, 'vertices'):
            points = mesh.vertices
            print(f"* 메시: {len(points):,}개 vertices")
            if hasattr(mesh, 'faces'):
                print(f"  + {len(mesh.faces):,}개 faces")
        else:
            print("X 포인트 데이터 없음")
            return
            
        # 기본 통계
        print(f"\n좌표 범위:")
        print(f"  X: {points[:,0].min():.6f} ~ {points[:,0].max():.6f}")
        print(f"  Y: {points[:,1].min():.6f} ~ {points[:,1].max():.6f}")
        print(f"  Z: {points[:,2].min():.6f} ~ {points[:,2].max():.6f}")
        
        # 크기
        size = points.max(axis=0) - points.min(axis=0)
        print(f"\n크기: {size[0]:.6f} × {size[1]:.6f} × {size[2]:.6f}")
        
        # 중심점
        center = (points.max(axis=0) + points.min(axis=0)) / 2
        print(f"중심: ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})")
        
        # 파일 크기
        file_size = os.path.getsize(filename)
        print(f"파일 크기: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        
        return points
        
    except Exception as e:
        print(f"X 오류: {e}")
        return None

# 포인트 클라우드 시각화 (텍스트)
def text_visualization(points, filename):
    """텍스트로 포인트 분포 시각화"""
    print(f"\n{filename} 포인트 분포 (XY 평면 투영):")
    print("-" * 40)
    
    # XY 평면으로 투영
    x = points[:, 0]
    y = points[:, 1]
    
    # 격자 생성 (20x10)
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    grid_x = 20
    grid_y = 10
    
    grid = np.zeros((grid_y, grid_x))
    
    # 포인트를 격자에 매핑
    for i in range(len(points)):
        if len(points) > 50000 and i % (len(points) // 50000) != 0:
            continue  # 샘플링
            
        px, py = x[i], y[i]
        gx = int((px - x_min) / (x_max - x_min) * (grid_x - 1))
        gy = int((py - y_min) / (y_max - y_min) * (grid_y - 1))
        
        gx = max(0, min(grid_x - 1, gx))
        gy = max(0, min(grid_y - 1, gy))
        
        grid[gy, gx] += 1
    
    # 정규화 및 출력
    max_count = grid.max()
    if max_count > 0:
        chars = " .:-=+*#%@"
        for row in grid:
            line = ""
            for count in row:
                char_idx = int(count / max_count * (len(chars) - 1))
                line += chars[char_idx]
            print(line)
    
    print(f"X: {x_min:.3f} ~ {x_max:.3f}")
    print(f"Y: {y_min:.3f} ~ {y_max:.3f}")

if __name__ == "__main__":
    print("파이프 포인트 클라우드 간단 분석")
    
    files = ['pipe_1.ply', 'pipe_2.ply', 'pipe_3.ply']
    
    for filename in files:
        points = simple_analyze(filename)
        if points is not None:
            text_visualization(points, filename)