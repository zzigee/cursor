#!/usr/bin/env python3
import trimesh
import numpy as np

def analyze_pointcloud(ply_file):
    """PLY 파일을 포인트 클라우드로 분석"""
    print(f"\n{'='*50}")
    print(f"분석: {ply_file}")
    print('='*50)
    
    try:
        # PLY 파일 로드
        mesh = trimesh.load(ply_file)
        
        print(f"타입: {type(mesh)}")
        
        if isinstance(mesh, trimesh.points.PointCloud):
            print("* 포인트 클라우드 형태")
            points = mesh.vertices
            has_colors = hasattr(mesh, 'colors') and mesh.colors is not None
            
        elif hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
            print("* 메시 형태 (vertices + faces)")
            points = mesh.vertices
            has_colors = hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors')
            print(f"  - Faces: {len(mesh.faces):,}개")
            
        elif hasattr(mesh, 'vertices'):
            print("* Vertices만 있는 형태")
            points = mesh.vertices  
            has_colors = False
            
        else:
            print("X 포인트 데이터를 찾을 수 없음")
            return
        
        # 기본 정보
        print(f"포인트 수: {len(points):,}개")
        print(f"색상 정보: {'있음' if has_colors else '없음'}")
        
        # 좌표 범위
        print(f"\n좌표 범위:")
        print(f"  X: {points[:, 0].min():8.6f} ~ {points[:, 0].max():8.6f} (폭: {points[:, 0].max()-points[:, 0].min():8.6f})")
        print(f"  Y: {points[:, 1].min():8.6f} ~ {points[:, 1].max():8.6f} (폭: {points[:, 1].max()-points[:, 1].min():8.6f})")
        print(f"  Z: {points[:, 2].min():8.6f} ~ {points[:, 2].max():8.6f} (폭: {points[:, 2].max()-points[:, 2].min():8.6f})")
        
        # 바운딩 박스
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        bbox_size = bbox_max - bbox_min
        bbox_center = (bbox_min + bbox_max) / 2
        
        print(f"\n바운딩 박스:")
        print(f"  크기: [{bbox_size[0]:8.6f}, {bbox_size[1]:8.6f}, {bbox_size[2]:8.6f}]")
        print(f"  중심: [{bbox_center[0]:8.6f}, {bbox_center[1]:8.6f}, {bbox_center[2]:8.6f}]")
        
        # 포인트 밀도 분석
        bbox_volume = np.prod(bbox_size)
        if bbox_volume > 0:
            density = len(points) / bbox_volume
            print(f"  포인트 밀도: {density:,.0f} 포인트/단위부피")
        
        # 샘플 포인트 출력 (처음 5개)
        print(f"\n샘플 포인트 (처음 5개):")
        for i in range(min(5, len(points))):
            x, y, z = points[i]
            print(f"  [{i:2d}]: ({x:8.6f}, {y:8.6f}, {z:8.6f})")
        
        # 단위 추정
        max_coord = np.abs(points).max()
        if max_coord > 1000:
            print(f"\n추정 단위: mm (최대 좌표값: {max_coord:.1f})")
        elif max_coord > 10:
            print(f"\n추정 단위: cm (최대 좌표값: {max_coord:.1f})")  
        else:
            print(f"\n추정 단위: m (최대 좌표값: {max_coord:.1f})")
            
        return points
        
    except Exception as e:
        print(f"X 오류 발생: {e}")
        return None

if __name__ == "__main__":
    print("PLY 파일 포인트 클라우드 분석")
    
    # pipe.ply 분석
    points1 = analyze_pointcloud('pipe.ply')
    
    # pipe_1.ply 분석  
    points2 = analyze_pointcloud('pipe_1.ply')
    
    # 비교
    if points1 is not None and points2 is not None:
        print(f"\n{'='*50}")
        print("비교 분석")
        print('='*50)
        print(f"포인트 수: pipe.ply ({len(points1):,}) vs pipe_1.ply ({len(points2):,})")
        ratio = len(points2) / len(points1)
        print(f"비율: pipe_1.ply가 pipe.ply보다 {ratio:.1f}배 많음")
        
        # 바운딩 박스 비교
        bbox1 = points1.max(axis=0) - points1.min(axis=0)
        bbox2 = points2.max(axis=0) - points2.min(axis=0)
        print(f"\n바운딩 박스 비교:")
        print(f"  pipe.ply  : [{bbox1[0]:8.6f}, {bbox1[1]:8.6f}, {bbox1[2]:8.6f}]")
        print(f"  pipe_1.ply: [{bbox2[0]:8.6f}, {bbox2[1]:8.6f}, {bbox2[2]:8.6f}]")