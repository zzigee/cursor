#!/usr/bin/env python3
import trimesh
import numpy as np

def analyze_pipe_pointcloud(ply_file):
    """파이프 PLY 파일을 포인트 클라우드로 분석"""
    print(f"\n{'='*60}")
    print(f"분석: {ply_file}")
    print('='*60)
    
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
            return None
        
        # 기본 정보
        print(f"포인트 수: {len(points):,}개")
        print(f"색상 정보: {'있음' if has_colors else '없음'}")
        
        # 좌표 범위
        print(f"\n좌표 범위:")
        x_range = points[:, 0].max() - points[:, 0].min()
        y_range = points[:, 1].max() - points[:, 1].min()
        z_range = points[:, 2].max() - points[:, 2].min()
        
        print(f"  X: {points[:, 0].min():10.6f} ~ {points[:, 0].max():10.6f} (폭: {x_range:10.6f})")
        print(f"  Y: {points[:, 1].min():10.6f} ~ {points[:, 1].max():10.6f} (폭: {y_range:10.6f})")
        print(f"  Z: {points[:, 2].min():10.6f} ~ {points[:, 2].max():10.6f} (폭: {z_range:10.6f})")
        
        # 바운딩 박스
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        bbox_size = bbox_max - bbox_min
        bbox_center = (bbox_min + bbox_max) / 2
        
        print(f"\n바운딩 박스:")
        print(f"  크기: [{bbox_size[0]:10.6f}, {bbox_size[1]:10.6f}, {bbox_size[2]:10.6f}]")
        print(f"  중심: [{bbox_center[0]:10.6f}, {bbox_center[1]:10.6f}, {bbox_center[2]:10.6f}]")
        
        # 포인트 밀도 분석
        bbox_volume = np.prod(bbox_size)
        if bbox_volume > 0:
            density = len(points) / bbox_volume
            print(f"  포인트 밀도: {density:,.0f} 포인트/단위부피")
        
        # 단위 추정
        max_coord = np.abs(points).max()
        if max_coord > 1000:
            unit = "mm"
            scale_factor = 0.001  # mm to m
        elif max_coord > 10:
            unit = "cm"
            scale_factor = 0.01   # cm to m
        else:
            unit = "m"
            scale_factor = 1.0    # already in m
            
        print(f"\n추정 단위: {unit} (최대 좌표값: {max_coord:.3f})")
        print(f"미터 변환용 스케일: {scale_factor}")
        
        # 형상 분석 (길이 vs 직경)
        dimensions = sorted(bbox_size, reverse=True)
        length = dimensions[0]
        width = dimensions[1]
        thickness = dimensions[2]
        
        print(f"\n형상 분석:")
        print(f"  최대 길이: {length:.6f}")
        print(f"  중간 폭:   {width:.6f}")
        print(f"  최소 두께: {thickness:.6f}")
        
        aspect_ratio = length / width if width > 0 else 0
        if aspect_ratio > 5:
            shape_type = "긴 파이프 형태"
        elif aspect_ratio > 2:
            shape_type = "중간 길이 파이프"
        else:
            shape_type = "정사각형/원형에 가까운 형태"
        print(f"  형태: {shape_type} (길이/폭 비율: {aspect_ratio:.1f})")
        
        # 샘플 포인트 출력 (균등 간격으로)
        print(f"\n샘플 포인트 (균등 간격 10개):")
        sample_indices = np.linspace(0, len(points)-1, min(10, len(points)), dtype=int)
        for i, idx in enumerate(sample_indices):
            x, y, z = points[idx]
            print(f"  [{i:2d}]: ({x:10.6f}, {y:10.6f}, {z:10.6f})")
        
        return {
            'points': points,
            'count': len(points),
            'bbox_size': bbox_size,
            'bbox_center': bbox_center,
            'unit': unit,
            'scale_factor': scale_factor,
            'shape_type': shape_type,
            'has_colors': has_colors
        }
        
    except Exception as e:
        print(f"X 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("파이프 포인트 클라우드 상세 분석")
    
    # 모든 파이프 파일 분석
    files = ['pipe.ply', 'pipe_1.ply', 'pipe_2.ply']
    results = {}
    
    for file in files:
        try:
            result = analyze_pipe_pointcloud(file)
            if result:
                results[file] = result
        except FileNotFoundError:
            print(f"\n{file} 파일을 찾을 수 없습니다.")
        except Exception as e:
            print(f"\n{file} 분석 중 오류: {e}")
    
    # 비교 분석
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("전체 비교 분석")
        print('='*60)
        
        for file, data in results.items():
            print(f"{file:12s}: {data['count']:8,}개 포인트, {data['unit']:2s} 단위, {data['shape_type']}")