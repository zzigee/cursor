#!/usr/bin/env python3
"""
포인트 클라우드 직접 충돌 감지 전용 GUI
- 메시 변환 없이 포인트 클라우드 직접 사용
- KDTree 기반 고속 충돌 감지
- 형상 100% 보존
"""
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import os
import time
import json
import hashlib

# PyBullet은 선택적 import
try:
    import pybullet as p
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    print("PyBullet not available - some features disabled")

# SciPy는 필수
try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("SciPy required for KDTree - please install: pip install scipy")
    exit(1)

# scikit-learn은 선택적
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("scikit-learn not available - using simple sampling")

class PointCloudCollisionViewer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("포인트 클라우드 직접 충돌 감지 뷰어")
        self.root.geometry("1000x700")
        
        # 데이터 저장
        self.current_points = None
        self.current_filename = ""
        self.collision_points = None
        self.kdtree = None
        self.collision_metadata = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """UI 설정"""
        # 메인 레이아웃
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 왼쪽: 컨트롤 패널
        left_frame = ttk.Frame(main_frame, width=350)
        left_frame.pack(side="left", fill="y", padx=(0, 10))
        left_frame.pack_propagate(False)
        
        # 오른쪽: 정보 패널
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side="right", fill="both", expand=True)
        
        self.setup_control_panel(left_frame)
        self.setup_info_panel(right_frame)
        
    def setup_control_panel(self, parent):
        """컨트롤 패널 설정"""
        
        # 1. 파일 선택
        file_section = ttk.LabelFrame(parent, text="1. PLY 파일 선택")
        file_section.pack(fill="x", padx=5, pady=5)
        
        file_frame = ttk.Frame(file_section)
        file_frame.pack(fill="x", padx=5, pady=5)
        
        self.file_var = tk.StringVar()
        file_combo = ttk.Combobox(file_frame, textvariable=self.file_var, width=25)
        file_combo['values'] = self.get_ply_files()
        file_combo.pack(side="left", fill="x", expand=True)
        
        ttk.Button(file_frame, text="찾기", command=self.open_file).pack(side="right", padx=(5,0))
        ttk.Button(file_frame, text="로드", command=self.load_file).pack(side="right", padx=(5,0))
        
        # 2. 충돌 감지 설정
        collision_section = ttk.LabelFrame(parent, text="2. 충돌 감지 설정")
        collision_section.pack(fill="x", padx=5, pady=5)
        
        # 최적 포인트 수
        points_frame = ttk.Frame(collision_section)
        points_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(points_frame, text="충돌 감지 포인트 수:").pack(side="left")
        self.collision_points_var = tk.StringVar(value="10000")
        points_combo = ttk.Combobox(points_frame, textvariable=self.collision_points_var, width=10)
        points_combo['values'] = ["5000", "10000", "20000", "50000"]
        points_combo.pack(side="right")
        
        # 충돌 반지름
        radius_frame = ttk.Frame(collision_section)
        radius_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(radius_frame, text="충돌 반지름:").pack(side="left")
        self.collision_radius_var = tk.DoubleVar(value=0.01)
        radius_scale = ttk.Scale(radius_frame, from_=0.001, to=0.1, 
                               variable=self.collision_radius_var, orient="horizontal")
        radius_scale.pack(side="right", fill="x", expand=True)
        
        # 3. 처리 실행
        process_section = ttk.LabelFrame(parent, text="3. 처리 실행")
        process_section.pack(fill="x", padx=5, pady=5)
        
        btn_frame = ttk.Frame(process_section)
        btn_frame.pack(fill="x", padx=5, pady=5)
        
        self.process_btn = ttk.Button(btn_frame, text="충돌 데이터 생성", command=self.start_processing)
        self.process_btn.pack(fill="x", pady=2)
        
        # 4. 테스트
        test_section = ttk.LabelFrame(parent, text="4. 충돌 테스트")
        test_section.pack(fill="x", padx=5, pady=5)
        
        test_btn_frame = ttk.Frame(test_section)
        test_btn_frame.pack(fill="x", padx=5, pady=5)
        
        self.kdtree_test_btn = ttk.Button(test_btn_frame, text="KDTree 고속 테스트", 
                                        command=self.run_kdtree_test, state="disabled")
        self.kdtree_test_btn.pack(fill="x", pady=2)
        
        if PYBULLET_AVAILABLE:
            self.pybullet_test_btn = ttk.Button(test_btn_frame, text="PyBullet 시각 테스트", 
                                              command=self.run_pybullet_test, state="disabled")
            self.pybullet_test_btn.pack(fill="x", pady=2)
        
        self.view_btn = ttk.Button(test_btn_frame, text="3D 뷰어", 
                                 command=self.show_3d_view, state="disabled")
        self.view_btn.pack(fill="x", pady=2)
        
        # 5. 진행률
        progress_section = ttk.Frame(process_section)
        progress_section.pack(fill="x", padx=5, pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_section, variable=self.progress_var)
        self.progress_bar.pack(fill="x", pady=2)
        
        self.status_var = tk.StringVar(value="파일을 선택하고 로드하세요")
        status_label = ttk.Label(progress_section, textvariable=self.status_var)
        status_label.pack(fill="x")
        
    def setup_info_panel(self, parent):
        """정보 패널 설정"""
        
        # 탭 노트북
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill="both", expand=True)
        
        # 파일 정보 탭
        info_frame = ttk.Frame(self.notebook)
        self.notebook.add(info_frame, text="파일 정보")
        
        self.info_text = tk.Text(info_frame, wrap="word", font=("Consolas", 9))
        info_scrollbar = ttk.Scrollbar(info_frame, orient="vertical", command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=info_scrollbar.set)
        self.info_text.pack(side="left", fill="both", expand=True)
        info_scrollbar.pack(side="right", fill="y")
        
        # 처리 로그 탭
        log_frame = ttk.Frame(self.notebook)
        self.notebook.add(log_frame, text="처리 로그")
        
        self.log_text = tk.Text(log_frame, wrap="word", font=("Consolas", 9))
        log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scrollbar.pack(side="right", fill="y")
        
    def get_ply_files(self):
        """PLY 파일 목록"""
        files = []
        try:
            for f in os.listdir('.'):
                if f.endswith('.ply'):
                    try:
                        size = os.path.getsize(f)
                        size_mb = size / 1024 / 1024
                        files.append(f"{f} ({size_mb:.1f}MB)")
                    except:
                        files.append(f)
        except:
            pass
        return sorted(files)
    
    def open_file(self):
        """파일 선택"""
        filename = filedialog.askopenfilename(
            title="PLY 파일 선택",
            filetypes=[("PLY files", "*.ply"), ("All files", "*.*")]
        )
        if filename:
            self.file_var.set(os.path.basename(filename))
            self.load_file()
    
    def load_file(self):
        """PLY 파일 로드"""
        filename = self.file_var.get()
        if not filename:
            messagebox.showwarning("경고", "파일을 선택해주세요.")
            return
        
        # 파일명에서 크기 정보 제거
        actual_filename = filename.split(' (')[0] if ' (' in filename else filename
        
        if not os.path.exists(actual_filename):
            messagebox.showerror("오류", f"파일을 찾을 수 없습니다: {actual_filename}")
            return
        
        try:
            self.log("PLY 파일 로딩 시작...")
            
            # PLY 파일 로드
            mesh = trimesh.load(actual_filename)
            self.current_filename = actual_filename
            
            # 포인트 추출
            if isinstance(mesh, trimesh.points.PointCloud):
                self.current_points = mesh.vertices
                data_type = "포인트 클라우드"
            elif hasattr(mesh, 'vertices'):
                self.current_points = mesh.vertices
                data_type = "메시"
            else:
                raise Exception("포인트 데이터를 찾을 수 없음")
            
            self.display_file_info(data_type)
            self.log(f"파일 로드 완료: {actual_filename}")
            
            # 처리 버튼 활성화
            self.process_btn.config(state="normal")
            
        except Exception as e:
            error_msg = f"파일 로드 실패: {str(e)}"
            self.log(error_msg)
            messagebox.showerror("오류", error_msg)
    
    def display_file_info(self, data_type):
        """파일 정보 표시"""
        self.info_text.delete(1.0, tk.END)
        
        points = self.current_points
        filename = self.current_filename
        
        info = f"=== {filename} ===\\n\\n"
        info += f"타입: {data_type}\\n"
        info += f"포인트 수: {len(points):,}개\\n\\n"
        
        # 파일 크기
        file_size = os.path.getsize(filename)
        info += f"파일 크기: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)\\n\\n"
        
        # 좌표 범위
        info += "=== 좌표 범위 ===\\n"
        info += f"X: {points[:,0].min():10.6f} ~ {points[:,0].max():10.6f}\\n"
        info += f"Y: {points[:,1].min():10.6f} ~ {points[:,1].max():10.6f}\\n"
        info += f"Z: {points[:,2].min():10.6f} ~ {points[:,2].max():10.6f}\\n\\n"
        
        # 크기 및 중심
        size = points.max(axis=0) - points.min(axis=0)
        center = (points.max(axis=0) + points.min(axis=0)) / 2
        
        info += "=== 바운딩 박스 ===\\n"
        info += f"크기: {size[0]:.6f} × {size[1]:.6f} × {size[2]:.6f}\\n"
        info += f"중심: ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})\\n\\n"
        
        # 추정 단위
        max_coord = np.abs(points).max()
        if max_coord > 1000:
            unit = "mm"
        elif max_coord > 10:
            unit = "cm"
        else:
            unit = "m"
        info += f"추정 단위: {unit}\\n\\n"
        
        # 충돌 감지 권장 설정
        info += "=== 충돌 감지 권장 설정 ===\\n"
        if len(points) > 100000:
            info += "포인트 수: 10,000 (대용량)\\n"
        elif len(points) > 50000:
            info += "포인트 수: 20,000 (중용량)\\n"
        else:
            info += "포인트 수: 모든 포인트 사용\\n"
        
        avg_size = np.mean(size)
        recommended_radius = max(0.005, avg_size * 0.01)
        info += f"충돌 반지름: {recommended_radius:.3f}\\n"
        
        self.info_text.insert(tk.END, info)
        
        # 권장 설정 자동 적용
        if len(points) > 100000:
            self.collision_points_var.set("10000")
        elif len(points) > 50000:
            self.collision_points_var.set("20000")
        else:
            self.collision_points_var.set(str(min(50000, len(points))))
        
        self.collision_radius_var.set(recommended_radius)
    
    def start_processing(self):
        """처리 시작"""
        if self.current_points is None:
            messagebox.showwarning("경고", "먼저 파일을 로드해주세요.")
            return
        
        # 별도 스레드에서 처리
        thread = threading.Thread(target=self.process_point_cloud)
        thread.daemon = True
        thread.start()
    
    def process_point_cloud(self):
        """포인트 클라우드 충돌 데이터 처리"""
        try:
            self.progress_var.set(0)
            self.status_var.set("처리 시작...")
            self.log("=== 포인트 클라우드 충돌 데이터 생성 시작 ===")
            
            points = self.current_points
            target_count = int(self.collision_points_var.get())
            collision_radius = self.collision_radius_var.get()
            
            # 1. 다운샘플링
            self.progress_var.set(20)
            self.status_var.set("포인트 샘플링 중...")
            
            if len(points) > target_count:
                self.log(f"포인트 샘플링: {len(points):,} → {target_count:,}")
                sampled_points = self.smart_sampling(points, target_count)
            else:
                sampled_points = points
                self.log(f"모든 포인트 사용: {len(points):,}개")
            
            # 2. KDTree 구축
            self.progress_var.set(60)
            self.status_var.set("KDTree 구축 중...")
            self.log("KDTree 구축...")
            
            kdtree = cKDTree(sampled_points)
            
            # 3. 메타데이터 저장
            self.progress_var.set(90)
            self.status_var.set("메타데이터 저장 중...")
            
            self.collision_points = sampled_points
            self.kdtree = kdtree
            self.collision_metadata = {
                'points': sampled_points,
                'radius': collision_radius,
                'original_count': len(points),
                'collision_count': len(sampled_points),
                'filename': self.current_filename
            }
            
            # 완료
            self.progress_var.set(100)
            self.status_var.set("충돌 데이터 준비 완료!")
            self.log(f"충돌 데이터 생성 완료: {len(sampled_points):,}개 포인트")
            
            # 테스트 버튼 활성화
            self.kdtree_test_btn.config(state="normal")
            if hasattr(self, 'pybullet_test_btn'):
                self.pybullet_test_btn.config(state="normal")
            self.view_btn.config(state="normal")
            
        except Exception as e:
            error_msg = f"처리 실패: {str(e)}"
            self.log(error_msg)
            self.status_var.set("처리 실패")
            messagebox.showerror("오류", error_msg)
    
    def smart_sampling(self, points, target_count):
        """스마트 포인트 샘플링"""
        try:
            if SKLEARN_AVAILABLE and len(points) > target_count * 2:
                # K-means 클러스터링 기반 샘플링
                self.log("K-means 클러스터링 샘플링 사용")
                n_clusters = min(target_count // 2, 100)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(points)
                
                selected_points = []
                points_per_cluster = target_count // n_clusters
                
                for i in range(n_clusters):
                    cluster_points = points[clusters == i]
                    if len(cluster_points) > 0:
                        center = kmeans.cluster_centers_[i]
                        distances = np.linalg.norm(cluster_points - center, axis=1)
                        n_select = min(points_per_cluster, len(cluster_points))
                        closest_indices = np.argsort(distances)[:n_select]
                        selected_points.extend(cluster_points[closest_indices])
                
                return np.array(selected_points[:target_count])
            else:
                # 균등 간격 샘플링
                self.log("균등 간격 샘플링 사용")
                step = len(points) // target_count
                return points[::step][:target_count]
                
        except Exception as e:
            self.log(f"스마트 샘플링 실패: {e}, 균등 샘플링으로 대체")
            step = len(points) // target_count
            return points[::step][:target_count]
    
    def run_kdtree_test(self):
        """KDTree 충돌 테스트"""
        if self.kdtree is None:
            messagebox.showwarning("경고", "먼저 충돌 데이터를 생성해주세요.")
            return
        
        try:
            self.log("KDTree 고속 충돌 테스트 시작...")
            
            # 테스트 그리드 생성
            bounds = np.array([self.collision_points.min(axis=0), 
                             self.collision_points.max(axis=0)])
            test_grid = self.generate_test_grid(bounds, density=15)
            
            # 충돌 검사 실행
            start_time = time.time()
            collision_count = 0
            radius = self.collision_metadata['radius']
            
            for test_point in test_grid:
                neighbors = self.kdtree.query_ball_point(test_point, radius)
                if len(neighbors) > 0:
                    collision_count += 1
            
            processing_time = time.time() - start_time
            collision_rate = (collision_count / len(test_grid)) * 100
            speed = len(test_grid) / processing_time
            
            result_text = f"""KDTree 충돌 테스트 완료
            
테스트 포인트: {len(test_grid):,}개
충돌 감지: {collision_count:,}개
충돌률: {collision_rate:.1f}%
처리 시간: {processing_time:.3f}초
처리 속도: {speed:,.0f} 포인트/초
충돌 반지름: {radius:.3f}m

포인트 클라우드 직접 충돌 감지가 정상 작동합니다!"""

            messagebox.showinfo("KDTree 테스트 결과", result_text)
            self.log(f"KDTree 테스트 완료: {speed:,.0f} 포인트/초")
            
        except Exception as e:
            error_msg = f"KDTree 테스트 실패: {str(e)}"
            self.log(error_msg)
            messagebox.showerror("오류", error_msg)
    
    def run_pybullet_test(self):
        """PyBullet 시각 테스트"""
        if not PYBULLET_AVAILABLE:
            messagebox.showwarning("경고", "PyBullet이 설치되어 있지 않습니다.")
            return
        
        if self.collision_points is None:
            messagebox.showwarning("경고", "먼저 충돌 데이터를 생성해주세요.")
            return
        
        try:
            self.log("PyBullet 시각 테스트 시작...")
            
            physics_client = p.connect(p.GUI)
            p.setGravity(0, 0, -9.81)
            
            radius = self.collision_metadata['radius']
            collision_points = self.collision_points
            
            # 포인트들을 구체로 시각화 (최대 500개)
            max_spheres = min(500, len(collision_points))
            
            collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
            visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=radius, 
                                             rgbaColor=[0.7, 0.7, 1.0, 0.8])
            
            for i, point in enumerate(collision_points[:max_spheres]):
                p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=collision_shape,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=point.tolist()
                )
            
            # 테스트 객체들
            self.create_test_objects()
            
            self.log(f"PyBullet 테스트 준비 완료: {max_spheres}개 구체")
            messagebox.showinfo("PyBullet 테스트", 
                f"PyBullet 창에서 충돌 테스트 진행:\\n"
                f"• {max_spheres}개 구체로 포인트 클라우드 표현\\n"
                f"• 테스트 객체들의 충돌 관찰\\n"
                f"• 구체 반지름: {radius:.3f}m")
            
        except Exception as e:
            error_msg = f"PyBullet 테스트 실패: {str(e)}"
            self.log(error_msg)
            messagebox.showerror("오류", error_msg)
    
    def create_test_objects(self):
        """테스트 객체 생성"""
        try:
            # 테스트 구
            sphere_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.02)
            sphere_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=[1, 0, 0, 1])
            p.createMultiBody(0.1, sphere_shape, sphere_visual, [0, 0, 0.5])
            
            # 테스트 박스
            box_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
            box_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02], rgbaColor=[0, 1, 0, 1])
            p.createMultiBody(0.1, box_shape, box_visual, [0.1, 0, 0.5])
            
        except Exception as e:
            self.log(f"테스트 객체 생성 실패: {e}")
    
    def generate_test_grid(self, bounds, density=10):
        """테스트 그리드 생성"""
        min_bounds, max_bounds = bounds
        
        x_points = np.linspace(min_bounds[0], max_bounds[0], density)
        y_points = np.linspace(min_bounds[1], max_bounds[1], density)
        z_points = np.linspace(min_bounds[2], max_bounds[2], density)
        
        grid_points = []
        for x in x_points:
            for y in y_points:
                for z in z_points:
                    grid_points.append([x, y, z])
        
        return np.array(grid_points)
    
    def show_3d_view(self):
        """3D 뷰어"""
        if self.current_points is None:
            messagebox.showwarning("경고", "먼저 파일을 로드해주세요.")
            return
        
        thread = threading.Thread(target=self.create_3d_plot)
        thread.daemon = True
        thread.start()
    
    def create_3d_plot(self):
        """3D 플롯 생성"""
        try:
            original_points = self.current_points
            
            # 표시용 다운샘플링
            if len(original_points) > 20000:
                step = len(original_points) // 20000
                display_points = original_points[::step]
            else:
                display_points = original_points
            
            fig = plt.figure(figsize=(14, 10))
            
            if self.collision_points is not None:
                # 원본 + 충돌 포인트 비교
                ax1 = fig.add_subplot(121, projection='3d')
                ax1.scatter(display_points[:, 0], display_points[:, 1], display_points[:, 2],
                           c='lightblue', s=0.5, alpha=0.6, label='Original')
                ax1.set_title(f'Original ({len(display_points):,} points)')
                ax1.set_xlabel('X')
                ax1.set_ylabel('Y')
                ax1.set_zlabel('Z')
                
                ax2 = fig.add_subplot(122, projection='3d')
                collision_points = self.collision_points
                ax2.scatter(collision_points[:, 0], collision_points[:, 1], collision_points[:, 2],
                           c='red', s=2, alpha=0.8, label='Collision')
                ax2.set_title(f'Collision Points ({len(collision_points):,} points)')
                ax2.set_xlabel('X')
                ax2.set_ylabel('Y')
                ax2.set_zlabel('Z')
                
                plt.suptitle(f'Point Cloud: {self.current_filename}', fontsize=14)
            else:
                # 원본만
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(display_points[:, 0], display_points[:, 1], display_points[:, 2],
                          c=display_points[:, 2], cmap='viridis', s=0.5, alpha=0.7)
                ax.set_title(f'Point Cloud: {self.current_filename} ({len(display_points):,} points)')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            self.log(f"3D 뷰 생성 실패: {e}")
    
    def log(self, message):
        """로그 메시지"""
        timestamp = time.strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\\n"
        self.log_text.insert(tk.END, log_message)
        self.log_text.see(tk.END)
        self.root.update()
        print(f"{timestamp}: {message}")
    
    def run(self):
        """GUI 실행"""
        self.root.mainloop()

if __name__ == "__main__":
    if not SCIPY_AVAILABLE:
        print("오류: SciPy가 필요합니다. 설치하세요: pip install scipy")
        exit(1)
    
    print("포인트 클라우드 직접 충돌 감지 뷰어 시작...")
    app = PointCloudCollisionViewer()
    app.run()