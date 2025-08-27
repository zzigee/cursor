#!/usr/bin/env python3
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
import subprocess
import tempfile

# PLY 처리 전용 모듈 import
try:
    import ply_processor
    from ply_processor import load_pipe_from_ply, load_ply_as_pybullet_body
    PLY_PROCESSOR_AVAILABLE = True
    print("ply_processor 모듈 로드 성공")
except ImportError:
    PLY_PROCESSOR_AVAILABLE = False
    print("ply_processor 모듈 없음 - 기본 처리 방식 사용")

# PyBullet은 선택적 import
try:
    import pybullet as p
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    print("PyBullet not available - collision testing disabled")

# 고급 메시 생성을 위한 추가 라이브러리들
try:
    from scipy.spatial import Delaunay, ConvexHull, cKDTree
    from sklearn.neighbors import NearestNeighbors
    SCIPY_AVAILABLE = True
    SKLEARN_AVAILABLE = True
except ImportError:
    try:
        from scipy.spatial import Delaunay, ConvexHull, cKDTree
        SCIPY_AVAILABLE = True
        SKLEARN_AVAILABLE = False
        print("scikit-learn not available - some features disabled")
    except ImportError:
        SCIPY_AVAILABLE = False
        SKLEARN_AVAILABLE = False
        print("SciPy not available - some mesh generation methods disabled")

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Open3D not available - advanced reconstruction methods disabled")

class CollisionProcessorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("PLY 포인트 클라우드 → 충돌 형상 변환기")
        self.root.geometry("1200x800")
        
        # 데이터 저장
        self.current_mesh = None
        self.current_filename = ""
        self.processed_mesh = None
        self.processing_stats = {}
        
        # 진행 상태
        self.is_processing = False
        self.processing_thread = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """메인 UI 설정"""
        # 메뉴 바
        self.create_menu()
        
        # 메인 프레임을 좌우로 분할
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 왼쪽 패널 (컨트롤)
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)
        
        # 오른쪽 패널 (정보 및 결과)
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=2)
        
        self.setup_control_panel(left_frame)
        self.setup_info_panel(right_frame)
        
    def create_menu(self):
        """메뉴 바 생성"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # 파일 메뉴
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="파일", menu=file_menu)
        file_menu.add_command(label="PLY 파일 열기...", command=self.open_file)
        file_menu.add_separator()
        file_menu.add_command(label="OBJ 파일 저장...", command=self.save_obj)
        file_menu.add_command(label="V-HACD 결과 저장...", command=self.save_vhacd)
        file_menu.add_separator()
        file_menu.add_command(label="종료", command=self.root.quit)
        
        # 도구 메뉴
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="도구", menu=tools_menu)
        tools_menu.add_command(label="설정 초기화", command=self.reset_settings)
        tools_menu.add_command(label="캐시 정리", command=self.clear_cache)
        
    def setup_control_panel(self, parent):
        """왼쪽 컨트롤 패널 설정"""
        
        # 1. 파일 선택 섹션
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
        
        # 파일 정보 표시
        info_frame = ttk.Frame(file_section)
        info_frame.pack(fill="x", padx=5, pady=2)
        self.file_info_label = ttk.Label(info_frame, text="파일을 선택해주세요.", foreground="gray")
        self.file_info_label.pack(side="left")
        
        # 새로고침 버튼
        ttk.Button(info_frame, text="새로고침", command=self.refresh_files).pack(side="right")
        
        # 2. 처리 옵션 섹션
        options_section = ttk.LabelFrame(parent, text="2. 처리 옵션")
        options_section.pack(fill="x", padx=5, pady=5)
        
        # 다운샘플링 옵션
        ds_frame = ttk.Frame(options_section)
        ds_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(ds_frame, text="포인트 수 제한:").pack(side="left")
        self.max_points_var = tk.StringVar(value="100000")
        max_points_combo = ttk.Combobox(ds_frame, textvariable=self.max_points_var, width=10)
        max_points_combo['values'] = ["50000", "100000", "200000", "500000", "제한없음"]
        max_points_combo.pack(side="right")
        
        # V-HACD 파라미터
        vhacd_frame = ttk.LabelFrame(options_section, text="V-HACD 파라미터")
        vhacd_frame.pack(fill="x", padx=5, pady=5)
        
        # 처리 모드 선택
        mode_frame = ttk.Frame(vhacd_frame)
        mode_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(mode_frame, text="처리 모드:").pack(side="left")
        self.vhacd_mode_var = tk.StringVar(value="일반 모드")
        mode_combo = ttk.Combobox(mode_frame, textvariable=self.vhacd_mode_var, width=12)
        mode_combo['values'] = ["고속 모드", "일반 모드", "고품질 모드"]
        mode_combo.pack(side="right")
        mode_combo.bind('<<ComboboxSelected>>', self.on_vhacd_mode_change)
        
        # 해상도 (동적)
        res_frame = ttk.Frame(vhacd_frame)
        res_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(res_frame, text="해상도:").pack(side="left")
        self.resolution_var = tk.StringVar(value="75000")
        self.res_combo = ttk.Combobox(res_frame, textvariable=self.resolution_var, width=10)
        self.res_combo['values'] = ["50000", "75000", "100000", "150000"]
        self.res_combo.pack(side="right")
        
        # 분해 깊이 (동적)
        depth_frame = ttk.Frame(vhacd_frame)
        depth_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(depth_frame, text="분해 깊이:").pack(side="left")
        self.depth_var = tk.IntVar(value=8)
        self.depth_scale = ttk.Scale(depth_frame, from_=4, to=12, variable=self.depth_var, orient="horizontal")
        self.depth_scale.pack(side="right", fill="x", expand=True)
        
        # Concavity (동적)
        conc_frame = ttk.Frame(vhacd_frame)
        conc_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(conc_frame, text="Concavity:").pack(side="left")
        self.concavity_var = tk.DoubleVar(value=0.025)
        self.conc_scale = ttk.Scale(conc_frame, from_=0.01, to=0.1, variable=self.concavity_var, orient="horizontal")
        self.conc_scale.pack(side="right", fill="x", expand=True)
        
        # 메시 생성 방법
        mesh_frame = ttk.LabelFrame(options_section, text="메시 생성 방법")
        mesh_frame.pack(fill="x", padx=5, pady=5)
        
        method_frame = ttk.Frame(mesh_frame)
        method_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(method_frame, text="방법:").pack(side="left")
        self.mesh_method_var = tk.StringVar(value="Convex Hull")
        method_combo = ttk.Combobox(method_frame, textvariable=self.mesh_method_var, width=15)
        method_combo['values'] = [
            "Point Cloud Direct",     # 포인트 클라우드 직접 사용
            "Convex Hull (Original)", 
            "Alpha Shape",
            "Mesh Reconstruction"
        ]
        method_combo.pack(side="right")
        method_combo.bind('<<ComboboxSelected>>', self.on_mesh_method_change)
        
        # Alpha 파라미터 (Alpha Shape용)
        alpha_frame = ttk.Frame(mesh_frame)
        alpha_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(alpha_frame, text="Alpha (형상 정밀도):").pack(side="left")
        self.alpha_var = tk.DoubleVar(value=0.1)
        alpha_scale = ttk.Scale(alpha_frame, from_=0.01, to=1.0, variable=self.alpha_var, orient="horizontal")
        alpha_scale.pack(side="right", fill="x", expand=True)
        
        # Point Cloud 직접 충돌 파라미터
        pc_frame = ttk.LabelFrame(options_section, text="포인트 클라우드 충돌 설정")
        pc_frame.pack(fill="x", padx=5, pady=5)
        
        # 구체 반지름
        radius_frame = ttk.Frame(pc_frame)
        radius_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(radius_frame, text="충돌 구체 반지름:").pack(side="left")
        self.collision_radius_var = tk.DoubleVar(value=0.01)
        radius_scale = ttk.Scale(radius_frame, from_=0.001, to=0.1, variable=self.collision_radius_var, orient="horizontal")
        radius_scale.pack(side="right", fill="x", expand=True)
        
        # 최적 포인트 수
        optimal_frame = ttk.Frame(pc_frame)
        optimal_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(optimal_frame, text="최적 포인트 수:").pack(side="left")
        self.optimal_points_var = tk.StringVar(value="10000")
        optimal_combo = ttk.Combobox(optimal_frame, textvariable=self.optimal_points_var, width=10)
        optimal_combo['values'] = ["5000", "10000", "20000", "50000"]
        optimal_combo.pack(side="right")
        
        # 3. 처리 실행 섹션
        process_section = ttk.LabelFrame(parent, text="3. 처리 실행")
        process_section.pack(fill="x", padx=5, pady=5)
        
        # 처리 버튼들
        btn_frame = ttk.Frame(process_section)
        btn_frame.pack(fill="x", padx=5, pady=5)
        
        self.process_btn = ttk.Button(btn_frame, text="충돌 형상 생성", command=self.start_processing)
        self.process_btn.pack(fill="x", pady=2)
        
        self.test_btn = ttk.Button(btn_frame, text="PyBullet 테스트", command=self.test_collision, state="disabled")
        self.test_btn.pack(fill="x", pady=2)
        
        self.view_btn = ttk.Button(btn_frame, text="3D 비교 뷰", command=self.show_comparison, state="disabled")
        self.view_btn.pack(fill="x", pady=2)
        
        # 진행률 표시
        progress_frame = ttk.Frame(process_section)
        progress_frame.pack(fill="x", padx=5, pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill="x", pady=2)
        
        self.status_var = tk.StringVar(value="대기 중...")
        self.status_label = ttk.Label(progress_frame, textvariable=self.status_var)
        self.status_label.pack(pady=2)
        
    def setup_info_panel(self, parent):
        """오른쪽 정보 패널 설정"""
        
        # 탭 노트북 생성
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 1. 파일 정보 탭
        info_frame = ttk.Frame(self.notebook)
        self.notebook.add(info_frame, text="파일 정보")
        
        self.info_text = tk.Text(info_frame, wrap="word", font=("Consolas", 9))
        info_scrollbar = ttk.Scrollbar(info_frame, orient="vertical", command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=info_scrollbar.set)
        self.info_text.pack(side="left", fill="both", expand=True)
        info_scrollbar.pack(side="right", fill="y")
        
        # 2. 처리 로그 탭
        log_frame = ttk.Frame(self.notebook)
        self.notebook.add(log_frame, text="처리 로그")
        
        self.log_text = tk.Text(log_frame, wrap="word", font=("Consolas", 9))
        log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scrollbar.pack(side="right", fill="y")
        
        # 3. 결과 비교 탭
        result_frame = ttk.Frame(self.notebook)
        self.notebook.add(result_frame, text="처리 결과")
        
        self.result_text = tk.Text(result_frame, wrap="word", font=("Consolas", 9))
        result_scrollbar = ttk.Scrollbar(result_frame, orient="vertical", command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=result_scrollbar.set)
        self.result_text.pack(side="left", fill="both", expand=True)
        result_scrollbar.pack(side="right", fill="y")
        
    def get_ply_files(self):
        """현재 디렉토리의 PLY 파일 목록 (크기 정보 포함)"""
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
    
    def refresh_files(self):
        """파일 목록 새로고침"""
        # 콤보박스 값 업데이트
        combo = None
        for widget in self.root.winfo_children():
            if isinstance(widget, ttk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.LabelFrame) and "PLY 파일" in child.cget('text'):
                        for subchild in child.winfo_children():
                            if isinstance(subchild, ttk.Frame):
                                for item in subchild.winfo_children():
                                    if isinstance(item, ttk.Combobox):
                                        combo = item
                                        break
        
        if combo:
            combo['values'] = self.get_ply_files()
            self.update_file_info()
    
    def update_file_info(self):
        """파일 정보 업데이트"""
        filename = self.file_var.get()
        if filename and hasattr(self, 'file_info_label'):
            # 파일명에서 크기 정보 제거
            actual_filename = filename.split(' (')[0] if ' (' in filename else filename
            if os.path.exists(actual_filename):
                try:
                    size = os.path.getsize(actual_filename)
                    size_mb = size / 1024 / 1024
                    self.file_info_label.config(text=f"선택됨: {actual_filename} ({size_mb:.1f}MB)", foreground="blue")
                except:
                    self.file_info_label.config(text=f"선택됨: {actual_filename}", foreground="blue")
            else:
                self.file_info_label.config(text="파일을 찾을 수 없습니다.", foreground="red")
        elif hasattr(self, 'file_info_label'):
            self.file_info_label.config(text="파일을 선택해주세요.", foreground="gray")
        
    def open_file(self):
        """파일 선택 대화상자"""
        filename = filedialog.askopenfilename(
            title="PLY 파일 선택",
            filetypes=[("PLY files", "*.ply"), ("All files", "*.*")]
        )
        if filename:
            self.file_var.set(os.path.basename(filename))
            self.load_file()
            
    def load_file(self):
        """선택된 PLY 파일 로드"""
        filename = self.file_var.get()
        if not filename:
            messagebox.showwarning("경고", "파일을 선택해주세요.")
            return
        
        # 파일명에서 크기 정보 제거 (예: "pipe_1.ply (10.5MB)" -> "pipe_1.ply")
        actual_filename = filename.split(' (')[0] if ' (' in filename else filename
            
        if not os.path.exists(actual_filename):
            messagebox.showerror("오류", f"파일을 찾을 수 없습니다: {actual_filename}")
            return
            
        try:
            self.log("PLY 파일 로딩 시작...")
            
            # PLY 파일 로드
            mesh = trimesh.load(actual_filename)
            self.current_mesh = mesh
            self.current_filename = actual_filename
            self.update_file_info()
            
            # 정보 표시
            self.display_file_info()
            self.log(f"파일 로드 완료: {filename}")
            
        except Exception as e:
            error_msg = f"파일 로드 실패: {str(e)}"
            messagebox.showerror("오류", error_msg)
            self.log(error_msg)
            
    def display_file_info(self):
        """파일 정보 표시"""
        if not self.current_mesh:
            return
            
        self.info_text.delete(1.0, tk.END)
        
        mesh = self.current_mesh
        filename = self.current_filename
        
        info = f"=== {filename} ===\n\n"
        
        # 기본 정보
        if isinstance(mesh, trimesh.points.PointCloud):
            points = mesh.vertices
            info += f"타입: 포인트 클라우드\n"
            info += f"포인트 수: {len(points):,}개\n"
            has_faces = False
        elif hasattr(mesh, 'vertices'):
            points = mesh.vertices
            info += f"타입: 메시\n"
            info += f"정점 수: {len(points):,}개\n"
            if hasattr(mesh, 'faces') and len(mesh.faces) > 0:
                info += f"면 수: {len(mesh.faces):,}개\n"
                has_faces = True
            else:
                has_faces = False
        else:
            info += "오류: 포인트 데이터를 찾을 수 없음\n"
            self.info_text.insert(tk.END, info)
            return
            
        # 파일 정보
        file_size = os.path.getsize(filename)
        info += f"파일 크기: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)\n\n"
        
        # 좌표 범위
        info += "=== 좌표 범위 ===\n"
        info += f"X: {points[:,0].min():10.6f} ~ {points[:,0].max():10.6f}\n"
        info += f"Y: {points[:,1].min():10.6f} ~ {points[:,1].max():10.6f}\n"
        info += f"Z: {points[:,2].min():10.6f} ~ {points[:,2].max():10.6f}\n\n"
        
        # 크기 및 중심
        size = points.max(axis=0) - points.min(axis=0)
        center = (points.max(axis=0) + points.min(axis=0)) / 2
        
        info += "=== 바운딩 박스 ===\n"
        info += f"크기: {size[0]:.6f} × {size[1]:.6f} × {size[2]:.6f}\n"
        info += f"중심: ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})\n\n"
        
        # 단위 추정
        max_coord = np.abs(points).max()
        if max_coord > 1000:
            unit = "mm"
            scale_factor = 0.001
        elif max_coord > 10:
            unit = "cm"
            scale_factor = 0.01
        else:
            unit = "m"
            scale_factor = 1.0
            
        info += f"추정 단위: {unit}\n"
        info += f"미터 변환 스케일: {scale_factor}\n\n"
        
        # 충돌 감지 처리 권장사항
        info += "=== 처리 권장사항 ===\n"
        if not has_faces:
            info += "* 포인트 클라우드 → Convex Hull 변환 필요\n"
        if len(points) > 200000:
            info += "* 대용량 데이터 → 다운샘플링 권장\n"
        if max_coord > 100:
            info += f"* 큰 좌표값 → 스케일 조정 필요 ({scale_factor}x)\n"
            
        self.info_text.insert(tk.END, info)
        
    def log(self, message):
        """처리 로그 추가"""
        timestamp = time.strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, log_message)
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def start_processing(self):
        """충돌 형상 생성 시작"""
        if not self.current_mesh:
            messagebox.showwarning("경고", "먼저 PLY 파일을 로드해주세요.")
            return
            
        if self.is_processing:
            messagebox.showinfo("정보", "이미 처리 중입니다.")
            return
            
        # 처리 스레드 시작
        self.is_processing = True
        self.process_btn.config(state="disabled")
        self.processing_thread = threading.Thread(target=self.process_collision_shape)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
    def process_collision_shape(self):
        """충돌 형상 생성 메인 프로세스"""
        try:
            self.progress_var.set(0)
            self.status_var.set("처리 시작...")
            self.log("=== 충돌 형상 생성 시작 ===")
            
            mesh = self.current_mesh
            filename = self.current_filename
            
            # 1. 포인트 추출
            self.progress_var.set(10)
            self.status_var.set("포인트 데이터 추출 중...")
            
            if isinstance(mesh, trimesh.points.PointCloud):
                points = mesh.vertices
                self.log(f"포인트 클라우드: {len(points):,}개 포인트")
            elif hasattr(mesh, 'vertices'):
                points = mesh.vertices
                self.log(f"메시 정점: {len(points):,}개")
            else:
                raise Exception("포인트 데이터를 찾을 수 없음")
                
            # 2. 스마트 다운샘플링 (urdf_visualizer_pybullet.py 방식)
            self.progress_var.set(20)
            self.status_var.set("다운샘플링 중...")
            
            original_count = len(points)
            
            # 극대 데이터셋 사전 필터링
            if original_count > 1000000:  # 100만개 초과
                self.log(f"경고: 극대 포인트 클라우드 ({original_count:,}개)")
                max_points = 100000
                step = original_count // max_points
                indices = np.arange(0, original_count, step)[:max_points]
                points = points[indices]
                self.log(f"강제 다운샘플링: {original_count:,} → {len(points):,}")
            else:
                # 일반 다운샘플링 (균등 간격 방식)
                max_points_str = self.max_points_var.get()
                if max_points_str != "제한없음":
                    max_points = int(max_points_str)
                    if len(points) > max_points:
                        self.log(f"균등 간격 다운샘플링: {len(points):,} → {max_points:,}")
                        # 랜덤이 아닌 균등한 간격으로 샘플링 (원본 형상 더 잘 보존)
                        step = len(points) // max_points
                        indices = np.arange(0, len(points), step)[:max_points]
                        points = points[indices]
                        self.log(f"균등 샘플링 완료: {len(points):,}개 포인트")
                    
            # 포인트 클라우드 객체 생성
            if isinstance(mesh, trimesh.points.PointCloud):
                processed_cloud = trimesh.points.PointCloud(points)
            else:
                processed_cloud = trimesh.points.PointCloud(points)
            
            # 3. 실용적 메시 생성 (urdf_visualizer_pybullet.py 방식 + 개선)
            self.progress_var.set(40)
            self.status_var.set("메시 생성 중...")
            
            # 타임아웃 시작 시간
            mesh_start_time = time.time()
            timeout_seconds = 60  # 60초 제한
            
            mesh_method = getattr(self, 'mesh_method_var', None)
            method = mesh_method.get() if mesh_method else "Convex Hull"
            
            try:
                if method == "Point Cloud Direct":
                    # 포인트 클라우드를 직접 충돌 감지에 사용 (ply_processor 활용)
                    hull_mesh = self.create_point_cloud_collision(points, mesh_start_time, timeout_seconds)
                elif method == "Convex Hull (Original)":
                    # urdf_visualizer_pybullet.py와 동일한 방식
                    self.log("Convex Hull 생성 중... (원본 방식)")
                    hull_mesh = processed_cloud.convex_hull
                elif method == "Alpha Shape":
                    hull_mesh = self.create_alpha_shape_timeout(points, mesh_start_time, timeout_seconds)
                elif method == "Mesh Reconstruction":
                    # 더 정밀한 메시 생성 방법
                    hull_mesh = self.create_precise_mesh(points, mesh_start_time, timeout_seconds)
                else:  # Default: Optimized Convex Hull
                    hull_mesh = processed_cloud.convex_hull
                
                # 타임아웃 체크
                if time.time() - mesh_start_time > timeout_seconds:
                    self.log("메시 생성 타임아웃 - Convex Hull로 폴백")
                    hull_mesh = processed_cloud.convex_hull
                
                self.log(f"{method} 완료: {len(hull_mesh.vertices):,} vertices, {len(hull_mesh.faces):,} faces")
                
            except Exception as e:
                self.log(f"{method} 실패: {e}")
                self.log("Convex Hull로 폴백")
                hull_mesh = processed_cloud.convex_hull
            
            # 4. OBJ 파일 저장
            self.progress_var.set(60)
            self.status_var.set("OBJ 파일 저장 중...")
            
            base_name = os.path.splitext(filename)[0]
            obj_path = f"{base_name}_processed.obj"
            hull_mesh.export(obj_path)
            self.log(f"OBJ 저장: {obj_path}")
            
            # 5. V-HACD 실행 (ply_processor 활용)
            self.progress_var.set(70)
            self.status_var.set("V-HACD 컨벡스 분해 중...")
            
            start_time = time.time()
            
            # ply_processor를 우선 사용
            vhacd_path = None
            if PLY_PROCESSOR_AVAILABLE:
                try:
                    self.log("ply_processor로 V-HACD 실행...")
                    vhacd_path = ply_processor.perform_convex_decomposition(obj_path)
                    processing_time = time.time() - start_time
                    self.log(f"ply_processor V-HACD 완료: {processing_time:.1f}초")
                except Exception as e:
                    self.log(f"ply_processor V-HACD 실패: {e}")
                    vhacd_path = None
            
            # ply_processor 실패시 기존 방식 사용
            if vhacd_path is None:
                self.log("기존 V-HACD 방식 사용...")
                expected_time = self.estimate_vhacd_time(len(hull_mesh.vertices), len(hull_mesh.faces))
                self.vhacd_estimated_time = expected_time
                self.log(f"예상 처리 시간: {expected_time:.1f}초")
                
                vhacd_path = self.run_vhacd(obj_path)
                processing_time = time.time() - start_time
                self.log(f"기존 V-HACD 완료: {processing_time:.1f}초")
            
            # 6. 결과 검증
            self.progress_var.set(90)
            self.status_var.set("결과 검증 중...")
            
            if os.path.exists(vhacd_path):
                self.log(f"V-HACD 완료: {vhacd_path}")
                self.processed_mesh = hull_mesh
                self.processing_stats = {
                    'original_points': len(mesh.vertices) if hasattr(mesh, 'vertices') else 0,
                    'processed_points': len(points),
                    'hull_vertices': len(hull_mesh.vertices),
                    'hull_faces': len(hull_mesh.faces),
                    'obj_path': obj_path,
                    'vhacd_path': vhacd_path
                }
                self.display_results()
            else:
                raise Exception("V-HACD 파일 생성 실패")
                
            # 완료
            self.progress_var.set(100)
            self.status_var.set("처리 완료!")
            self.log("=== 충돌 형상 생성 완료 ===")
            
            # UI 상태 업데이트
            self.test_btn.config(state="normal")
            self.view_btn.config(state="normal")
            
        except Exception as e:
            error_msg = f"처리 실패: {str(e)}"
            self.log(error_msg)
            self.status_var.set("처리 실패")
            messagebox.showerror("오류", error_msg)
            
        finally:
            self.is_processing = False
            self.process_btn.config(state="normal")
            
    def run_vhacd(self, obj_path):
        """V-HACD 실행"""
        try:
            # V-HACD 파라미터
            resolution = int(self.resolution_var.get())
            depth = int(self.depth_var.get())
            concavity = float(self.concavity_var.get())
            
            vhacd_path = obj_path.replace('.obj', '_vhacd.obj')
            
            # PyBullet의 V-HACD 사용 (사용 가능한 경우)
            if PYBULLET_AVAILABLE:
                self.log("PyBullet V-HACD 실행...")
                
                # PyBullet 초기화 (DIRECT 모드)
                physics_client = p.connect(p.DIRECT)
                
                try:
                    # 진행 상태 업데이트를 위한 스레드 시작
                    self.vhacd_running = True
                    progress_thread = threading.Thread(target=self.update_vhacd_progress)
                    progress_thread.daemon = True
                    progress_thread.start()
                    
                    p.vhacd(
                        obj_path,
                        vhacd_path,
                        "vhacd_log.txt",
                        concavity=concavity,
                        resolution=resolution,
                        depth=depth,
                        planeDownsampling=6,
                        convexhullDownsampling=6,
                        alpha=0.04,
                        beta=0.05,
                        gamma=0.01,
                        minVolumePerCH=0.002,
                        maxNumVerticesPerCH=32,
                        pca=0,
                        mode=0,
                        convexhullApproximation=1
                    )
                    self.vhacd_running = False
                    self.log("PyBullet V-HACD 완료")
                    
                finally:
                    p.disconnect(physics_client)
                    
            else:
                # 외부 V-HACD 실행 파일 사용
                self.log("외부 V-HACD 실행...")
                raise Exception("PyBullet을 사용할 수 없어 V-HACD 실행 불가")
                
            return vhacd_path
            
        except Exception as e:
            self.log(f"V-HACD 실행 실패: {e}")
            raise
    
    def estimate_vhacd_time(self, vertices, faces):
        """V-HACD 처리 시간 예상 (경험적 공식)"""
        # 기본 공식: log(vertices) + log(faces) * 복잡도 계수
        base_time = np.log10(max(vertices, 100)) + np.log10(max(faces, 100))
        complexity = int(self.resolution_var.get()) / 50000.0  # 해상도 기반 복잡도
        return base_time * complexity * 2.0  # 대략적인 시간 (초)
    
    def update_vhacd_progress(self):
        """V-HACD 진행 상태 업데이트 (별도 스레드)"""
        progress = 70
        elapsed_time = 0
        
        while hasattr(self, 'vhacd_running') and self.vhacd_running:
            time.sleep(0.5)
            elapsed_time += 0.5
            
            # 로그 파일 모니터링
            if os.path.exists("vhacd_log.txt"):
                try:
                    with open("vhacd_log.txt", 'r') as f:
                        lines = f.readlines()
                        if lines:
                            last_line = lines[-1].strip()
                            # 로그에서 진행 상태 추출 (PyBullet V-HACD는 진행률을 로그에 출력하지 않음)
                            # 대신 경과 시간 기반으로 추정
                            estimated_total = getattr(self, 'vhacd_estimated_time', 10.0)
                            progress_increment = min(15, (elapsed_time / estimated_total) * 15)
                            new_progress = min(85, 70 + progress_increment)
                            self.progress_var.set(int(new_progress))
                            
                            # 상태 메시지 업데이트
                            self.status_var.set(f"V-HACD 진행 중... ({elapsed_time:.1f}s)")
                except:
                    pass
            else:
                # 로그 파일이 없으면 시간 기반으로만 추정
                estimated_total = getattr(self, 'vhacd_estimated_time', 10.0)
                progress_increment = min(15, (elapsed_time / estimated_total) * 15)
                new_progress = min(85, 70 + progress_increment)
                self.progress_var.set(int(new_progress))
                self.status_var.set(f"V-HACD 진행 중... ({elapsed_time:.1f}s)")
            
    def display_results(self):
        """처리 결과 표시"""
        if not self.processing_stats:
            return
            
        self.result_text.delete(1.0, tk.END)
        
        stats = self.processing_stats
        
        result = "=== 처리 결과 ===\n\n"
        result += f"원본 포인트 수: {stats['original_points']:,}개\n"
        result += f"처리된 포인트 수: {stats['processed_points']:,}개\n"
        result += f"Convex Hull 정점 수: {stats['hull_vertices']:,}개\n"
        result += f"Convex Hull 면 수: {stats['hull_faces']:,}개\n\n"
        
        result += "=== 생성된 파일 ===\n"
        result += f"OBJ 파일: {stats['obj_path']}\n"
        result += f"V-HACD 파일: {stats['vhacd_path']}\n\n"
        
        # 압축률 계산
        if stats['original_points'] > 0:
            compression_ratio = stats['processed_points'] / stats['original_points']
            result += f"압축률: {compression_ratio:.2%}\n"
            
        result += "\n=== 다음 단계 ===\n"
        result += "1. 'PyBullet 테스트' 버튼으로 충돌 형상 검증\n"
        result += "2. '3D 비교 뷰' 버튼으로 원본과 비교\n"
        result += "3. 메뉴에서 결과 파일 저장\n"
        
        self.result_text.insert(tk.END, result)
        
        # 결과 탭으로 전환
        self.notebook.select(2)
        
    def test_collision(self):
        """PyBullet으로 충돌 형상 테스트 (향상된 버전)"""
        if not PYBULLET_AVAILABLE:
            messagebox.showwarning("경고", "PyBullet이 설치되어 있지 않습니다.")
            return
            
        if not self.processing_stats:
            messagebox.showwarning("경고", "먼저 충돌 형상을 생성해주세요.")
            return
        
        # 포인트 클라우드 직접 모드 체크
        is_point_cloud_direct = (hasattr(self, 'point_cloud_metadata') and 
                                self.point_cloud_metadata.get('method') == 'point_cloud_direct')
        
        if is_point_cloud_direct:
            # 포인트 클라우드 직접 모드용 테스트 옵션
            test_choice = messagebox.askyesnocancel(
                "포인트 클라우드 충돌 테스트", 
                "Yes: KDTree 고속 충돌 테스트\nNo: PyBullet 구체 충돌 테스트\nCancel: 취소"
            )
        else:
            # 일반 메시 모드용 테스트 옵션
            test_choice = messagebox.askyesnocancel(
                "충돌 테스트 모드 선택", 
                "Yes: 대화형 GUI 테스트\nNo: 자동 충돌 검사\nCancel: 취소"
            )
        
        if test_choice is None:  # Cancel
            return
        elif is_point_cloud_direct:
            if test_choice:  # Yes - KDTree test
                self._run_kdtree_collision_test()
            else:  # No - PyBullet sphere test
                self._run_pybullet_sphere_test()
        else:
            if test_choice:  # Yes - Interactive GUI test
                self._run_interactive_collision_test()
            else:  # No - Automatic collision test
                self._run_automatic_collision_test()
    
    def _run_kdtree_collision_test(self):
        """KDTree를 이용한 고속 포인트 클라우드 충돌 테스트"""
        try:
            self.log("KDTree 고속 충돌 테스트 시작...")
            
            if not hasattr(self, 'point_cloud_metadata'):
                messagebox.showerror("오류", "포인트 클라우드 메타데이터가 없습니다.")
                return
            
            metadata = self.point_cloud_metadata
            collision_points = metadata['collision_points']
            collision_radius = metadata['collision_radius']
            
            # 테스트용 그리드 포인트 생성
            bounds = np.array([collision_points.min(axis=0), collision_points.max(axis=0)])
            test_grid = self.generate_test_grid(bounds, density=20)
            
            self.log(f"테스트 그리드: {len(test_grid):,}개 포인트")
            
            # KDTree 충돌 검사 실행
            start_time = time.time()
            collision_results = []
            
            for i, test_point in enumerate(test_grid):
                is_collision = self.check_point_cloud_collision(test_point, metadata)
                collision_results.append({
                    'point': test_point,
                    'collision': is_collision
                })
                
                if (i + 1) % 1000 == 0:
                    self.log(f"진행: {i+1:,}/{len(test_grid):,}")
            
            processing_time = time.time() - start_time
            collision_count = sum(1 for r in collision_results if r['collision'])
            
            # 결과 분석
            collision_rate = (collision_count / len(test_grid)) * 100
            speed = len(test_grid) / processing_time
            
            result_text = f"""KDTree 충돌 테스트 완료
            
테스트 포인트: {len(test_grid):,}개
충돌 감지: {collision_count:,}개
충돌률: {collision_rate:.1f}%
처리 시간: {processing_time:.2f}초
처리 속도: {speed:,.0f} 포인트/초

포인트 클라우드 직접 충돌 감지가 {'정상적으로' if collision_count > 0 else '비정상적으로'} 작동합니다."""

            messagebox.showinfo("KDTree 테스트 결과", result_text)
            self.log(f"KDTree 테스트 완료: {speed:,.0f} 포인트/초")
            
        except Exception as e:
            error_msg = f"KDTree 테스트 실패: {str(e)}"
            self.log(error_msg)
            messagebox.showerror("오류", error_msg)
    
    def _run_pybullet_sphere_test(self):
        """PyBullet 구체 충돌 테스트 (ply_processor 활용)"""
        try:
            if not PYBULLET_AVAILABLE:
                messagebox.showwarning("경고", "PyBullet이 설치되어 있지 않습니다.")
                return
            
            self.log("PyBullet 고급 시각화 테스트 시작...")
            
            physics_client = p.connect(p.GUI)
            p.setGravity(0, 0, -9.81)
            
            # 바닥 평면 추가
            p.loadURDF("plane.urdf") if os.path.exists("plane.urdf") else None
            
            try:
                # 1. ply_processor를 사용한 PLY 파일 로드 (작동하는 방식 적용)
                if PLY_PROCESSOR_AVAILABLE and hasattr(self, 'current_filename'):
                    self.log("ply_processor로 PLY 파일 로드 중 (작동하는 방식 적용)...")
                    
                    # 파일명만 추출 (경로 제거)
                    import os
                    ply_filename = os.path.basename(self.current_filename)
                    
                    # 작동하는 방식: load_pipe_from_ply 사용
                    ply_body_id = load_pipe_from_ply(
                        ply_filename=ply_filename,
                        fallback_to_default=True,
                        position=[0, 0, 0],
                        scale=0.001,  # mm을 m로 변환 (작동하는 파일과 동일)
                        mass=0.5  # 적절한 질량
                    )
                    
                    if ply_body_id is not None:
                        self.log("원본 PLY 파일이 성공적으로 로드되었습니다!")
                        
                        # 2. 충돌 포인트들을 작은 구체로 표시 (대비용)
                        self._add_collision_point_spheres()
                        
                        # 3. 테스트 객체들
                        test_objects = self.create_enhanced_test_objects()
                        
                        messagebox.showinfo("고급 PyBullet 테스트", 
                            f"향상된 PLY 시각화 완료:\n"
                            f"• 원본 PLY 파일: 고품질 렌더링\n"
                            f"• 충돌 포인트: 작은 구체로 표시\n"
                            f"• 테스트 객체: 충돌 확인용\n"
                            f"• ply_processor 모듈 사용")
                        
                    else:
                        # ply_processor 실패시 기본 방식
                        self.log("ply_processor 로드 실패, 기본 구체 방식 사용")
                        self._run_basic_sphere_test()
                else:
                    # ply_processor 없으면 기본 방식
                    self.log("ply_processor 없음, 기본 구체 방식 사용")
                    self._run_basic_sphere_test()
                
            finally:
                pass
                
        except Exception as e:
            error_msg = f"PyBullet 테스트 실패: {str(e)}"
            self.log(error_msg)
            messagebox.showerror("오류", error_msg)
    
    def _add_collision_point_spheres(self):
        """충돌 포인트들을 작은 구체로 추가"""
        try:
            metadata = self.point_cloud_metadata
            collision_points = metadata['collision_points']
            collision_radius = metadata['collision_radius']
            
            # 작은 빨간 구체로 충돌 포인트 표시
            max_spheres = min(200, len(collision_points))  # 성능을 위해 제한
            small_radius = collision_radius * 0.5  # 더 작게
            
            collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=small_radius)
            visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=small_radius, 
                                             rgbaColor=[1.0, 0.2, 0.2, 0.6])  # 반투명 빨강
            
            self.log(f"충돌 포인트 구체 생성: {max_spheres}개")
            
            for i, point in enumerate(collision_points[:max_spheres]):
                p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=collision_shape,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=point.tolist()
                )
        except Exception as e:
            self.log(f"충돌 포인트 구체 생성 실패: {e}")
    
    def _run_basic_sphere_test(self):
        """기본 구체 테스트 (원래 방식)"""
        try:
            metadata = self.point_cloud_metadata
            collision_points = metadata['collision_points']
            collision_radius = metadata['collision_radius']
            
            # 구체들을 PyBullet에 생성 (그라데이션 색상)
            max_spheres = min(500, len(collision_points))
            
            for i, point in enumerate(collision_points[:max_spheres]):
                # 높이에 따른 그라데이션 색상
                height_ratio = (point[2] - collision_points[:, 2].min()) / \
                              (collision_points[:, 2].max() - collision_points[:, 2].min() + 1e-6)
                
                # 파란색에서 빨간색으로 그라데이션
                color = [height_ratio, 0.3, 1.0 - height_ratio, 0.8]
                
                collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=collision_radius)
                visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=collision_radius, 
                                                 rgbaColor=color)
                
                p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=collision_shape,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=point.tolist()
                )
            
            self.log(f"그라데이션 구체 생성 완료: {max_spheres}개")
            
        except Exception as e:
            self.log(f"기본 구체 테스트 실패: {e}")
    
    def create_enhanced_test_objects(self):
        """향상된 테스트 객체 생성"""
        test_objects = []
        
        try:
            # 1. 다양한 크기의 테스트 구들
            for i, (radius, color, pos) in enumerate([
                (0.01, [1, 0, 0, 1], [0, 0, 0.5]),     # 작은 빨간 구
                (0.02, [0, 1, 0, 1], [0.1, 0, 0.5]),   # 중간 녹색 구  
                (0.03, [0, 0, 1, 1], [-0.1, 0, 0.5])   # 큰 파란 구
            ]):
                sphere_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
                sphere_visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
                sphere_id = p.createMultiBody(0.1, sphere_shape, sphere_visual, pos)
                test_objects.append((f"Test Sphere {i+1}", sphere_id))
            
            # 2. 회전하는 박스
            box_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.01, 0.03])
            box_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.01, 0.03], 
                                           rgbaColor=[1, 1, 0, 1])
            box_id = p.createMultiBody(0.1, box_shape, box_visual, [0, 0.1, 0.5])
            test_objects.append(("Rotating Box", box_id))
            
            # 3. 원통형 객체
            cylinder_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.015, height=0.05)
            cylinder_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.015, length=0.05,
                                                rgbaColor=[1, 0, 1, 1])
            cylinder_id = p.createMultiBody(0.1, cylinder_shape, cylinder_visual, [0, -0.1, 0.5])
            test_objects.append(("Test Cylinder", cylinder_id))
            
        except Exception as e:
            self.log(f"향상된 테스트 객체 생성 실패: {e}")
        
        return test_objects
    
    def generate_test_grid(self, bounds, density=10):
        """테스트용 그리드 포인트 생성"""
        min_bounds, max_bounds = bounds
        
        # 각 축에 대해 균등한 간격의 포인트 생성
        x_points = np.linspace(min_bounds[0], max_bounds[0], density)
        y_points = np.linspace(min_bounds[1], max_bounds[1], density)
        z_points = np.linspace(min_bounds[2], max_bounds[2], density)
        
        # 3D 그리드 생성
        grid_points = []
        for x in x_points:
            for y in y_points:
                for z in z_points:
                    grid_points.append([x, y, z])
        
        return np.array(grid_points)
    
    def create_test_objects_for_spheres(self):
        """구체 테스트용 동적 객체들 생성"""
        test_objects = []
        
        try:
            # 1. 테스트 구
            sphere_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.02)
            sphere_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=[1, 0, 0, 1])
            sphere_id = p.createMultiBody(0.1, sphere_shape, sphere_visual, [0, 0, 0.5])
            test_objects.append(("Test Sphere", sphere_id))
            
            # 2. 테스트 박스
            box_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
            box_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02], rgbaColor=[0, 1, 0, 1])
            box_id = p.createMultiBody(0.1, box_shape, box_visual, [0.1, 0, 0.5])
            test_objects.append(("Test Box", box_id))
            
        except Exception as e:
            self.log(f"테스트 객체 생성 실패: {e}")
        
        return test_objects
    
    def _run_interactive_collision_test(self):
        """대화형 충돌 테스트 (ply_processor 활용)"""
        try:
            self.log("대화형 충돌 테스트 시작...")
            
            # PyBullet GUI 초기화
            physics_client = p.connect(p.GUI)
            p.setGravity(0, 0, -9.81)
            p.setAdditionalSearchPath(".")
            
            # 바닥 평면 추가
            plane_id = p.loadURDF("plane.urdf") if os.path.exists("plane.urdf") else None
            
            # ply_processor를 우선 사용 (작동하는 방식 적용)
            main_body = None
            if PLY_PROCESSOR_AVAILABLE and hasattr(self, 'current_filename'):
                self.log("ply_processor로 향상된 PLY 렌더링 (작동하는 방식 적용)...")
                
                # 파일명만 추출 (경로 제거)
                import os
                ply_filename = os.path.basename(self.current_filename)
                
                # 작동하는 방식: load_pipe_from_ply 사용
                main_body = load_pipe_from_ply(
                    ply_filename=ply_filename,
                    fallback_to_default=True,
                    position=[0, 0, 0],
                    scale=0.001,  # mm을 m로 변환 (작동하는 파일과 동일)
                    mass=0.5  # 적절한 질량 설정
                )
                
                if main_body is not None:
                    self.log("ply_processor로 PLY 로드 성공!")
            
            # ply_processor 실패시 기존 방식 사용
            if main_body is None and hasattr(self, 'processing_stats'):
                self.log("기존 메시 파일 방식 사용...")
                
                vhacd_path = self.processing_stats['vhacd_path']
                obj_path = self.processing_stats['obj_path']
                
                # 메인 충돌 객체 생성 (기존 방식)
                collision_shape = p.createCollisionShape(
                    p.GEOM_MESH,
                    fileName=vhacd_path,
                    meshScale=[1.0, 1.0, 1.0]
                )
                
                visual_shape = p.createVisualShape(
                    p.GEOM_MESH,
                    fileName=obj_path,
                    meshScale=[1.0, 1.0, 1.0],
                    rgbaColor=[0.7, 0.7, 1.0, 0.8]
                )
                
                main_body = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=collision_shape,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=[0, 0, 0]
                )
            
            if main_body is None:
                self.log("충돌 객체 생성 실패!")
                return
            
            # 다양한 테스트 객체들
            test_objects = []
            
            # 1. 빨간 구 (떨어지는 테스트)
            sphere_col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.05)
            sphere_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 1])
            sphere_id = p.createMultiBody(0.1, sphere_col, sphere_vis, [0, 0, 1])
            test_objects.append(("Red Sphere", sphere_id))
            
            # 2. 파란 상자
            box_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05])
            box_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05], rgbaColor=[0, 0, 1, 1])
            box_id = p.createMultiBody(0.1, box_col, box_vis, [0.2, 0, 1])
            test_objects.append(("Blue Box", box_id))
            
            # 3. 녹색 캡슐
            capsule_col = p.createCollisionShape(p.GEOM_CAPSULE, radius=0.03, height=0.1)
            capsule_vis = p.createVisualShape(p.GEOM_CAPSULE, radius=0.03, length=0.1, rgbaColor=[0, 1, 0, 1])
            capsule_id = p.createMultiBody(0.1, capsule_col, capsule_vis, [-0.2, 0, 1])
            test_objects.append(("Green Capsule", capsule_id))
            
            self.log(f"테스트 객체 생성: {len(test_objects)}개")
            self.log("PyBullet GUI에서 상호작용하세요:")
            self.log("- 마우스로 카메라 조작")
            self.log("- 객체들이 메시와 충돌하는지 확인")
            self.log("- 'g' 키로 중력 토글")
            self.log("- 'r' 키로 리셋")
            
            # GUI 정보 표시
            messagebox.showinfo("대화형 테스트", 
                "PyBullet 창에서 충돌 테스트 진행:\n"
                "• 다양한 색상의 객체들이 메시와 충돌\n"
                "• 마우스로 카메라 조작 가능\n"
                "• g키: 중력 on/off\n"
                "• r키: 객체 위치 리셋\n"
                "• 창을 닫으면 테스트 종료")
            
            # 키보드 콜백 설정 (PyBullet GUI에서 지원되지 않을 수 있음)
            # 대신 간단한 루프로 처리
            
        except Exception as e:
            error_msg = f"대화형 테스트 실패: {str(e)}"
            self.log(error_msg)
            messagebox.showerror("오류", error_msg)
    
    def _run_automatic_collision_test(self):
        """자동 충돌 검사"""
        try:
            self.log("자동 충돌 검사 시작...")
            
            # PyBullet DIRECT 모드 (GUI 없음)
            physics_client = p.connect(p.DIRECT)
            
            vhacd_path = self.processing_stats['vhacd_path']
            
            # 충돌 형상 생성
            collision_shape = p.createCollisionShape(
                p.GEOM_MESH,
                fileName=vhacd_path
            )
            
            main_body = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_shape,
                basePosition=[0, 0, 0]
            )
            
            # 테스트 포인트들 (그리드 형태로 배치)
            test_results = []
            test_count = 0
            collision_count = 0
            
            # 메시 바운딩 박스 계산
            if hasattr(self.current_mesh, 'vertices'):
                vertices = self.current_mesh.vertices
                min_bounds = vertices.min(axis=0)
                max_bounds = vertices.max(axis=0)
                center = (min_bounds + max_bounds) / 2
            else:
                center = np.array([0, 0, 0])
                min_bounds = np.array([-1, -1, -1])
                max_bounds = np.array([1, 1, 1])
            
            self.log(f"테스트 영역: {min_bounds} ~ {max_bounds}")
            
            # 테스트 구 생성
            test_sphere = p.createCollisionShape(p.GEOM_SPHERE, radius=0.01)
            
            # 그리드 테스트
            for i, x in enumerate(np.linspace(min_bounds[0], max_bounds[0], 5)):
                for j, y in enumerate(np.linspace(min_bounds[1], max_bounds[1], 5)):
                    for k, z in enumerate(np.linspace(min_bounds[2], max_bounds[2], 5)):
                        test_pos = [x, y, z]
                        
                        # 테스트 구 배치
                        test_body = p.createMultiBody(
                            baseMass=0,
                            baseCollisionShapeIndex=test_sphere,
                            basePosition=test_pos
                        )
                        
                        # 충돌 검사
                        contact_points = p.getContactPoints(main_body, test_body)
                        is_collision = len(contact_points) > 0
                        
                        test_results.append({
                            'position': test_pos,
                            'collision': is_collision,
                            'contact_count': len(contact_points)
                        })
                        
                        if is_collision:
                            collision_count += 1
                        test_count += 1
                        
                        # 테스트 바디 제거
                        p.removeBody(test_body)
            
            # 결과 분석
            collision_rate = (collision_count / test_count) * 100
            
            self.log(f"충돌 테스트 결과:")
            self.log(f"- 총 테스트 포인트: {test_count}")
            self.log(f"- 충돌 감지: {collision_count}")
            self.log(f"- 충돌률: {collision_rate:.1f}%")
            
            # 결과 상세 표시
            result_summary = f"""자동 충돌 테스트 완료
            
테스트 범위: {min_bounds} ~ {max_bounds}
총 테스트 포인트: {test_count}
충돌 감지된 포인트: {collision_count}
충돌 감지율: {collision_rate:.1f}%

충돌 형상이 {'정상적으로' if collision_count > 0 else '제대로'} 작동하고 있습니다."""
            
            messagebox.showinfo("자동 테스트 결과", result_summary)
            
            p.disconnect(physics_client)
            
        except Exception as e:
            error_msg = f"자동 테스트 실패: {str(e)}"
            self.log(error_msg)
            messagebox.showerror("오류", error_msg)
            
    def show_comparison(self):
        """원본과 처리된 메시 비교 표시 (향상된 버전)"""
        if not self.processed_mesh or not self.current_mesh:
            messagebox.showwarning("경고", "먼저 충돌 형상을 생성해주세요.")
            return
        
        # 비교 모드 선택
        comparison_choice = messagebox.askyesnocancel(
            "비교 뷰 모드 선택",
            "Yes: 상세 분석 뷰 (6개 패널)\nNo: 간단 비교 뷰 (2개 패널)\nCancel: 취소"
        )
        
        if comparison_choice is None:  # Cancel
            return
        elif comparison_choice:  # Yes - Detailed view
            thread = threading.Thread(target=self.create_detailed_comparison)
        else:  # No - Simple view
            thread = threading.Thread(target=self.create_simple_comparison)
        
        try:
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            error_msg = f"비교 뷰 생성 실패: {str(e)}"
            self.log(error_msg)
            messagebox.showerror("오류", error_msg)
            
    def create_simple_comparison(self):
        """간단한 비교 뷰 생성 (2개 패널)"""
        try:
            self.log("간단 비교 뷰 생성 중...")
            
            original_mesh = self.current_mesh
            processed_mesh = self.processed_mesh
            
            # 원본 포인트 추출
            if isinstance(original_mesh, trimesh.points.PointCloud):
                original_points = original_mesh.vertices
            else:
                original_points = original_mesh.vertices
                
            # 처리된 메시 포인트
            processed_points = processed_mesh.vertices
            
            # 다운샘플링 (표시용)
            if len(original_points) > 20000:
                indices = np.random.choice(len(original_points), 20000, replace=False)
                original_points = original_points[indices]
                
            # 플롯 생성
            fig = plt.figure(figsize=(16, 8))
            
            # 원본 (왼쪽)
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2],
                       c=original_points[:, 2], cmap='viridis', s=0.5, alpha=0.6)
            ax1.set_title(f'Original Point Cloud\n({len(original_points):,} points displayed)')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            
            # 처리된 메시 (오른쪽)
            ax2 = fig.add_subplot(122, projection='3d')
            ax2.plot_trisurf(processed_points[:, 0], processed_points[:, 1], processed_points[:, 2],
                           triangles=processed_mesh.faces, alpha=0.7, cmap='plasma')
            ax2.set_title(f'Processed Convex Hull\n({len(processed_points):,} vertices, {len(processed_mesh.faces):,} faces)')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            
            plt.suptitle(f'Simple Comparison: {self.current_filename}', fontsize=14)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            self.log(f"간단 비교 뷰 생성 실패: {e}")
    
    def create_detailed_comparison(self):
        """상세한 분석 비교 뷰 생성 (6개 패널)"""
        try:
            self.log("상세 분석 비교 뷰 생성 중...")
            
            original_mesh = self.current_mesh
            processed_mesh = self.processed_mesh
            
            # 원본 포인트 추출
            if isinstance(original_mesh, trimesh.points.PointCloud):
                original_points = original_mesh.vertices
                is_point_cloud = True
            else:
                original_points = original_mesh.vertices
                is_point_cloud = False
                
            # 처리된 메시 포인트
            processed_points = processed_mesh.vertices
            
            # 다운샘플링 (표시용)
            display_points = original_points
            if len(original_points) > 15000:
                indices = np.random.choice(len(original_points), 15000, replace=False)
                display_points = original_points[indices]
            
            # 6개 패널 플롯 생성
            fig = plt.figure(figsize=(20, 12))
            
            # 1. 원본 3D 뷰
            ax1 = fig.add_subplot(231, projection='3d')
            scatter1 = ax1.scatter(display_points[:, 0], display_points[:, 1], display_points[:, 2],
                       c=display_points[:, 2], cmap='viridis', s=1, alpha=0.7)
            ax1.set_title(f'Original {"Point Cloud" if is_point_cloud else "Mesh"}\n({len(display_points):,} points)')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            
            # 2. 처리된 메시 3D 뷰
            ax2 = fig.add_subplot(232, projection='3d')
            ax2.plot_trisurf(processed_points[:, 0], processed_points[:, 1], processed_points[:, 2],
                           triangles=processed_mesh.faces, alpha=0.8, cmap='plasma')
            ax2.set_title(f'Convex Hull Mesh\n({len(processed_points):,} vertices)')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            
            # 3. 오버레이 뷰 (원본 + 처리됨)
            ax3 = fig.add_subplot(233, projection='3d')
            ax3.scatter(display_points[:, 0], display_points[:, 1], display_points[:, 2],
                       c='blue', s=0.5, alpha=0.3, label='Original')
            ax3.plot_trisurf(processed_points[:, 0], processed_points[:, 1], processed_points[:, 2],
                           triangles=processed_mesh.faces, alpha=0.6, color='red', label='Hull')
            ax3.set_title('Overlay Comparison')
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.set_zlabel('Z')
            ax3.legend()
            
            # 4. XY 투영 비교
            ax4 = fig.add_subplot(234)
            ax4.scatter(display_points[:, 0], display_points[:, 1], c='blue', s=1, alpha=0.5, label='Original')
            hull_2d = processed_points[:, :2]  # XY projection
            try:
                from scipy.spatial import ConvexHull
                hull_xy = ConvexHull(hull_2d)
                for simplex in hull_xy.simplices:
                    ax4.plot(hull_2d[simplex, 0], hull_2d[simplex, 1], 'r-', alpha=0.8)
                ax4.fill(hull_2d[hull_xy.vertices, 0], hull_2d[hull_xy.vertices, 1], 'red', alpha=0.2)
            except:
                ax4.scatter(processed_points[:, 0], processed_points[:, 1], c='red', s=2, alpha=0.7)
            ax4.set_title('XY Projection')
            ax4.set_xlabel('X')
            ax4.set_ylabel('Y')
            ax4.set_aspect('equal')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # 5. 통계 비교 (히스토그램)
            ax5 = fig.add_subplot(235)
            
            # 원본과 처리된 포인트의 분포 비교
            orig_distances = np.linalg.norm(original_points - original_points.mean(axis=0), axis=1)
            proc_distances = np.linalg.norm(processed_points - processed_points.mean(axis=0), axis=1)
            
            ax5.hist(orig_distances, bins=50, alpha=0.5, label='Original', color='blue', density=True)
            ax5.hist(proc_distances, bins=30, alpha=0.5, label='Processed', color='red', density=True)
            ax5.set_title('Distance from Center Distribution')
            ax5.set_xlabel('Distance')
            ax5.set_ylabel('Density')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            
            # 6. 처리 통계 정보
            ax6 = fig.add_subplot(236)
            ax6.axis('off')
            
            # 통계 계산
            orig_bbox = original_points.max(axis=0) - original_points.min(axis=0)
            proc_bbox = processed_points.max(axis=0) - processed_points.min(axis=0)
            orig_volume = np.prod(orig_bbox)
            proc_volume = processed_mesh.volume if hasattr(processed_mesh, 'volume') else np.prod(proc_bbox)
            
            reduction_ratio = len(processed_points) / len(original_points) * 100
            
            stats_text = f"""Processing Statistics:
            
Original:
• Points: {len(original_points):,}
• Bounding Box: {orig_bbox[0]:.3f} × {orig_bbox[1]:.3f} × {orig_bbox[2]:.3f}
• Est. Volume: {orig_volume:.6f}

Processed:
• Vertices: {len(processed_points):,}
• Faces: {len(processed_mesh.faces):,}
• Bounding Box: {proc_bbox[0]:.3f} × {proc_bbox[1]:.3f} × {proc_bbox[2]:.3f}
• Volume: {proc_volume:.6f}

Reduction:
• Point Reduction: {100-reduction_ratio:.1f}%
• Memory Efficiency: {len(processed_points)/len(original_points)*100:.1f}%

Processing Time:
• Parameters: res={self.resolution_var.get()}, depth={self.depth_var.get()}
• Concavity: {self.concavity_var.get()}"""
            
            ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=9,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
            
            plt.suptitle(f'Detailed Analysis: {self.current_filename}', fontsize=16)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            self.log(f"상세 비교 뷰 생성 실패: {e}")
            # Fallback to simple comparison
            self.create_simple_comparison()
            
    def save_obj(self):
        """OBJ 파일 저장"""
        if not self.processing_stats:
            messagebox.showwarning("경고", "먼저 충돌 형상을 생성해주세요.")
            return
            
        filename = filedialog.asksaveasfilename(
            title="OBJ 파일 저장",
            defaultextension=".obj",
            filetypes=[("OBJ files", "*.obj"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                import shutil
                shutil.copy(self.processing_stats['obj_path'], filename)
                messagebox.showinfo("정보", f"OBJ 파일 저장 완료:\n{filename}")
            except Exception as e:
                messagebox.showerror("오류", f"파일 저장 실패: {e}")
                
    def save_vhacd(self):
        """V-HACD 결과 파일 저장"""
        if not self.processing_stats:
            messagebox.showwarning("경고", "먼저 충돌 형상을 생성해주세요.")
            return
            
        filename = filedialog.asksaveasfilename(
            title="V-HACD 파일 저장",
            defaultextension=".obj",
            filetypes=[("OBJ files", "*.obj"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                import shutil
                shutil.copy(self.processing_stats['vhacd_path'], filename)
                messagebox.showinfo("정보", f"V-HACD 파일 저장 완료:\n{filename}")
            except Exception as e:
                messagebox.showerror("오류", f"파일 저장 실패: {e}")
                
    def reset_settings(self):
        """설정 초기화"""
        self.max_points_var.set("100000")
        self.resolution_var.set("50000")
        self.depth_var.set(6)
        self.concavity_var.set(0.05)
        self.log("설정이 초기화되었습니다.")
        
    def clear_cache(self):
        """캐시 파일 정리"""
        try:
            cache_files = []
            for f in os.listdir('.'):
                if f.endswith(('_processed.obj', '_vhacd.obj', '.cache_info')):
                    cache_files.append(f)
                    
            if cache_files:
                result = messagebox.askyesno("확인", f"{len(cache_files)}개의 캐시 파일을 삭제하시겠습니까?")
                if result:
                    for f in cache_files:
                        os.remove(f)
                    self.log(f"{len(cache_files)}개 캐시 파일 삭제 완료")
            else:
                messagebox.showinfo("정보", "삭제할 캐시 파일이 없습니다.")
                
        except Exception as e:
            messagebox.showerror("오류", f"캐시 정리 실패: {e}")
            
    def run(self):
        """GUI 실행"""
        self.root.mainloop()

    def on_vhacd_mode_change(self, event=None):
        """V-HACD 처리 모드 변경 시 파라미터 자동 조정"""
        mode = self.vhacd_mode_var.get()
        
        if mode == "고속 모드":
            # urdf_visualizer_pybullet.py의 fast_mode 파라미터
            self.resolution_var.set("50000")
            self.depth_var.set(6)
            self.concavity_var.set(0.05)
            self.log("고속 모드: 성능 우선 (빠르지만 단순한 형상)")
            
        elif mode == "일반 모드":
            # urdf_visualizer_pybullet.py의 일반 모드 파라미터
            self.resolution_var.set("75000")
            self.depth_var.set(8)
            self.concavity_var.set(0.025)
            self.log("일반 모드: 품질과 성능 균형 (권장)")
            
        elif mode == "고품질 모드":
            # 더 정밀한 결과를 위한 파라미터
            self.resolution_var.set("100000")
            self.depth_var.set(10)
            self.concavity_var.set(0.015)
            self.log("고품질 모드: 품질 우선 (느리지만 정밀한 형상)")
    
    def on_mesh_method_change(self, event=None):
        """메시 생성 방법 변경 시 호출"""
        method = self.mesh_method_var.get()
        self.log(f"메시 생성 방법 변경: {method}")
        
        # 각 방법에 대한 설명 출력
        descriptions = {
            "Point Cloud Direct": "포인트 클라우드 직접 충돌 (형상 100% 보존, 빠름)",
            "Convex Hull (Original)": "urdf_visualizer_pybullet.py와 동일 방식 (가장 안정적)",
            "Alpha Shape": "형상을 더 정확히 유지 (오목한 부분도 보존)",
            "Mesh Reconstruction": "고정밀 표면 재구성 (최고 품질)"
        }
        
        if method in descriptions:
            self.log(f"  → {descriptions[method]}")
    
    def create_point_cloud_collision(self, points, start_time, timeout_seconds):
        """포인트 클라우드 직접 충돌 감지용 처리"""
        try:
            if time.time() - start_time > timeout_seconds:
                raise Exception("타임아웃")
            
            self.log("포인트 클라우드 직접 충돌 처리 중...")
            
            # 1. 최적 포인트 수로 스마트 샘플링
            optimal_count = int(self.optimal_points_var.get())
            collision_radius = self.collision_radius_var.get()
            
            if len(points) > optimal_count:
                self.log(f"충돌 최적화 샘플링: {len(points):,} → {optimal_count:,}")
                # 공간적으로 균등한 샘플링 (KDTree 기반)
                optimized_points = self.spatial_uniform_sampling(points, optimal_count)
            else:
                optimized_points = points
            
            # 2. 포인트 클라우드를 구체들로 변환하여 PyBullet에서 사용
            sphere_mesh = self.create_multi_sphere_collision(optimized_points, collision_radius)
            
            # 3. 충돌 감지용 메타데이터 저장
            self.point_cloud_metadata = {
                'original_points': points,
                'collision_points': optimized_points,
                'collision_radius': collision_radius,
                'method': 'point_cloud_direct',
                'kdtree': self.build_collision_kdtree(optimized_points, collision_radius)
            }
            
            self.log(f"포인트 클라우드 직접 충돌 완료: {len(optimized_points):,}개 구체")
            return sphere_mesh
            
        except Exception as e:
            self.log(f"포인트 클라우드 직접 충돌 실패: {e}")
            raise
    
    def spatial_uniform_sampling(self, points, target_count):
        """공간적으로 균등한 포인트 샘플링 (중요 영역 보존)"""
        try:
            if not SKLEARN_AVAILABLE:
                # sklearn이 없으면 균등 간격 샘플링
                step = len(points) // target_count
                return points[::step][:target_count]
            
            # 1. k-means 클러스터링으로 공간 분할
            from sklearn.cluster import KMeans
            
            n_clusters = min(target_count // 2, 100)  # 클러스터 수 제한
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(points)
            
            # 2. 각 클러스터에서 대표 포인트들 선택
            selected_points = []
            points_per_cluster = target_count // n_clusters
            
            for i in range(n_clusters):
                cluster_points = points[clusters == i]
                if len(cluster_points) > 0:
                    # 클러스터 중심에 가까운 포인트들 선택
                    center = kmeans.cluster_centers_[i]
                    distances = np.linalg.norm(cluster_points - center, axis=1)
                    
                    n_select = min(points_per_cluster, len(cluster_points))
                    closest_indices = np.argsort(distances)[:n_select]
                    selected_points.extend(cluster_points[closest_indices])
            
            # 3. 부족한 포인트는 랜덤하게 추가
            if len(selected_points) < target_count:
                remaining = target_count - len(selected_points)
                all_indices = set(range(len(points)))
                used_indices = set()
                
                # 이미 선택된 포인트들의 인덱스 찾기 (근사)
                for sel_point in selected_points:
                    distances = np.linalg.norm(points - sel_point, axis=1)
                    used_indices.add(np.argmin(distances))
                
                available_indices = list(all_indices - used_indices)
                if available_indices:
                    additional_indices = np.random.choice(
                        available_indices, 
                        min(remaining, len(available_indices)), 
                        replace=False
                    )
                    selected_points.extend(points[additional_indices])
            
            return np.array(selected_points[:target_count])
            
        except Exception as e:
            self.log(f"공간 샘플링 실패: {e}, 균등 샘플링으로 대체")
            step = len(points) // target_count
            return points[::step][:target_count]
    
    def create_multi_sphere_collision(self, points, radius):
        """다중 구체로 구성된 충돌 메시 생성"""
        try:
            # 각 포인트를 중심으로 하는 구체들을 하나의 메시로 결합
            spheres = []
            
            for i, point in enumerate(points):
                # 간단한 구체 메시 생성
                sphere = trimesh.creation.icosphere(subdivisions=1, radius=radius)
                sphere.vertices += point  # 포인트 위치로 이동
                spheres.append(sphere)
                
                # 너무 많은 구체는 메모리 문제 발생 가능
                if i > 1000:  # 1000개 제한
                    self.log(f"구체 수 제한으로 {i+1}개에서 중단")
                    break
            
            if not spheres:
                raise Exception("구체 생성 실패")
            
            # 모든 구체를 하나의 메시로 결합
            combined_mesh = trimesh.util.concatenate(spheres)
            
            return combined_mesh
            
        except Exception as e:
            self.log(f"다중 구체 생성 실패: {e}")
            # 폴백: 단일 convex hull
            cloud = trimesh.points.PointCloud(points)
            return cloud.convex_hull
    
    def build_collision_kdtree(self, points, radius):
        """고속 충돌 감지를 위한 KDTree 구축"""
        try:
            if SCIPY_AVAILABLE:
                kdtree = cKDTree(points)
                return {
                    'tree': kdtree,
                    'points': points,
                    'radius': radius
                }
        except:
            pass
        return None
    
    def check_point_cloud_collision(self, test_point, metadata=None):
        """포인트 클라우드와 테스트 포인트 간 충돌 검사"""
        try:
            if metadata is None:
                metadata = getattr(self, 'point_cloud_metadata', None)
            
            if metadata is None:
                return False
            
            kdtree_info = metadata.get('kdtree')
            if kdtree_info is None:
                return False
            
            tree = kdtree_info['tree']
            radius = kdtree_info['radius']
            
            # 반지름 내의 포인트들 검색
            neighbors = tree.query_ball_point(test_point, radius)
            
            return len(neighbors) > 0
            
        except:
            return False
    
    def create_alpha_shape_timeout(self, points, start_time, timeout_seconds):
        """Alpha Shape (타임아웃 보호 포함)"""
        try:
            if time.time() - start_time > timeout_seconds:
                raise Exception("타임아웃")
                
            alpha = self.alpha_var.get()
            self.log(f"Alpha Shape 생성 중 (alpha={alpha:.3f})...")
            
            if not SCIPY_AVAILABLE:
                raise Exception("SciPy 없음")
            
            # 큰 데이터셋에서는 샘플링
            work_points = points
            if len(points) > 10000:
                self.log(f"Alpha Shape용 추가 샘플링: {len(points):,} → 10,000")
                step = len(points) // 10000
                indices = np.arange(0, len(points), step)[:10000]
                work_points = points[indices]
            
            # Delaunay triangulation
            tri = Delaunay(work_points)
            
            # 표면 면 추출 (간단한 방식)
            hull = ConvexHull(work_points)
            hull_faces = hull.simplices
            
            # Alpha 필터링을 단순화
            filtered_faces = []
            for face in hull_faces:
                # 삼각형 면적 기반 필터링
                triangle_points = work_points[face]
                area = 0.5 * np.linalg.norm(np.cross(
                    triangle_points[1] - triangle_points[0],
                    triangle_points[2] - triangle_points[0]
                ))
                
                if area < alpha:
                    filtered_faces.append(face)
            
            if not filtered_faces:
                # Alpha가 너무 작으면 원본 hull 사용
                filtered_faces = hull_faces
                
            alpha_mesh = trimesh.Trimesh(vertices=work_points, faces=filtered_faces)
            
            self.log(f"Alpha Shape 완료")
            return alpha_mesh
            
        except Exception as e:
            self.log(f"Alpha Shape 실패: {e}")
            raise
    
    def create_precise_mesh(self, points, start_time, timeout_seconds):
        """고정밀 메시 재구성 (실용적 접근)"""
        try:
            if time.time() - start_time > timeout_seconds:
                raise Exception("타임아웃")
                
            self.log("고정밀 메시 재구성 중...")
            
            # 1. 포인트 정제 및 노이즈 제거
            work_points = self.clean_point_cloud(points)
            
            if time.time() - start_time > timeout_seconds:
                raise Exception("타임아웃")
            
            # 2. 다단계 convex hull 접근
            if OPEN3D_AVAILABLE:
                precise_mesh = self.create_o3d_reconstruction(work_points)
            else:
                # Open3D 없으면 개선된 convex hull 사용
                precise_mesh = self.create_enhanced_convex_hull(work_points)
            
            self.log("고정밀 메시 재구성 완료")
            return precise_mesh
            
        except Exception as e:
            self.log(f"고정밀 메시 재구성 실패: {e}")
            raise
    
    def clean_point_cloud(self, points):
        """포인트 클라우드 정제"""
        try:
            # 중복점 제거
            unique_points, indices = np.unique(points, axis=0, return_index=True)
            
            if len(unique_points) < len(points):
                self.log(f"중복점 제거: {len(points):,} → {len(unique_points):,}")
            
            # 이상치 제거 (간단한 통계적 방법)
            if len(unique_points) > 100:
                center = unique_points.mean(axis=0)
                distances = np.linalg.norm(unique_points - center, axis=1)
                threshold = np.percentile(distances, 95)  # 상위 5% 제거
                
                inliers = unique_points[distances <= threshold]
                if len(inliers) >= len(unique_points) * 0.8:  # 80% 이상 유지
                    unique_points = inliers
                    self.log(f"이상치 제거: {len(distances):,} → {len(inliers):,}")
            
            return unique_points
            
        except:
            return points
    
    def create_o3d_reconstruction(self, points):
        """Open3D를 이용한 고정밀 재구성"""
        try:
            # Open3D 포인트 클라우드 생성
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # 법선 추정
            pcd.estimate_normals()
            
            # Poisson 재구성 (파라미터 최적화)
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, 
                depth=8,  # 적당한 깊이
                width=0, 
                scale=1.0, 
                linear_fit=False
            )
            
            # 불필요한 부분 제거
            bbox = pcd.get_axis_aligned_bounding_box()
            mesh = mesh.crop(bbox)
            
            # Open3D → trimesh 변환
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            
            return trimesh.Trimesh(vertices=vertices, faces=faces)
            
        except:
            raise Exception("Open3D 재구성 실패")
    
    def create_enhanced_convex_hull(self, points):
        """개선된 convex hull (다단계 접근)"""
        try:
            # 기본 convex hull
            cloud = trimesh.points.PointCloud(points)
            base_hull = cloud.convex_hull
            
            # 더 세밀한 표면을 위해 subdivide
            base_hull = base_hull.subdivide()
            
            return base_hull
            
        except:
            raise Exception("Enhanced convex hull 실패")
    
    def create_alpha_shape(self, points):
        """Alpha Shape을 이용한 메시 생성 (형상 보존이 우수)"""
        try:
            self.log("Alpha Shape 생성 중...")
            alpha = self.alpha_var.get()
            
            if not SCIPY_AVAILABLE:
                self.log("SciPy가 없어 Convex Hull로 대체")
                return trimesh.points.PointCloud(points).convex_hull
            
            # Alpha Shape 구현 (Delaunay + Alpha 필터링)
            cloud = trimesh.points.PointCloud(points)
            
            # 3D Delaunay triangulation
            tri = Delaunay(points)
            
            # Alpha 필터링: circumradius가 alpha보다 작은 tetrahedron만 유지
            valid_simplices = []
            for simplex in tri.simplices:
                # 사면체의 circumradius 계산
                tetra_points = points[simplex]
                circumradius = self.compute_circumradius_3d(tetra_points)
                
                if circumradius < alpha:
                    # 사면체의 각 면을 추가
                    faces = [
                        [simplex[0], simplex[1], simplex[2]],
                        [simplex[0], simplex[1], simplex[3]],
                        [simplex[0], simplex[2], simplex[3]],
                        [simplex[1], simplex[2], simplex[3]]
                    ]
                    valid_simplices.extend(faces)
            
            if not valid_simplices:
                self.log("Alpha Shape 생성 실패, Convex Hull로 대체")
                return cloud.convex_hull
                
            # 중복 면 제거
            unique_faces = []
            face_set = set()
            for face in valid_simplices:
                face_tuple = tuple(sorted(face))
                if face_tuple not in face_set:
                    face_set.add(face_tuple)
                    unique_faces.append(face)
            
            if len(unique_faces) < 4:
                self.log("Alpha Shape 면이 너무 적음, Convex Hull로 대체")
                return cloud.convex_hull
                
            # Trimesh 객체 생성
            alpha_mesh = trimesh.Trimesh(vertices=points, faces=unique_faces)
            
            # 유효성 검사
            if not alpha_mesh.is_valid:
                alpha_mesh.fix_normals()
                
            self.log(f"Alpha Shape 완료 (alpha={alpha:.3f})")
            return alpha_mesh
            
        except Exception as e:
            self.log(f"Alpha Shape 실패: {e}, Convex Hull로 대체")
            return trimesh.points.PointCloud(points).convex_hull
    
    def create_poisson_mesh(self, points):
        """Poisson Surface Reconstruction (Open3D 사용)"""
        try:
            if not OPEN3D_AVAILABLE:
                self.log("Open3D가 없어 Convex Hull로 대체")
                return trimesh.points.PointCloud(points).convex_hull
                
            self.log("Poisson Reconstruction 시작...")
            
            # Open3D 포인트 클라우드 생성
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # 법선 벡터 추정
            pcd.estimate_normals()
            
            # Poisson reconstruction
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=9, width=0, scale=1.1, linear_fit=False
            )
            
            # Open3D 메시를 trimesh로 변환
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            
            poisson_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            self.log("Poisson Reconstruction 완료")
            return poisson_mesh
            
        except Exception as e:
            self.log(f"Poisson Reconstruction 실패: {e}, Convex Hull로 대체")
            return trimesh.points.PointCloud(points).convex_hull
    
    def create_ball_pivoting_mesh(self, points):
        """Ball Pivoting Algorithm (Open3D 사용)"""
        try:
            if not OPEN3D_AVAILABLE:
                self.log("Open3D가 없어 Convex Hull로 대체")
                return trimesh.points.PointCloud(points).convex_hull
                
            self.log("Ball Pivoting Algorithm 시작...")
            
            # Open3D 포인트 클라우드 생성
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # 법선 벡터 추정
            pcd.estimate_normals()
            
            # 포인트 간 평균 거리 계산
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            
            # Ball pivoting
            radii = [avg_dist, avg_dist * 2, avg_dist * 4]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii)
            )
            
            # Open3D 메시를 trimesh로 변환
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            
            if len(faces) == 0:
                self.log("Ball Pivoting 실패 (면 생성되지 않음), Convex Hull로 대체")
                return trimesh.points.PointCloud(points).convex_hull
                
            ball_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            self.log("Ball Pivoting Algorithm 완료")
            return ball_mesh
            
        except Exception as e:
            self.log(f"Ball Pivoting 실패: {e}, Convex Hull로 대체")
            return trimesh.points.PointCloud(points).convex_hull
    
    def create_delaunay_mesh(self, points):
        """3D Delaunay Triangulation을 이용한 메시 생성"""
        try:
            if not SCIPY_AVAILABLE:
                self.log("SciPy가 없어 Convex Hull로 대체")
                return trimesh.points.PointCloud(points).convex_hull
                
            self.log("Delaunay 3D Triangulation 시작...")
            
            # 포인트 수가 너무 많으면 샘플링
            if len(points) > 5000:
                self.log(f"포인트가 많아 샘플링: {len(points)} → 5000")
                indices = np.random.choice(len(points), 5000, replace=False)
                points = points[indices]
            
            # 3D Delaunay triangulation
            tri = Delaunay(points)
            
            # 표면 면만 추출 (convex hull 기반)
            hull = ConvexHull(points)
            surface_faces = []
            
            # 각 tetrahedron의 면 중 hull에 속하는 면만 선택
            for simplex in tri.simplices:
                faces = [
                    [simplex[0], simplex[1], simplex[2]],
                    [simplex[0], simplex[1], simplex[3]],
                    [simplex[0], simplex[2], simplex[3]],
                    [simplex[1], simplex[2], simplex[3]]
                ]
                
                for face in faces:
                    # 면이 convex hull의 일부인지 확인
                    if self.is_face_on_hull(face, hull.simplices):
                        surface_faces.append(face)
            
            if not surface_faces:
                self.log("Delaunay 표면 추출 실패, Convex Hull로 대체")
                return trimesh.points.PointCloud(points).convex_hull
            
            # 중복 제거
            unique_faces = []
            face_set = set()
            for face in surface_faces:
                face_tuple = tuple(sorted(face))
                if face_tuple not in face_set:
                    face_set.add(face_tuple)
                    unique_faces.append(face)
            
            delaunay_mesh = trimesh.Trimesh(vertices=points, faces=unique_faces)
            
            self.log("Delaunay 3D Triangulation 완료")
            return delaunay_mesh
            
        except Exception as e:
            self.log(f"Delaunay 3D 실패: {e}, Convex Hull로 대체")
            return trimesh.points.PointCloud(points).convex_hull
    
    def compute_circumradius_3d(self, tetra_points):
        """사면체의 circumradius 계산"""
        try:
            # 4개 점으로 이루어진 사면체의 외접구 반지름
            p0, p1, p2, p3 = tetra_points
            
            # 벡터 계산
            a = np.linalg.norm(p1 - p0)
            b = np.linalg.norm(p2 - p0)
            c = np.linalg.norm(p3 - p0)
            d = np.linalg.norm(p2 - p1)
            e = np.linalg.norm(p3 - p1)
            f = np.linalg.norm(p3 - p2)
            
            # 사면체 부피 계산
            volume = abs(np.dot(p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6.0
            
            if volume < 1e-10:  # 거의 평면인 경우
                return float('inf')
            
            # Circumradius 공식
            numerator = np.sqrt((a*d + b*e + c*f) * (a*d + b*e - c*f) * 
                               (a*d - b*e + c*f) * (-a*d + b*e + c*f))
            
            if numerator < 1e-10:
                return float('inf')
                
            circumradius = numerator / (24.0 * volume)
            return circumradius
            
        except:
            return float('inf')
    
    def is_face_on_hull(self, face, hull_simplices):
        """면이 convex hull의 일부인지 확인"""
        face_set = set(face)
        for simplex in hull_simplices:
            if face_set.issubset(set(simplex)):
                return True
        return False
    
    def run(self):
        """GUI 실행"""
        self.root.mainloop()

if __name__ == "__main__":
    app = CollisionProcessorGUI()
    app.run()