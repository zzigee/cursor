#!/usr/bin/env python3
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os

class PointCloudViewer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("파이프 포인트 클라우드 뷰어")
        self.root.geometry("800x600")
        
        self.current_points = None
        self.current_filename = ""
        
        self.setup_ui()
        
    def setup_ui(self):
        # 파일 선택 프레임
        file_frame = ttk.Frame(self.root)
        file_frame.pack(pady=10, padx=10, fill="x")
        
        ttk.Label(file_frame, text="PLY 파일 선택:").pack(side="left")
        
        self.file_var = tk.StringVar()
        file_combo = ttk.Combobox(file_frame, textvariable=self.file_var, width=30)
        file_combo['values'] = self.get_ply_files()
        file_combo.pack(side="left", padx=10)
        
        load_btn = ttk.Button(file_frame, text="로드", command=self.load_file)
        load_btn.pack(side="left", padx=5)
        
        view_btn = ttk.Button(file_frame, text="3D 뷰", command=self.show_3d_view)
        view_btn.pack(side="left", padx=5)
        
        # 정보 표시 프레임
        info_frame = ttk.LabelFrame(self.root, text="파일 정보")
        info_frame.pack(pady=10, padx=10, fill="both", expand=True)
        
        # 스크롤 가능한 텍스트
        text_frame = ttk.Frame(info_frame)
        text_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.info_text = tk.Text(text_frame, wrap="word", font=("Consolas", 10))
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=scrollbar.set)
        
        self.info_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 옵션 프레임
        option_frame = ttk.LabelFrame(self.root, text="표시 옵션")
        option_frame.pack(pady=10, padx=10, fill="x")
        
        # 다운샘플링 옵션
        ttk.Label(option_frame, text="다운샘플링:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.sample_var = tk.StringVar(value="50000")
        sample_combo = ttk.Combobox(option_frame, textvariable=self.sample_var, width=10)
        sample_combo['values'] = ["10000", "25000", "50000", "100000", "모든 포인트"]
        sample_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # 색상 옵션
        ttk.Label(option_frame, text="색상 기준:").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.color_var = tk.StringVar(value="Z 좌표")
        color_combo = ttk.Combobox(option_frame, textvariable=self.color_var, width=10)
        color_combo['values'] = ["Z 좌표", "Y 좌표", "X 좌표", "거리", "단일색"]
        color_combo.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        
    def get_ply_files(self):
        """현재 디렉토리의 PLY 파일 목록"""
        files = []
        for f in os.listdir('.'):
            if f.endswith('.ply'):
                files.append(f)
        return files
        
    def load_file(self):
        """선택된 PLY 파일 로드"""
        filename = self.file_var.get()
        if not filename:
            messagebox.showwarning("경고", "파일을 선택해주세요.")
            return
            
        if not os.path.exists(filename):
            messagebox.showerror("오류", f"파일을 찾을 수 없습니다: {filename}")
            return
            
        try:
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, f"로딩 중: {filename}...\n")
            self.root.update()
            
            # PLY 파일 로드
            mesh = trimesh.load(filename)
            
            if isinstance(mesh, trimesh.points.PointCloud):
                self.current_points = mesh.vertices
                data_type = "포인트 클라우드"
                has_faces = False
            elif hasattr(mesh, 'vertices'):
                self.current_points = mesh.vertices
                data_type = "메시"
                has_faces = hasattr(mesh, 'faces') and len(mesh.faces) > 0
            else:
                messagebox.showerror("오류", "포인트 데이터를 찾을 수 없습니다.")
                return
                
            self.current_filename = filename
            self.display_info(filename, data_type, has_faces, mesh)
            
        except Exception as e:
            messagebox.showerror("오류", f"파일 로드 실패:\n{str(e)}")
            
    def display_info(self, filename, data_type, has_faces, mesh):
        """파일 정보 표시"""
        self.info_text.delete(1.0, tk.END)
        
        points = self.current_points
        
        info = f"=== {filename} ===\n\n"
        info += f"타입: {data_type}\n"
        info += f"포인트 수: {len(points):,}개\n"
        
        if has_faces:
            info += f"면 수: {len(mesh.faces):,}개\n"
            
        # 파일 크기
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
        elif max_coord > 10:
            unit = "cm"
        else:
            unit = "m"
        info += f"추정 단위: {unit}\n\n"
        
        # 형상 분석
        dimensions = sorted(size, reverse=True)
        aspect_ratio = dimensions[0] / dimensions[1] if dimensions[1] > 0 else 0
        
        info += "=== 형상 분석 ===\n"
        info += f"최대 길이: {dimensions[0]:.6f}\n"
        info += f"중간 폭:   {dimensions[1]:.6f}\n"
        info += f"최소 두께: {dimensions[2]:.6f}\n"
        info += f"길이/폭 비율: {aspect_ratio:.1f}\n\n"
        
        # 샘플 포인트
        info += "=== 샘플 포인트 (10개) ===\n"
        sample_indices = np.linspace(0, len(points)-1, min(10, len(points)), dtype=int)
        for i, idx in enumerate(sample_indices):
            x, y, z = points[idx]
            info += f"[{i:2d}]: ({x:8.6f}, {y:8.6f}, {z:8.6f})\n"
            
        self.info_text.insert(tk.END, info)
        
    def show_3d_view(self):
        """3D 뷰어 표시"""
        if self.current_points is None:
            messagebox.showwarning("경고", "먼저 파일을 로드해주세요.")
            return
            
        # 별도 스레드에서 실행
        thread = threading.Thread(target=self.create_3d_plot)
        thread.daemon = True
        thread.start()
        
    def create_3d_plot(self):
        """3D 플롯 생성"""
        try:
            points = self.current_points.copy()
            
            # 다운샘플링
            sample_size = self.sample_var.get()
            if sample_size != "모든 포인트":
                max_points = int(sample_size)
                if len(points) > max_points:
                    indices = np.random.choice(len(points), max_points, replace=False)
                    points = points[indices]
                    
            # 3D 플롯 생성
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            
            # 색상 결정
            color_mode = self.color_var.get()
            if color_mode == "Z 좌표":
                colors = points[:, 2]
                colormap = 'viridis'
            elif color_mode == "Y 좌표":
                colors = points[:, 1]
                colormap = 'plasma'
            elif color_mode == "X 좌표":
                colors = points[:, 0]
                colormap = 'coolwarm'
            elif color_mode == "거리":
                center = points.mean(axis=0)
                colors = np.linalg.norm(points - center, axis=1)
                colormap = 'hot'
            else:  # 단일색
                colors = 'blue'
                colormap = None
                
            # 포인트 그리기
            if colormap:
                scatter = ax.scatter(
                    points[:, 0], points[:, 1], points[:, 2],
                    c=colors, cmap=colormap, s=1, alpha=0.6
                )
                plt.colorbar(scatter, shrink=0.8)
            else:
                ax.scatter(
                    points[:, 0], points[:, 1], points[:, 2],
                    c=colors, s=1, alpha=0.6
                )
            
            # 축 설정
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'{self.current_filename} - {len(points):,} 포인트')
            
            # 축 비율 맞추기
            max_range = np.array([
                points[:,0].max() - points[:,0].min(),
                points[:,1].max() - points[:,1].min(),
                points[:,2].max() - points[:,2].min()
            ]).max() / 2.0
            
            mid_x = (points[:,0].max() + points[:,0].min()) * 0.5
            mid_y = (points[:,1].max() + points[:,1].min()) * 0.5
            mid_z = (points[:,2].max() + points[:,2].min()) * 0.5
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            messagebox.showerror("오류", f"3D 뷰 생성 실패:\n{str(e)}")
            
    def run(self):
        """GUI 실행"""
        self.root.mainloop()

if __name__ == "__main__":
    viewer = PointCloudViewer()
    viewer.run()