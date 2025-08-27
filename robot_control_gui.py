#!/usr/bin/env python3
"""
Robot Control GUI - 별도 창으로 분리된 로봇 제어 인터페이스
PyBullet 시뮬레이터와 독립적으로 동작하는 GUI 컨트롤 패널
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import math
from dataclasses import dataclass
from typing import Dict, Any, Callable

@dataclass
class RobotConfig:
    """로봇 설정 데이터 클래스"""
    # Base position and orientation
    base_x: float = 0.0
    base_y: float = 0.0
    base_z: float = 0.0
    base_roll: float = 0.0
    base_pitch: float = 0.0
    base_yaw: float = math.pi  # 180도
    
    # Start position and orientation
    start_x: float = 0.0
    start_y: float = 0.0
    start_z: float = 0.0
    start_roll: float = 0.0
    start_pitch: float = 0.0
    start_yaw: float = 0.0
    
    # End position and orientation
    end_x: float = 0.0
    end_y: float = 0.2
    end_z: float = -0.3
    end_roll: float = 0.0
    end_pitch: float = 0.0
    end_yaw: float = 0.0

@dataclass 
class PipeConfig:
    """파이프 설정 데이터 클래스"""
    x: float = 0.0
    y: float = 1.65
    z: float = 1.10
    roll: float = 1.65
    pitch: float = 0.13
    yaw: float = 0.13

@dataclass
class SimulationConfig:
    """시뮬레이션 설정 데이터 클래스"""
    speed: float = 1.0

class RobotControlGUI:
    """별도 창 로봇 제어 GUI 클래스"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Robot Control Panel")
        self.root.geometry("800x900")
        self.root.resizable(True, True)
        
        # 데이터 저장
        self.robot1_config = RobotConfig()
        self.robot2_config = RobotConfig()
        self.robot2_config.base_x = 1.0  # Robot2 기본 위치
        self.robot2_config.start_x = 0.5
        self.robot2_config.end_x = 0.5
        
        self.pipe_config = PipeConfig()
        self.sim_config = SimulationConfig()
        
        # 콜백 함수들 (시뮬레이터와 연결용)
        self.callbacks: Dict[str, Callable] = {}
        
        # GUI 컴포넌트들
        self.widgets = {}
        
        # GUI 설정
        self.setup_gui()
        
        # 업데이트 스레드 실행 여부
        self.running = True
        
    def register_callback(self, event_name: str, callback: Callable):
        """콜백 함수 등록"""
        self.callbacks[event_name] = callback
        
    def setup_gui(self):
        """GUI 레이아웃 설정"""
        # 스크롤 가능한 메인 프레임
        main_canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=main_canvas.yview)
        scrollable_frame = ttk.Frame(main_canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )
        
        main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Robot 1 설정
        self.setup_robot_controls(scrollable_frame, "Robot 1", self.robot1_config, 0)
        
        # Robot 2 설정
        self.setup_robot_controls(scrollable_frame, "Robot 2", self.robot2_config, 1)
        
        # 파이프 설정
        self.setup_pipe_controls(scrollable_frame)
        
        # 시뮬레이션 제어
        self.setup_simulation_controls(scrollable_frame)
        
        # 패킹
        main_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 마우스 휠 스크롤 바인딩
        def _on_mousewheel(event):
            main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        main_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
    def setup_robot_controls(self, parent, robot_name: str, config: RobotConfig, robot_idx: int):
        """로봇 제어 패널 설정"""
        # 메인 프레임
        main_frame = ttk.LabelFrame(parent, text=f"{robot_name} Controls", padding=10)
        main_frame.pack(fill="x", padx=10, pady=5)
        
        # Base Position & Orientation
        base_frame = ttk.LabelFrame(main_frame, text="Base Position & Orientation", padding=5)
        base_frame.pack(fill="x", pady=5)
        
        base_pos_frame = ttk.Frame(base_frame)
        base_pos_frame.pack(fill="x", pady=2)
        
        # Base Position
        ttk.Label(base_pos_frame, text="Position (m):").grid(row=0, column=0, sticky="w", padx=5)
        
        base_x_var = tk.DoubleVar(value=config.base_x)
        base_y_var = tk.DoubleVar(value=config.base_y)
        base_z_var = tk.DoubleVar(value=config.base_z)
        
        ttk.Label(base_pos_frame, text="X:").grid(row=0, column=1, padx=5)
        base_x_entry = ttk.Entry(base_pos_frame, textvariable=base_x_var, width=8)
        base_x_entry.grid(row=0, column=2, padx=2)
        
        ttk.Label(base_pos_frame, text="Y:").grid(row=0, column=3, padx=5)
        base_y_entry = ttk.Entry(base_pos_frame, textvariable=base_y_var, width=8)
        base_y_entry.grid(row=0, column=4, padx=2)
        
        ttk.Label(base_pos_frame, text="Z:").grid(row=0, column=5, padx=5)
        base_z_entry = ttk.Entry(base_pos_frame, textvariable=base_z_var, width=8)
        base_z_entry.grid(row=0, column=6, padx=2)
        
        # Base Orientation
        base_rot_frame = ttk.Frame(base_frame)
        base_rot_frame.pack(fill="x", pady=2)
        
        ttk.Label(base_rot_frame, text="Orientation (rad):").grid(row=0, column=0, sticky="w", padx=5)
        
        base_roll_var = tk.DoubleVar(value=config.base_roll)
        base_pitch_var = tk.DoubleVar(value=config.base_pitch)
        base_yaw_var = tk.DoubleVar(value=config.base_yaw)
        
        ttk.Label(base_rot_frame, text="Roll:").grid(row=0, column=1, padx=5)
        base_roll_entry = ttk.Entry(base_rot_frame, textvariable=base_roll_var, width=8)
        base_roll_entry.grid(row=0, column=2, padx=2)
        
        ttk.Label(base_rot_frame, text="Pitch:").grid(row=0, column=3, padx=5)
        base_pitch_entry = ttk.Entry(base_rot_frame, textvariable=base_pitch_var, width=8)
        base_pitch_entry.grid(row=0, column=4, padx=2)
        
        ttk.Label(base_rot_frame, text="Yaw:").grid(row=0, column=5, padx=5)
        base_yaw_entry = ttk.Entry(base_rot_frame, textvariable=base_yaw_var, width=8)
        base_yaw_entry.grid(row=0, column=6, padx=2)
        
        # Apply Base 버튼
        apply_base_btn = ttk.Button(
            base_frame, 
            text=f"Apply {robot_name} Base",
            command=lambda: self._on_apply_base(robot_idx)
        )
        apply_base_btn.pack(pady=5)
        
        # Start Position & Orientation
        start_frame = ttk.LabelFrame(main_frame, text="Start Position & Orientation", padding=5)
        start_frame.pack(fill="x", pady=5)
        
        # Start Position 컨트롤들 (간단화)
        start_pos_frame = ttk.Frame(start_frame)
        start_pos_frame.pack(fill="x", pady=2)
        ttk.Label(start_pos_frame, text="Position (X, Y, Z):").pack(anchor="w")
        
        # End Position & Orientation  
        end_frame = ttk.LabelFrame(main_frame, text="End Position & Orientation", padding=5)
        end_frame.pack(fill="x", pady=5)
        
        # End Position 컨트롤들 (간단화)
        end_pos_frame = ttk.Frame(end_frame)
        end_pos_frame.pack(fill="x", pady=2)
        ttk.Label(end_pos_frame, text="Position (X, Y, Z):").pack(anchor="w")
        
        # 로봇 제어 버튼들
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill="x", pady=5)
        
        start_btn = ttk.Button(
            control_frame,
            text=f"Start {robot_name}",
            command=lambda: self._on_robot_start(robot_idx)
        )
        start_btn.pack(side="left", padx=5)
        
        # 위젯들 저장 (나중에 값을 읽기 위해)
        robot_key = f"robot{robot_idx + 1}"
        self.widgets[f"{robot_key}_base_x"] = base_x_var
        self.widgets[f"{robot_key}_base_y"] = base_y_var
        self.widgets[f"{robot_key}_base_z"] = base_z_var
        self.widgets[f"{robot_key}_base_roll"] = base_roll_var
        self.widgets[f"{robot_key}_base_pitch"] = base_pitch_var
        self.widgets[f"{robot_key}_base_yaw"] = base_yaw_var
        
    def _create_position_controls(self, parent, prefix: str, config: RobotConfig, attr_prefix: str):
        """위치/자세 컨트롤 생성"""
        # Position
        pos_frame = ttk.Frame(parent)
        pos_frame.pack(fill="x", pady=2)
        
        ttk.Label(pos_frame, text="Position (m):").grid(row=0, column=0, sticky="w", padx=5)
        
        x_var = tk.DoubleVar(value=getattr(config, f"{attr_prefix}_x"))
        y_var = tk.DoubleVar(value=getattr(config, f"{attr_prefix}_y"))
        z_var = tk.DoubleVar(value=getattr(config, f"{attr_prefix}_z"))
        
        ttk.Label(pos_frame, text="X:").grid(row=0, column=1, padx=5)
        x_entry = ttk.Entry(pos_frame, textvariable=x_var, width=8)
        x_entry.grid(row=0, column=2, padx=2)
        
        ttk.Label(pos_frame, text="Y:").grid(row=0, column=3, padx=5)
        y_entry = ttk.Entry(pos_frame, textvariable=y_var, width=8)
        y_entry.grid(row=0, column=4, padx=2)
        
        ttk.Label(pos_frame, text="Z:").grid(row=0, column=5, padx=5)
        z_entry = ttk.Entry(pos_frame, textvariable=z_var, width=8)
        z_entry.grid(row=0, column=6, padx=2)
        
        # Orientation
        rot_frame = ttk.Frame(parent)
        rot_frame.pack(fill="x", pady=2)
        
        ttk.Label(rot_frame, text="Orientation (rad):").grid(row=0, column=0, sticky="w", padx=5)
        
        roll_var = tk.DoubleVar(value=getattr(config, f"{attr_prefix}_roll"))
        pitch_var = tk.DoubleVar(value=getattr(config, f"{attr_prefix}_pitch"))
        yaw_var = tk.DoubleVar(value=getattr(config, f"{attr_prefix}_yaw"))
        
        ttk.Label(rot_frame, text="Roll:").grid(row=0, column=1, padx=5)
        roll_entry = ttk.Entry(rot_frame, textvariable=roll_var, width=8)
        roll_entry.grid(row=0, column=2, padx=2)
        
        ttk.Label(rot_frame, text="Pitch:").grid(row=0, column=3, padx=5)
        pitch_entry = ttk.Entry(rot_frame, textvariable=pitch_var, width=8)
        pitch_entry.grid(row=0, column=4, padx=2)
        
        ttk.Label(rot_frame, text="Yaw:").grid(row=0, column=5, padx=5)
        yaw_entry = ttk.Entry(rot_frame, textvariable=yaw_var, width=8)
        yaw_entry.grid(row=0, column=6, padx=2)
        
    def setup_pipe_controls(self, parent):
        """파이프 제어 패널 설정"""
        pipe_frame = ttk.LabelFrame(parent, text="Pipe Controls", padding=10)
        pipe_frame.pack(fill="x", padx=10, pady=5)
        
        # Position
        pos_frame = ttk.Frame(pipe_frame)
        pos_frame.pack(fill="x", pady=2)
        
        ttk.Label(pos_frame, text="Position (m):").grid(row=0, column=0, sticky="w", padx=5)
        
        pipe_x_var = tk.DoubleVar(value=self.pipe_config.x)
        pipe_y_var = tk.DoubleVar(value=self.pipe_config.y)
        pipe_z_var = tk.DoubleVar(value=self.pipe_config.z)
        
        ttk.Label(pos_frame, text="X:").grid(row=0, column=1, padx=5)
        ttk.Entry(pos_frame, textvariable=pipe_x_var, width=8).grid(row=0, column=2, padx=2)
        
        ttk.Label(pos_frame, text="Y:").grid(row=0, column=3, padx=5)
        ttk.Entry(pos_frame, textvariable=pipe_y_var, width=8).grid(row=0, column=4, padx=2)
        
        ttk.Label(pos_frame, text="Z:").grid(row=0, column=5, padx=5)
        ttk.Entry(pos_frame, textvariable=pipe_z_var, width=8).grid(row=0, column=6, padx=2)
        
        # Orientation
        rot_frame = ttk.Frame(pipe_frame)
        rot_frame.pack(fill="x", pady=2)
        
        ttk.Label(rot_frame, text="Orientation (rad):").grid(row=0, column=0, sticky="w", padx=5)
        
        pipe_roll_var = tk.DoubleVar(value=self.pipe_config.roll)
        pipe_pitch_var = tk.DoubleVar(value=self.pipe_config.pitch)
        pipe_yaw_var = tk.DoubleVar(value=self.pipe_config.yaw)
        
        ttk.Label(rot_frame, text="Roll:").grid(row=0, column=1, padx=5)
        ttk.Entry(rot_frame, textvariable=pipe_roll_var, width=8).grid(row=0, column=2, padx=2)
        
        ttk.Label(rot_frame, text="Pitch:").grid(row=0, column=3, padx=5)
        ttk.Entry(rot_frame, textvariable=pipe_pitch_var, width=8).grid(row=0, column=4, padx=2)
        
        ttk.Label(rot_frame, text="Yaw:").grid(row=0, column=5, padx=5)
        ttk.Entry(rot_frame, textvariable=pipe_yaw_var, width=8).grid(row=0, column=6, padx=2)
        
        # 위젯 저장
        self.widgets["pipe_x"] = pipe_x_var
        self.widgets["pipe_y"] = pipe_y_var
        self.widgets["pipe_z"] = pipe_z_var
        self.widgets["pipe_roll"] = pipe_roll_var
        self.widgets["pipe_pitch"] = pipe_pitch_var
        self.widgets["pipe_yaw"] = pipe_yaw_var
        
    def setup_simulation_controls(self, parent):
        """시뮬레이션 제어 패널 설정"""
        sim_frame = ttk.LabelFrame(parent, text="Simulation Controls", padding=10)
        sim_frame.pack(fill="x", padx=10, pady=5)
        
        # Speed control
        speed_frame = ttk.Frame(sim_frame)
        speed_frame.pack(fill="x", pady=5)
        
        ttk.Label(speed_frame, text="Simulation Speed:").pack(side="left", padx=5)
        
        speed_var = tk.DoubleVar(value=self.sim_config.speed)
        speed_scale = ttk.Scale(
            speed_frame, 
            from_=0.1, 
            to=2.0, 
            variable=speed_var, 
            orient="horizontal",
            length=200
        )
        speed_scale.pack(side="left", padx=5)
        
        speed_label = ttk.Label(speed_frame, text=f"{speed_var.get():.1f}")
        speed_label.pack(side="left", padx=5)
        
        def update_speed_label(*args):
            speed_label.config(text=f"{speed_var.get():.1f}")
            
        speed_var.trace_add("write", update_speed_label)
        
        # Control buttons
        btn_frame = ttk.Frame(sim_frame)
        btn_frame.pack(fill="x", pady=10)
        
        ttk.Button(
            btn_frame,
            text="Reset All",
            command=self._on_reset_all
        ).pack(side="left", padx=5)
        
        ttk.Button(
            btn_frame,
            text="Emergency Stop",
            command=self._on_emergency_stop
        ).pack(side="left", padx=5)
        
        # Status display
        status_frame = ttk.LabelFrame(sim_frame, text="Status", padding=5)
        status_frame.pack(fill="x", pady=5)
        
        self.status_text = tk.Text(status_frame, height=4, wrap="word")
        status_scroll = ttk.Scrollbar(status_frame, command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=status_scroll.set)
        
        self.status_text.pack(side="left", fill="both", expand=True)
        status_scroll.pack(side="right", fill="y")
        
        # 위젯 저장
        self.widgets["speed"] = speed_var
        
    def _on_apply_base(self, robot_idx: int):
        """베이스 적용 버튼 클릭"""
        try:
            robot_key = f"robot{robot_idx + 1}"
            base_config = {
                "x": self.widgets[f"{robot_key}_base_x"].get(),
                "y": self.widgets[f"{robot_key}_base_y"].get(),
                "z": self.widgets[f"{robot_key}_base_z"].get(),
                "roll": self.widgets[f"{robot_key}_base_roll"].get(),
                "pitch": self.widgets[f"{robot_key}_base_pitch"].get(),
                "yaw": self.widgets[f"{robot_key}_base_yaw"].get()
            }
            
            if "apply_robot_base" in self.callbacks:
                self.callbacks["apply_robot_base"](robot_idx, base_config)
                self._log_status(f"Robot {robot_idx + 1} base position applied successfully")
            else:
                self._log_status("Warning: No callback registered for apply_robot_base")
                
        except Exception as e:
            self._log_status(f"Error applying Robot {robot_idx + 1} base: {str(e)}")
            messagebox.showerror("Error", f"Failed to apply robot base: {str(e)}")
            
    def _on_robot_start(self, robot_idx: int):
        """로봇 시작 버튼 클릭"""
        try:
            if "robot_start" in self.callbacks:
                self.callbacks["robot_start"](robot_idx)
                self._log_status(f"Robot {robot_idx + 1} path planning started")
            else:
                self._log_status("Warning: No callback registered for robot_start")
                
        except Exception as e:
            self._log_status(f"Error starting Robot {robot_idx + 1}: {str(e)}")
            messagebox.showerror("Error", f"Failed to start robot: {str(e)}")
            
    def _on_reset_all(self):
        """전체 리셋 버튼 클릭"""
        try:
            if "reset_all" in self.callbacks:
                self.callbacks["reset_all"]()
                self._log_status("Simulation reset completed")
            else:
                self._log_status("Warning: No callback registered for reset_all")
                
        except Exception as e:
            self._log_status(f"Error resetting simulation: {str(e)}")
            messagebox.showerror("Error", f"Failed to reset simulation: {str(e)}")
            
    def _on_emergency_stop(self):
        """비상 정지 버튼 클릭"""
        try:
            if "emergency_stop" in self.callbacks:
                self.callbacks["emergency_stop"]()
                self._log_status("Emergency stop activated")
            else:
                self._log_status("Warning: No callback registered for emergency_stop")
                
        except Exception as e:
            self._log_status(f"Error during emergency stop: {str(e)}")
            
    def _log_status(self, message: str):
        """상태 로그 추가"""
        timestamp = time.strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.see(tk.END)
        
    def get_config_values(self) -> Dict[str, Any]:
        """현재 설정 값들 반환"""
        values = {}
        for key, widget in self.widgets.items():
            if hasattr(widget, 'get'):
                values[key] = widget.get()
        return values
        
    def update_status(self, message: str):
        """외부에서 상태 업데이트"""
        self.root.after(0, lambda: self._log_status(message))
        
    def run(self):
        """GUI 실행"""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
            self.root.mainloop()
        except Exception as e:
            print(f"GUI Error: {e}")
            
    def _on_closing(self):
        """창 닫기 이벤트"""
        self.running = False
        if "gui_closing" in self.callbacks:
            self.callbacks["gui_closing"]()
        self.root.destroy()

if __name__ == "__main__":
    # 테스트용 실행
    gui = RobotControlGUI()
    
    def test_callback(name):
        def callback(*args):
            print(f"Callback triggered: {name} with args: {args}")
        return callback
    
    # 테스트 콜백 등록
    gui.register_callback("apply_robot_base", test_callback("apply_robot_base"))
    gui.register_callback("robot_start", test_callback("robot_start"))
    gui.register_callback("reset_all", test_callback("reset_all"))
    
    gui.run()