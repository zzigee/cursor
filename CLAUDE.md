# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyBullet-based robot path planning and collision detection simulator for industrial robotics. The project has evolved into a comprehensive robotics simulation platform with multiple specialized tools:

1. **RRT* Path Planning** - Core robot path planning with collision avoidance
2. **Point Cloud Processing** - Advanced 3D mesh and point cloud collision detection
3. **Multi-Robot Simulation** - Dual robot coordination and collision detection
4. **Interactive GUI Tools** - Specialized viewers for different collision detection approaches

## Core Architecture

### Main Applications

**Primary Simulator (`urdf_visualizer_pybullet.py`)**
- Complete robot path planning simulation with PyBullet GUI controls
- RRT* algorithm with collision-free path planning for RB10-1300E robot
- Multi-robot support with inter-robot collision detection
- ~900 lines with modular function-based architecture

**Collision Processing GUI (`collision_gui_viewer.py`)**
- Advanced mesh processing with multiple collision detection methods
- Integrates trimesh, Open3D, SciPy, and scikit-learn
- V-HACD convex decomposition with progress tracking
- Tkinter-based interface with real-time processing feedback

**Point Cloud Collision Viewer (`point_cloud_collision_viewer.py`)**
- Direct point cloud collision detection using KDTree
- No mesh conversion required - preserves 100% geometry fidelity
- High-performance collision checking for complex geometries
- Requires SciPy (mandatory), scikit-learn (optional)

**Simple GUI Viewers**
- `gui_pointcloud_viewer.py` - Basic PLY file visualization
- `simple_gui_viewer.py` - Lightweight point cloud viewer
- `show_pipe2_gui.py` - Pipe-specific visualization tools

### Key Architectural Patterns

1. **3D Mesh Processing Pipeline**
   - PLY → OBJ conversion using trimesh
   - V-HACD convex decomposition for accurate physics collision
   - Automatic mesh scaling (mm to m units) with caching system
   - Multiple processing backends (trimesh, Open3D, SciPy)

2. **Physics Simulation Layer**
   - PyBullet physics engine integration
   - Real-time collision detection between robot links and obstacles
   - Multi-robot collision detection with color-coded visualization
   - Joint motor control with position/velocity constraints

3. **Path Planning Module**
   - RRT* algorithm implementation with collision checking
   - Inverse kinematics for Cartesian to joint space conversion
   - Path visualization and CSV trajectory export
   - Multi-robot path coordination

4. **Collision Detection Systems**
   - **PyBullet Physics**: Full physics simulation with convex decomposition
   - **KDTree Point Cloud**: Direct point cloud collision without mesh conversion
   - **Hybrid Approach**: Combines mesh and point cloud methods
   - **Caching System**: MD5-based caching for processed meshes

5. **GUI Control Systems**
   - **PyBullet Debug GUI**: Sliders for 6DOF pose control, pipe placement, camera controls
   - **Tkinter Interfaces**: Advanced processing controls with progress tracking
   - **Real-time Visualization**: Live collision highlighting and camera manipulation

## Development Commands

### Core Dependencies
```bash
# Essential packages
pip install pybullet numpy trimesh matplotlib keyboard

# Advanced collision detection
pip install scipy scikit-learn

# Enhanced 3D processing (optional)
pip install open3d
```

### Running Applications
```bash
# Main robot path planning simulator
python urdf_visualizer_pybullet.py

# Advanced collision mesh processor
python collision_gui_viewer.py

# Direct point cloud collision detector
python point_cloud_collision_viewer.py

# Simple point cloud viewer
python gui_pointcloud_viewer.py

# Analysis and debugging tools
python analyze_pipe2.py
python compare_pipes.py
python debug_pipe_1.py
```

### Generated Output Files
- `joint_trajectory.csv` - Robot joint angles over time
- `robot1_trajectory.csv` / `robot2_trajectory.csv` - Multi-robot trajectories
- `*_vhacd.obj` - V-HACD convex decomposed collision meshes
- `*_processed.obj` - Processed mesh files
- `*.cache_info` - MD5 cache validation files
- `rb10_1300e_modified.urdf` - URDF with absolute mesh paths
- `vhacd_log.txt` - V-HACD processing logs

## Robot Models and Assets

### Supported Robot Models
- **RB10-1300E Industrial Robot** (Primary)
  - 6-axis articulated robot arm, 1300mm reach
  - Mesh files: `meshes/rb10_1300e/visual/*.dae` and `meshes/rb10_1300e/collision/*.stl`
  - URDF variants: `rb10_1300e_RT.urdf`, `rb10_1300e_DDA.urdf`

- **Generic Models** (For testing)
  - `robot.urdf` - Simple test robot
  - `gantry.urdf` - Gantry system

### 3D Asset Processing
- **Input Formats**: PLY, OBJ files for obstacles/pipes
- **Robot Formats**: URDF with DAE visual + STL collision meshes
- **Processing Pipeline**: PLY → OBJ → V-HACD convex decomposition
- **Unit Handling**: Automatic mm to m conversion with scaling detection
- **Caching**: MD5-based cache validation for expensive V-HACD operations

## Key Functions and Data Flow

### Critical Functions by Module

**Core Path Planning (`urdf_visualizer_pybullet.py`)**
- `rrt_star_plan()` - Main RRT* path planning algorithm
- `check_collision()` - Multi-robot collision detection with detailed link reporting
- `perform_convex_decomposition()` - V-HACD mesh processing with caching
- `modify_urdf_mesh_paths()` - URDF path correction for absolute paths
- `visualize_path()` - 3D path visualization and animation

**Mesh Processing (`ply_processor.py`)**
- `convert_ply_to_obj()` - PLY to OBJ conversion
- `perform_convex_decomposition()` - V-HACD wrapper with logging
- `load_pipe_from_ply()` - PyBullet PLY loader
- `load_ply_as_pybullet_body()` - Direct PLY to PyBullet body

**GUI Applications**
- `CollisionProcessorGUI` class - Advanced mesh processing interface
- `PointCloudCollisionViewer` class - Direct point cloud collision detection
- `PointCloudViewer` class - Basic PLY file visualization

### Simulation Flow
1. **Initialization**: PyBullet physics engine setup
2. **Asset Processing**: Load and process 3D meshes (PLY→OBJ→V-HACD) with caching
3. **Robot Loading**: Load robot URDF with corrected absolute mesh paths
4. **GUI Setup**: Initialize debug sliders and visualization controls
5. **User Input**: Configure start/end positions, pipe placement via GUI
6. **Path Planning**: Execute RRT* with multi-robot collision avoidance
7. **Visualization**: Animate planned paths with real-time collision highlighting
8. **Data Export**: Save trajectory data to CSV files

## Collision Detection Architecture

### Multi-Layered Collision Systems
1. **PyBullet Physics**: Full rigid body simulation with V-HACD convex hulls
2. **KDTree Point Cloud**: Direct distance queries without mesh conversion
3. **Hybrid Methods**: Combines both approaches for optimal performance/accuracy
4. **Color-Coded Visualization**: Red (robot-pipe), orange (robot-robot), green (safe paths)

### Performance Considerations
- **Caching**: Expensive V-HACD operations cached with MD5 validation
- **Optional Dependencies**: Graceful degradation when advanced libraries unavailable
- **Multi-threaded Processing**: GUI applications use threading for responsiveness
- **Memory Management**: Point cloud downsampling for large datasets

## Interactive Controls

### PyBullet Debug GUI
- **Robot 1 Controls**: Start/End positions (X/Y/Z), orientations (RPY), path planning triggers
- **Robot 2 Controls**: Independent multi-robot coordination controls  
- **Pipe Controls**: 6DOF pipe placement and orientation
- **Camera Controls**: Distance, yaw, pitch with reset functionality
- **Simulation Controls**: Speed adjustment, start/reset/stop buttons

### Tkinter GUI Features
- **File Management**: Load/save PLY/OBJ files with file browser
- **Processing Controls**: V-HACD parameter adjustment, progress tracking
- **Visualization**: Real-time 3D plots, mesh statistics, processing logs
- **Export Options**: Multiple mesh format outputs with quality settings

## Dependencies and Optional Features

### Required Dependencies
- `pybullet` - Physics simulation and robot control
- `numpy` - Numerical computations
- `trimesh` - 3D mesh processing
- `matplotlib` - 3D visualization and plotting

### Optional Dependencies (Graceful Degradation)
- `scipy` - Required for KDTree collision detection
- `scikit-learn` - Enhanced clustering and sampling
- `open3d` - Advanced 3D reconstruction methods
- `keyboard` - Keyboard shortcuts in simulators

### Feature Matrix
| Feature | PyBullet | SciPy | scikit-learn | Open3D |
|---------|----------|-------|--------------|--------|
| Basic simulation | ✓ | | | |
| RRT* planning | ✓ | | | |
| Point cloud collision | ✓ | ✓ | | |
| Advanced clustering | ✓ | ✓ | ✓ | |
| Mesh reconstruction | ✓ | ✓ | ✓ | ✓ |