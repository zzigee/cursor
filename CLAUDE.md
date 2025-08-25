# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyBullet-based robot path planning simulator for industrial robotics. The project implements RRT* (Rapidly-exploring Random Tree star) algorithm for collision-free path planning with a 6-axis RB10-1300E industrial robot model and 3D pipe obstacles.

## Core Architecture

### Main Components

**Primary Simulator (`urdf_visualizer_pybullet.py`)**
- Complete robot path planning simulation with GUI controls
- Integrates 3D mesh processing, physics simulation, and path planning
- ~870 lines with modular function-based architecture

**Simple Visualizer (`urdf_visualizer_pybullet.py_로봇생성.py`)**
- Basic robot loading and visualization
- Minimal implementation for testing robot models

### Key Architectural Patterns

1. **3D Mesh Processing Pipeline**
   - PLY → OBJ conversion using trimesh
   - V-HACD convex decomposition for accurate collision detection
   - Automatic mesh scaling (mm to m units)

2. **Physics Simulation Layer**
   - PyBullet physics engine integration
   - Real-time collision detection between robot links and obstacles
   - Joint motor control with position/velocity constraints

3. **Path Planning Module**
   - RRT* algorithm implementation with collision checking
   - Inverse kinematics for Cartesian to joint space conversion
   - Path visualization and trajectory export

4. **GUI Control System**
   - PyBullet debug GUI with sliders for:
     - Start/end positions and orientations (6DOF)
     - Pipe placement and orientation
     - Camera controls and simulation speed
   - Real-time parameter adjustment during simulation

## Development Commands

### Installation
```bash
pip install pybullet numpy trimesh matplotlib keyboard
```

### Running Simulations
```bash
# Main robot path planning simulator
python urdf_visualizer_pybullet.py

# Simple robot visualization (for testing)
python "urdf_visualizer_pybullet.py_로봇생성.py"
```

### Output Files Generated
- `joint_trajectory.csv` - Robot joint angles over time
- `pipe.obj_vhacd.obj` - Convex decomposed collision mesh
- `rb10_1300e_modified.urdf` - URDF with absolute mesh paths
- `log.txt` - V-HACD processing log

## Key Functions and Data Flow

### Critical Functions
- `rrt_star_plan()` - Main path planning algorithm (line 211)
- `check_collision()` - Collision detection with detailed link reporting (line 78)
- `perform_convex_decomposition()` - V-HACD mesh processing (line 30)
- `visualize_path()` - 3D path visualization (line 246)
- `modify_urdf_mesh_paths()` - URDF path correction (line 57)

### Simulation Flow
1. Initialize PyBullet physics engine
2. Load and process 3D meshes (PLY→OBJ→V-HACD)
3. Load robot URDF with corrected mesh paths
4. Set up GUI controls and debug visualization
5. Wait for user input (start/end positions, pipe placement)
6. Execute RRT* path planning with collision avoidance
7. Visualize and animate the planned path
8. Export trajectory data

## Robot Model Details

**RB10-1300E Industrial Robot**
- 6-axis articulated robot arm
- Mesh files in `meshes/rb10_1300e/` (visual DAE + collision STL)
- Reach: 1300mm, payload capacity considerations built into URDF
- Joint limits and dynamics defined in URDF

## Working with 3D Assets

### Supported Formats
- **Input**: PLY, OBJ files for obstacles/pipes
- **Robot**: URDF with DAE visual meshes and STL collision meshes
- **Processing**: Automatic V-HACD convex decomposition for physics

### Mesh Processing Notes
- All meshes assumed to be in mm units, auto-scaled to meters
- V-HACD parameters tuned for reasonable processing time vs accuracy
- Original meshes used for visualization, decomposed meshes for collision

## Simulation Controls

### GUI Parameters
- **Position Controls**: Start X/Y/Z, End X/Y/Z (meters)
- **Orientation Controls**: RPY angles for start/end poses (radians)
- **Pipe Controls**: Position and orientation of obstacle
- **Camera Controls**: Distance, yaw, pitch with reset functionality
- **Simulation**: Speed control and start/reset buttons

### Interactive Features
- Real-time collision visualization (red highlighting)
- Live camera adjustment during simulation
- Path visualization with green lines showing TCP trajectory
- Keyboard shortcuts (Space for camera reset)