# URDF Visualizer (PyBullet Version)

This package provides a simple URDF visualizer using PyBullet without ROS.

## Prerequisites

- Python 3.8 or newer
- Required Python packages:
  - pybullet
  - numpy

## Installation

1. Install required Python packages:
   ```bash
   pip install pybullet numpy
   ```

## Usage

1. Run the URDF visualizer:
   ```bash
   python urdf_visualizer_pybullet.py
   ```

2. The visualization window will open showing:
   - A blue base link (0.5 x 0.5 x 0.1 meters)
   - A red arm link (0.1 x 0.1 x 0.5 meters)
   - The arm will automatically move back and forth

3. Controls:
   - Use mouse to rotate the view
   - Use mouse wheel to zoom in/out
   - Press Ctrl+C to exit

## Robot Description

The robot consists of:
- A blue base link (0.5 x 0.5 x 0.1 meters)
- A red arm link (0.1 x 0.1 x 0.5 meters)
- A revolute joint connecting the base and arm 