# Fluid Simulator Specification

## Physical Model
- **Type**: 2D Incompressible Navier-Stokes Equations.
- **Method**: MAC (Marker-and-Cell) Grid solver with PIC/FLIP hybrid particles.
- **Incompressibility**: ∇·u = 0 enforced via Pressure Projection.

## Domain & Resolution
- **Window Size**: 800x600 pixels.
- **Grid Resolution**: 80x60 cells (cell size = 10 pixels).
- **Time Step (dt)**: Fixed (e.g., 1/60s) with potential CFL-based substepping.

## Performance Targets
- **Target FPS**: 60 FPS on modern CPU.
- **Pressure Solver**: Jacobi iterations capped at 40-100 to maintain real-time performance.
- **Vectorization**: Heavy use of Numpy for grid operations.

## Features
- Interactive water pouring (Left Click).
- Water removal (Right Click).
- Global reset (X Key).
- Stable stacking and sloshing.
