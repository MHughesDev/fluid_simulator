![Fluid Simulator Demo](demo.gif)

# Real-Time Fluid Simulator

A real-time, interactive 2D fluid simulator built from scratch in Python. It solves the incompressible Navier-Stokes equations using a hybrid PIC/FLIP particle method on a staggered MAC grid, rendered live at 60 FPS with Pygame.

This project was built to explore the numerical methods behind modern fluid simulation — the same family of techniques used in visual effects pipelines and physics engines.

---

## How It Works

Fluid simulation is a fundamentally hard problem: at every frame, the velocity field must remain physically consistent (divergence-free), particles must be advected without numerical blow-up, and all of this must happen fast enough to be interactive.

This simulator solves that with a three-layer pipeline:

### 1. MAC Grid (Marker-and-Cell)
Velocity components are stored on staggered cell faces rather than at cell centers — horizontal velocity `u` on vertical faces, vertical velocity `v` on horizontal faces. This staggered layout is a standard technique that prevents pressure-velocity decoupling and produces stable, artifact-free results.

### 2. PIC/FLIP Hybrid Particle System
Particles carry the fluid's state (position and velocity) across frames. Each simulation step involves two transfers:

- **Particle-to-Grid (P2G):** Particle velocities are splatted onto the MAC grid using bilinear interpolation and weighted averaging — fully vectorized with NumPy's `add.at`.
- **Grid-to-Particle (G2P):** After the grid is solved, corrected velocities are interpolated back to each particle using a tunable blend of PIC (stable, damping) and FLIP (energy-preserving, detailed). The simulator runs at 90% FLIP / 10% PIC for lively, responsive fluid behavior.

### 3. Pressure Projection
Incompressibility (∇·u = 0) is enforced every frame via a Jacobi iterative pressure solve:

1. Compute velocity divergence across the grid
2. Solve the Poisson equation (∇²p = div u) over 50 iterations
3. Subtract the pressure gradient from the velocity field to make it divergence-free

Air cells are held at zero pressure to simulate a free surface, giving the fluid realistic open-boundary behavior.

---

## Simulation Pipeline (Per Frame)

```
P2G → Apply Gravity → Enforce Boundary Conditions → Pressure Projection → G2P → Advect Particles → Collision Clamp
```

---

## Controls

| Input | Action |
|-------|--------|
| Left Click (hold) | Pour water at cursor |
| Right Click (hold) | Remove water at cursor |
| Arrow Up / Down | Increase / decrease pour rate |
| X | Reset simulation |

---

## Technical Specs

| Parameter | Value |
|-----------|-------|
| Window | 800 × 600 px |
| Cell Size | 12 px |
| Grid | 66 × 50 cells |
| Pressure Iterations | 50 (Jacobi) |
| PIC/FLIP Ratio | 10% PIC / 90% FLIP |
| Target FPS | 60 |
| CFL Safety Cap | 0.9 × cell size per step |

---

## Stack

- **Python** — simulation logic
- **NumPy** — vectorized grid operations (P2G/G2P, pressure solve, advection)
- **Pygame** — real-time rendering and input handling

---

## Getting Started

```bat
.\run.bat
```

This installs dependencies and launches the simulator. Requires Python 3.8+.

```bat
pip install -r requirements.txt
python fluid_sim.py
```
