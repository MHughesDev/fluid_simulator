# Fluid Simulator Architecture

## Data Layout: MAC Grid (Staggered)
Velocities are stored on cell faces to ensure stable pressure-velocity coupling.

- **u (Horizontal Velocity)**: Grid of size (NX+1, NY). Located at vertical faces.
- **v (Vertical Velocity)**: Grid of size (NX, NY+1). Located at horizontal faces.
- **p (Pressure)**: Grid of size (NX, NY). Located at cell centers.
- **type (Cell Mask)**: Solid, Fluid, or Air.

## Hybrid Particle System: PIC/FLIP
- **Particles**: Carry position and velocity.
- **Transfer**:
    - **Particle to Grid (P2G)**: Splat particle velocities onto the MAC grid faces.
    - **Grid to Particle (G2P)**: Interpolate grid velocities back to particles (PIC for stability, FLIP for detail).

## Simulation Pipeline
1. **P2G**: Transfer particle velocities to MAC grid.
2. **Apply Forces**: Add gravity to the `v` field.
3. **Boundary Conditions**: Set velocities at solid boundaries to zero.
4. **Pressure Projection**:
    - Calculate divergence (div u).
    - Solve Poisson equation (∇²p = div u) iteratively.
    - Correct velocities (u = u - grad p).
5. **G2P**: Transfer corrected velocities back to particles.
6. **Advection**: Move particles through the velocity field.
