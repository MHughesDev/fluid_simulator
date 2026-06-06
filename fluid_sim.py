import pygame
import numpy as np
import json
import time

def log_debug(hypothesis_id, message, data=None):
    log_entry = {
        "sessionId": "debug-session",
        "runId": "post-fix",
        "hypothesisId": hypothesis_id,
        "location": "fluid_sim.py",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000)
    }
    try:
        with open(r"c:\Users\Mason\Desktop\coding_projects\fluid_simulator\.cursor\debug.log", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except:
        pass

# Constants
WIDTH, HEIGHT = 800, 600
CELL_SIZE = 12
NX = WIDTH // CELL_SIZE
NY = HEIGHT // CELL_SIZE
GRAVITY = 0.15
DT = 0.5 # Reduced for stability
ITERATIONS = 50 
PIC_FLIP_RATIO = 0.1  # 0.1 PIC, 0.9 FLIP
MAX_VEL = CELL_SIZE * 0.9 # CFL safety cap


# Cell Types
AIR = 0
FLUID = 1
SOLID = 2

class FluidSimulator:
    def __init__(self):
        # MAC Grid
        self.u = np.zeros((NX + 1, NY), dtype=np.float32)
        self.v = np.zeros((NX, NY + 1), dtype=np.float32)
        self.u_prev = np.zeros_like(self.u)
        self.v_prev = np.zeros_like(self.v)
        self.p = np.zeros((NX, NY), dtype=np.float32)
        self.cell_type = np.full((NX, NY), AIR, dtype=np.int8)
        
        # Particles
        self.particle_pos = np.zeros((0, 2), dtype=np.float32)
        self.particle_vel = np.zeros((0, 2), dtype=np.float32)
        
        # Pre-calculate grid indices for P2G/G2P
        self.gx_indices = np.arange(NX + 1)
        self.gy_indices = np.arange(NY + 1)
        
        # UI
        self.water_amount = 20

    def add_water(self, mouse_pos):
        num = self.water_amount
        new_pos = mouse_pos + np.random.uniform(-20, 20, (num, 2))
        new_vel = np.zeros((num, 2), dtype=np.float32)
        
        self.particle_pos = np.vstack([self.particle_pos, new_pos])
        self.particle_vel = np.vstack([self.particle_vel, new_vel])

    def remove_water(self, mouse_pos, radius=50):
        if len(self.particle_pos) == 0: return
        dists = np.sum((self.particle_pos - mouse_pos)**2, axis=1)
        mask = dists > radius**2
        self.particle_pos = self.particle_pos[mask]
        self.particle_vel = self.particle_vel[mask]

    def clear(self):
        self.particle_pos = np.zeros((0, 2), dtype=np.float32)
        self.particle_vel = np.zeros((0, 2), dtype=np.float32)
        self.u.fill(0)
        self.v.fill(0)
        self.p.fill(0)

    def p2g(self):
        """Transfer particle velocities to grid using vectorized splatting."""
        # #region agent log
        log_debug("B", "p2g start", {"num_particles": len(self.particle_pos)})
        # #endregion
        self.u.fill(0)
        self.v.fill(0)
        u_weight = np.zeros_like(self.u)
        v_weight = np.zeros_like(self.v)
        
        px, py = self.particle_pos[:, 0] / CELL_SIZE, self.particle_pos[:, 1] / CELL_SIZE
        
        # Transfer to U grid
        gx, gy = px, py - 0.5
        ix, iy = gx.astype(int), gy.astype(int)
        fx, fy = gx - ix, gy - iy
        
        mask = (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY - 1)
        ix, iy, fx, fy = ix[mask], iy[mask], fx[mask], fy[mask]
        pv = self.particle_vel[mask, 0]
        
        # Vectorized atomic add-like operation using np.add.at
        np.add.at(self.u, (ix, iy), pv * (1-fx) * (1-fy))
        np.add.at(u_weight, (ix, iy), (1-fx) * (1-fy))
        np.add.at(self.u, (ix+1, iy), pv * fx * (1-fy))
        np.add.at(u_weight, (ix+1, iy), fx * (1-fy))
        np.add.at(self.u, (ix, iy+1), pv * (1-fx) * fy)
        np.add.at(u_weight, (ix, iy+1), (1-fx) * fy)
        np.add.at(self.u, (ix+1, iy+1), pv * fx * fy)
        np.add.at(u_weight, (ix+1, iy+1), fx * fy)

        # Transfer to V grid
        gx, gy = px - 0.5, py
        ix, iy = gx.astype(int), gy.astype(int)
        fx, fy = gx - ix, gy - iy
        
        mask = (ix >= 0) & (ix < NX - 1) & (iy >= 0) & (iy < NY)
        ix, iy, fx, fy = ix[mask], iy[mask], fx[mask], fy[mask]
        pv = self.particle_vel[mask, 1]
        
        np.add.at(self.v, (ix, iy), pv * (1-fx) * (1-fy))
        np.add.at(v_weight, (ix, iy), (1-fx) * (1-fy))
        np.add.at(self.v, (ix+1, iy), pv * fx * (1-fy))
        np.add.at(v_weight, (ix+1, iy), fx * (1-fy))
        np.add.at(self.v, (ix, iy+1), pv * (1-fx) * fy)
        np.add.at(v_weight, (ix, iy+1), (1-fx) * fy)
        np.add.at(self.v, (ix+1, iy+1), pv * fx * fy)
        np.add.at(v_weight, (ix+1, iy+1), fx * fy)

        u_mask = u_weight > 0
        self.u[u_mask] /= u_weight[u_mask]
        v_mask = v_weight > 0
        self.v[v_mask] /= v_weight[v_mask]
        
        # #region agent log
        if np.any(np.isnan(self.u)) or np.any(np.isnan(self.v)):
            log_debug("B", "p2g produced NaN", {"u_nans": int(np.isnan(self.u).sum()), "v_nans": int(np.isnan(self.v).sum())})
        # #endregion
        
        self.u_prev[:] = self.u
        self.v_prev[:] = self.v

    def solve_pressure(self):
        """Vectorized Jacobi Pressure Solver."""
        # 1. Update Fluid Mask
        self.cell_type.fill(AIR)
        px, py = (self.particle_pos[:, 0] / CELL_SIZE).astype(int), (self.particle_pos[:, 1] / CELL_SIZE).astype(int)
        mask = (px >= 0) & (px < NX) & (py >= 0) & (py < NY)
        self.cell_type[px[mask], py[mask]] = FLUID
        
        # 2. Divergence
        div = (self.u[1:, :] - self.u[:-1, :]) + (self.v[:, 1:] - self.v[:, :-1])
        
        # 3. Solve Poisson using Vectorized Jacobi
        self.p.fill(0)
        fluid_mask = (self.cell_type == FLUID)
        
        # Correct neighbor counting: count how many neighbors are within bounds and not SOLID
        # (In our case, everything outside the grid is treated as SOLID)
        neighbor_count = np.zeros((NX, NY), dtype=np.float32)
        
        # Use shifting to count neighbors
        p_count = np.zeros((NX, NY), dtype=np.float32)
        p_count[1:, :] += 1 # Has left neighbor
        p_count[:-1, :] += 1 # Has right neighbor
        p_count[:, 1:] += 1 # Has top neighbor
        p_count[:, :-1] += 1 # Has bottom neighbor
        
        # For now, all boundaries are SOLID, so we only count internal neighbors
        neighbor_count = p_count 
        
        # #region agent log
        fluid_neighbor_counts = neighbor_count[fluid_mask]
        if len(fluid_neighbor_counts) > 0:
            log_debug("A", "neighbor_count stats", {
                "min": float(np.min(fluid_neighbor_counts)),
                "max": float(np.max(fluid_neighbor_counts)),
                "zeros": int(np.sum(fluid_neighbor_counts == 0)),
                "negatives": int(np.sum(fluid_neighbor_counts < 0))
            })
        # #endregion

        for _ in range(ITERATIONS):
            p_sum = np.zeros_like(self.p)
            p_sum[1:, :] += self.p[:-1, :] # Left
            p_sum[:-1, :] += self.p[1:, :] # Right
            p_sum[:, 1:] += self.p[:, :-1] # Top
            p_sum[:, :-1] += self.p[:, 1:] # Bottom
            
            # The key for free surface: p=0 in AIR cells
            # Our fluid_mask already handles this
            # Add small epsilon to avoid div by zero if no neighbors
            self.p[fluid_mask] = (p_sum[fluid_mask] - div[fluid_mask]) / (neighbor_count[fluid_mask] + 1e-6)
            self.p[~fluid_mask] = 0 # Explicitly set air cells to 0 pressure
        
        # #region agent log
        if np.any(np.isnan(self.p)) or np.any(np.isinf(self.p)):
            log_debug("A", "pressure solve exploded", {
                "nans": int(np.isnan(self.p).sum()),
                "infs": int(np.isinf(self.p).sum()),
                "max_p": float(np.max(np.abs(self.p[~np.isnan(self.p)])))
            })
        # #endregion

        # 4. Apply Pressure Gradient
        self.u[1:-1, :] -= (self.p[1:, :] - self.p[:-1, :])
        self.v[:, 1:-1] -= (self.p[:, 1:] - self.p[:, :-1])

    def g2p(self):
        """Vectorized Grid to Particles (PIC/FLIP)."""
        px, py = self.particle_pos[:, 0] / CELL_SIZE, self.particle_pos[:, 1] / CELL_SIZE
        
        # Interpolate U
        gx, gy = px, py - 0.5
        ix, iy = gx.astype(int), gy.astype(int)
        fx, fy = gx - ix, gy - iy
        mask_u = (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY - 1)
        
        u_pic = np.zeros(len(self.particle_pos), dtype=np.float32)
        u_prev = np.zeros_like(u_pic)
        
        m = mask_u
        ixm, iym, fxm, fym = ix[m], iy[m], fx[m], fy[m]
        u_pic[m] = (1-fxm)*(1-fym)*self.u[ixm, iym] + fxm*(1-fym)*self.u[ixm+1, iym] + \
                   (1-fxm)*fym*self.u[ixm, iym+1] + fxm*fym*self.u[ixm+1, iym+1]
        u_prev[m] = (1-fxm)*(1-fym)*self.u_prev[ixm, iym] + fxm*(1-fym)*self.u_prev[ixm+1, iym] + \
                    (1-fxm)*fym*self.u_prev[ixm, iym+1] + fxm*fym*self.u_prev[ixm+1, iym+1]

        # Interpolate V
        gx, gy = px - 0.5, py
        ix, iy = gx.astype(int), gy.astype(int)
        fx, fy = gx - ix, gy - iy
        mask_v = (ix >= 0) & (ix < NX - 1) & (iy >= 0) & (iy < NY)
        
        v_pic = np.zeros(len(self.particle_pos), dtype=np.float32)
        v_prev = np.zeros_like(v_pic)
        
        m = mask_v
        ixm, iym, fxm, fym = ix[m], iy[m], fx[m], fy[m]
        v_pic[m] = (1-fxm)*(1-fym)*self.v[ixm, iym] + fxm*(1-fym)*self.v[ixm+1, iym] + \
                   (1-fxm)*fym*self.v[ixm, iym+1] + fxm*fym*self.v[ixm+1, iym+1]
        v_prev[m] = (1-fxm)*(1-fym)*self.v_prev[ixm, iym] + fxm*(1-fym)*self.v_prev[ixm+1, iym] + \
                    (1-fxm)*fym*self.v_prev[ixm, iym+1] + fxm*fym*self.v_prev[ixm+1, iym+1]

        # PIC/FLIP update
        v_flip_x = self.particle_vel[:, 0] + (u_pic - u_prev)
        v_flip_y = self.particle_vel[:, 1] + (v_pic - v_prev)
        
        self.particle_vel[:, 0] = PIC_FLIP_RATIO * u_pic + (1 - PIC_FLIP_RATIO) * v_flip_x
        self.particle_vel[:, 1] = PIC_FLIP_RATIO * v_pic + (1 - PIC_FLIP_RATIO) * v_flip_y
        
        # Clamp velocity to prevent explosions (CFL condition)
        vel_mag = np.sqrt(self.particle_vel[:, 0]**2 + self.particle_vel[:, 1]**2)
        overspeed = vel_mag > MAX_VEL
        if np.any(overspeed):
            self.particle_vel[overspeed] *= (MAX_VEL / vel_mag[overspeed])[:, None]

    def update(self):
        if len(self.particle_pos) == 0: return
        
        # 1. P2G
        self.p2g()
        
        # 2. Gravity
        self.v[:, :] += GRAVITY * DT
        
        # 3. Boundaries (Solid walls)
        self.u[0, :] = 0
        self.u[-1, :] = 0
        self.v[:, 0] = 0
        self.v[:, -1] = 0
        
        # 4. Pressure
        self.solve_pressure()
        
        # 5. G2P
        self.g2p()
        
        # 6. Advect
        self.particle_pos += self.particle_vel * DT
        
        # #region agent log
        max_vel = float(np.max(np.linalg.norm(self.particle_vel, axis=1))) if len(self.particle_vel) > 0 else 0
        if max_vel > 20 or np.any(np.isnan(self.particle_pos)):
            log_debug("C", "advection instability", {
                "max_vel": max_vel,
                "nans": int(np.isnan(self.particle_pos).sum()),
                "num_particles": len(self.particle_pos)
            })
        # #endregion
        
        # 7. Collisions (Keep particles inside)
        self.particle_pos[:, 0] = np.clip(self.particle_pos[:, 0], 2, WIDTH - 2)
        self.particle_pos[:, 1] = np.clip(self.particle_pos[:, 1], 2, HEIGHT - 2)

    def draw(self, screen):
        # Draw water as smooth blobs using a temporary surface
        fluid_surf = pygame.Surface((WIDTH, HEIGHT))
        fluid_surf.set_colorkey((0, 0, 0))
        
        # Pre-render a water blob
        blob_r = 12
        blob = pygame.Surface((blob_r*2, blob_r*2), pygame.SRCALPHA)
        for r in range(blob_r, 0, -1):
            alpha = int(150 * (1 - r/blob_r)**2)
            pygame.draw.circle(blob, (40, 120, 255, alpha), (blob_r, blob_r), r)

        for p in self.particle_pos:
            fluid_surf.blit(blob, (int(p[0]) - blob_r, int(p[1]) - blob_r), special_flags=pygame.BLEND_RGBA_ADD)
            
        # Draw some sparkles/highlights
        for p in self.particle_pos[::10]: # only some particles
            pygame.draw.circle(fluid_surf, (200, 230, 255, 50), p.astype(int), 2)
            
        screen.blit(fluid_surf, (0, 0))

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Real-time Incompressible Fluid Sim (MAC/FLIP)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)

    sim = FluidSimulator()

    running = True
    while running:
        screen.fill((10, 12, 18))
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_x:
                    sim.clear()
                if event.key == pygame.K_UP:
                    sim.water_amount = min(200, sim.water_amount + 10)
                if event.key == pygame.K_DOWN:
                    sim.water_amount = max(10, sim.water_amount - 10)

        mouse_pressed = pygame.mouse.get_pressed()
        mouse_pos = np.array(pygame.mouse.get_pos(), dtype=np.float32)
        
        if mouse_pressed[0]:
            sim.add_water(mouse_pos)
        if mouse_pressed[2]:
            sim.remove_water(mouse_pos)

        sim.update()
        sim.draw(screen)

        # UI
        fps = int(clock.get_fps())
        text = font.render(f"FPS: {fps} | Particles: {len(sim.particle_pos)}", True, (255, 255, 255))
        controls = font.render("Left Click: Pour | Right Click: Remove | X: Reset", True, (150, 150, 150))
        screen.blit(text, (10, 10))
        screen.blit(controls, (10, 30))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
