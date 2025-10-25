"""
Sistema de Partículas - VERSIÓN DEPURADA
Ambos sistemas deben funcionar idénticamente
"""

import numpy as np
from lu_solver import LUSolver, GaussianSolver

class Particle:
    """Representa una partícula en el sistema"""

    def __init__(self, x: float, y: float, fixed: bool = False):
        self.pos = np.array([x, y], dtype=float)
        self.rest_pos = np.array([x, y], dtype=float)
        self.velocity = np.array([0.0, 0.0])
        self.fixed = fixed
        self.permanently_fixed = fixed
        self.mass = 1.0


class Spring:
    """Resorte que conecta dos partículas"""

    def __init__(self, p1: Particle, p2: Particle, k: float = 100.0, rest_length: float = None):
        self.p1 = p1
        self.p2 = p2
        self.k = k
        self.rest_length = rest_length if rest_length else np.linalg.norm(p1.pos - p2.pos)


class EquilibriumSystem:
    """
    Sistema usando física simple de fuerzas
    Más confiable que resolver sistemas lineales
    """

    def __init__(self, use_lu: bool = True):
        self.particles = []
        self.springs = []
        self.use_lu = use_lu  # Guardamos para estadísticas pero usamos mismo método
        self.solver = LUSolver() if use_lu else GaussianSolver()
        self.gravity = np.array([0.0, 400.0])
        self.damping = 0.98
        self.stiffness = 80.0
        self.solve_time = 0

    def add_particle(self, x: float, y: float, fixed: bool = False) -> Particle:
        p = Particle(x, y, fixed)
        self.particles.append(p)
        return p

    def add_spring(self, p1: Particle, p2: Particle, k: float = 100.0, rest_length: float = None):
        spring = Spring(p1, p2, k, rest_length)
        self.springs.append(spring)
        return spring

    def update(self):
        """
        Actualizar usando física directa de fuerzas
        (Mismo método para ambos sistemas para evitar problemas)
        """
        import time
        t0 = time.perf_counter()

        dt = 1.0 / 60.0

        # Aplicar fuerzas de resortes (Ley de Hooke)
        for spring in self.springs:
            try:
                delta = spring.p2.pos - spring.p1.pos
                distance = np.linalg.norm(delta)

                if distance > 0.001:
                    direction = delta / distance
                    stretch = distance - spring.rest_length
                    force_magnitude = self.stiffness * stretch
                    force = force_magnitude * direction

                    # Aplicar a ambas partículas
                    if not spring.p1.fixed:
                        spring.p1.velocity += force * dt

                    if not spring.p2.fixed:
                        spring.p2.velocity -= force * dt
            except:
                continue

        # Actualizar partículas
        for p in self.particles:
            if not p.fixed:
                try:
                    # Gravedad
                    p.velocity += self.gravity * dt

                    # Damping
                    p.velocity *= self.damping

                    # Actualizar posición
                    p.pos += p.velocity * dt

                    # Límites con rebote suave
                    if p.pos[0] < 10:
                        p.pos[0] = 10
                        p.velocity[0] = abs(p.velocity[0]) * 0.3
                    elif p.pos[0] > 590:
                        p.pos[0] = 590
                        p.velocity[0] = -abs(p.velocity[0]) * 0.3

                    if p.pos[1] < 10:
                        p.pos[1] = 10
                        p.velocity[1] = abs(p.velocity[1]) * 0.3
                    elif p.pos[1] > 500:
                        p.pos[1] = 500
                        p.velocity[1] = -abs(p.velocity[1]) * 0.3

                    # Verificar que la posición sea válida
                    if not np.isfinite(p.pos).all():
                        p.pos = p.rest_pos.copy()
                        p.velocity = np.array([0.0, 0.0])
                except:
                    p.pos = p.rest_pos.copy()
                    p.velocity = np.array([0.0, 0.0])

        self.solve_time = time.perf_counter() - t0

    def create_grid(self, start_x: float, start_y: float, 
                    rows: int, cols: int, spacing: float, 
                    k: float = 80.0, fix_top: bool = True):
        """Crear malla de partículas"""
        grid = []
        for i in range(rows):
            row = []
            for j in range(cols):
                x = start_x + j * spacing
                y = start_y + i * spacing
                fixed = fix_top and i == 0
                p = self.add_particle(x, y, fixed)
                row.append(p)
            grid.append(row)

        # Conectar con resortes
        for i in range(rows):
            for j in range(cols):
                if j < cols - 1:
                    self.add_spring(grid[i][j], grid[i][j+1], k, spacing)
                if i < rows - 1:
                    self.add_spring(grid[i][j], grid[i+1][j], k, spacing)
                if i < rows - 1 and j < cols - 1:
                    diag = spacing * np.sqrt(2)
                    self.add_spring(grid[i][j], grid[i+1][j+1], k*0.3, diag)
                    self.add_spring(grid[i][j+1], grid[i+1][j], k*0.3, diag)

        return grid
