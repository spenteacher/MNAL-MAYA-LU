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


class Spring:
    """Resorte que conecta dos partículas"""

    def __init__(
        self, p1: Particle, p2: Particle, k: float = 100.0, rest_length: float = None
    ):
        self.p1 = p1
        self.p2 = p2
        self.k = k
        self.rest_length = (
            rest_length if rest_length else np.linalg.norm(p1.pos - p2.pos)
        )


class EquilibriumSystem:
    """
    Sistema que USA LU o GAUSS para resolver K·x = f
    Ahora con gravedad correctamente aplicada
    """

    def __init__(self, use_lu: bool = True):
        self.particles = []
        self.springs = []
        self.use_lu = use_lu
        self.solver = LUSolver() if use_lu else GaussianSolver()
        self.gravity = np.array([0.0, 500.0])  # Gravedad hacia abajo
        self.solve_time = 0
        self.K_matrix = None
        self.K_decomposed = False

    def add_particle(self, x: float, y: float, fixed: bool = False) -> Particle:
        p = Particle(x, y, fixed)
        self.particles.append(p)
        return p

    def add_spring(
        self, p1: Particle, p2: Particle, k: float = 100.0, rest_length: float = None
    ):
        spring = Spring(p1, p2, k, rest_length)
        self.springs.append(spring)
        return spring

    def build_system(self):
        n_free = sum(1 for p in self.particles if not p.fixed)
        if n_free == 0:
            return None, None, None, None

        # Mapea las partículas con índices
        free_map = {}
        idx = 0
        for i, p in enumerate(self.particles):
            if not p.fixed:
                free_map[i] = idx
                idx += 1

        # Construir matriz K (solo estructura de resortes)
        K = np.zeros((n_free, n_free))
        f_x = np.zeros(n_free)
        f_y = np.zeros(n_free)

        for spring in self.springs:
            i1 = self.particles.index(spring.p1)
            i2 = self.particles.index(spring.p2)
            k = spring.k

            # Se actualiza K
            if not spring.p1.fixed and not spring.p2.fixed:
                idx1, idx2 = free_map[i1], free_map[i2]
                K[idx1, idx1] += k
                K[idx2, idx2] += k
                K[idx1, idx2] -= k
                K[idx2, idx1] -= k
            elif not spring.p1.fixed:
                idx1 = free_map[i1]
                K[idx1, idx1] += k
            elif not spring.p2.fixed:
                idx2 = free_map[i2]
                K[idx2, idx2] += k

            # Contribución al vector de fuerzas f
            # Calcular dirección del resorte en reposo
            rest_dir = spring.p2.rest_pos - spring.p1.rest_pos
            rest_length = np.linalg.norm(rest_dir)
            # Proteje de dividir por números muy cercanos al 0 y que salte error
            if rest_length > 1e-6:
                rest_dir = rest_dir / rest_length
                # Fuerza debida a la longitud natural del resorte
                force_magnitude = k * spring.rest_length
                force_x = force_magnitude * rest_dir[0]
                force_y = force_magnitude * rest_dir[1]

                if not spring.p1.fixed and not spring.p2.fixed:
                    idx1 = free_map[i1]
                    idx2 = free_map[i2]
                    f_x[idx1] -= force_x
                    f_y[idx1] -= force_y
                    f_x[idx2] += force_x
                    f_y[idx2] += force_y
                elif not spring.p1.fixed:
                    idx1 = free_map[i1]
                    # La partícula fija p2 ejerce fuerza sobre p1
                    f_x[idx1] += k * spring.p2.pos[0] - force_x
                    f_y[idx1] += k * spring.p2.pos[1] - force_y
                elif not spring.p2.fixed:
                    idx2 = free_map[i2]
                    # La partícula fija p1 ejerce fuerza sobre p2
                    f_x[idx2] += k * spring.p1.pos[0] + force_x
                    f_y[idx2] += k * spring.p1.pos[1] + force_y

        # Añadir gravedad
        for i, p in enumerate(self.particles):
            if not p.fixed:
                idx = free_map[i]
                f_x[idx] += self.gravity[0]
                f_y[idx] += self.gravity[1]

        # Regularización
        K += np.eye(n_free) * 0.1

        return K, f_x, f_y, free_map

    def update(self):
        """
        Resolver K·x = f usando LU o GAUSS
        """
        import time

        t0 = time.perf_counter()

        n_free = sum(1 for p in self.particles if not p.fixed)
        if n_free == 0:
            self.solve_time = 0
            return

        try:
            K, f_x, f_y, free_map = self.build_system()
            if K is None:
                return

            # Reconstruir K si cambió el número de partículas libres
            rebuild_K = self.K_matrix is None or self.K_matrix.shape[0] != K.shape[0]

            if rebuild_K:
                self.K_matrix = K
                self.K_decomposed = False

            # AQUÍ ESTÁ LA DIFERENCIA: LU vs GAUSS
            if self.use_lu:
                # LU: Descomponer una vez
                if not self.K_decomposed:
                    self.solver.decompose(K)
                    self.K_decomposed = True

                # Resolver usando LU
                x_new = self.solver.solve(f_x, reuse_decomposition=True)
                y_new = self.solver.solve(f_y, reuse_decomposition=True)
            else:
                # GAUSS: Desde cero cada vez
                x_new = self.solver.solve(K, f_x)
                y_new = self.solver.solve(K, f_y)

            if not (np.isfinite(x_new).all() and np.isfinite(y_new).all()):
                return

            # Actualizar posiciones con suavizado
            alpha = 0.2
            for i, p in enumerate(self.particles):
                if not p.fixed:
                    idx = free_map[i]
                    new_pos = np.array([x_new[idx], y_new[idx]])

                    # Calcular velocidad para animación
                    p.velocity = (new_pos - p.pos) * alpha * 10

                    # Actualizar posición
                    p.pos = p.pos * (1 - alpha) + new_pos * alpha
                    p.velocity *= 0.9  # Damping

                    # Límites
                    p.pos[0] = np.clip(p.pos[0], 10, 590)
                    p.pos[1] = np.clip(p.pos[1], 10, 500)

                    # Verificar validez
                    if not np.isfinite(p.pos).all():
                        p.pos = p.rest_pos.copy()
                        p.velocity = np.array([0.0, 0.0])

            self.solve_time = time.perf_counter() - t0

        except Exception as e:
            print(f"Error en sistema {'LU' if self.use_lu else 'Gauss'}: {e}")
            self.solve_time = 0

    def create_grid(
        self,
        start_x: float,
        start_y: float,
        rows: int,
        cols: int,
        spacing: float,
        k: float = 50.0,
        fix_top: bool = True,
    ):
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
                    self.add_spring(grid[i][j], grid[i][j + 1], k, spacing)
                if i < rows - 1:
                    self.add_spring(grid[i][j], grid[i + 1][j], k, spacing)
                if i < rows - 1 and j < cols - 1:
                    diag = spacing * np.sqrt(2)
                    self.add_spring(grid[i][j], grid[i + 1][j + 1], k * 0.25, diag)
                    self.add_spring(grid[i][j + 1], grid[i + 1][j], k * 0.25, diag)

        return grid
