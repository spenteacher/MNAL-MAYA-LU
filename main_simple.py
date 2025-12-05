"""
Simulador de Equilibrio - Descomposición LU
VERSIÓN CORREGIDA: Física visual con comparación LU vs Gauss
"""

import sys

import numpy as np
import pygame

from lu_solver import benchmark_solvers
from physics_simple import EquilibriumSystem

WIDTH, HEIGHT = 1200, 700
FPS = 60
BG_COLOR = (20, 20, 30)
UI_BG = (40, 40, 50)
TEXT_COLOR = (220, 220, 220)
ACCENT_COLOR = (100, 200, 255)
DRAG_COLOR = (255, 255, 100)


class Simulator:
    """Simulador principal"""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Simulador de Equilibrio - Descomposición LU")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)

        # Sistemas
        self.system_lu = EquilibriumSystem(use_lu=True)
        self.system_gauss = EquilibriumSystem(use_lu=False)

        # Estado
        self.paused = False
        self.show_info = True
        self.dragging = None
        self.dragging_gauss = None

        # Métricas
        self.frame_times_lu = []
        self.frame_times_gauss = []
        self.max_history = 100

        self.setup_scene()

    def setup_scene(self):
        """Crear escena"""
        # LU (izquierda)
        self.system_lu.create_grid(
            start_x=100, start_y=100, rows=7, cols=7, spacing=35, k=100.0, fix_top=True
        )

        # Gauss (derecha)
        self.system_gauss.create_grid(
            start_x=650, start_y=100, rows=7, cols=7, spacing=35, k=100.0, fix_top=True
        )

    def handle_events(self):
        """Procesar eventos"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.reset()
                elif event.key == pygame.K_i:
                    self.show_info = not self.show_info
                elif event.key == pygame.K_b:
                    self.run_benchmark()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.handle_mouse_down(event.pos)

            elif event.type == pygame.MOUSEBUTTONUP:
                if self.dragging:
                    self.dragging.external_force = np.array([0.0, 0.0])
                    self.dragging.fixed = self.dragging.permanently_fixed
                self.dragging = None

                if self.dragging_gauss:
                    self.dragging_gauss.external_force = np.array([0.0, 0.0])
                    self.dragging_gauss.fixed = self.dragging_gauss.permanently_fixed
                self.dragging_gauss = None

            elif event.type == pygame.MOUSEMOTION:
                if self.dragging:
                    # Fijar partícula arrastrada en posición del ratón
                    self.dragging.pos = np.array(event.pos, dtype=float)
                    self.dragging.velocity = np.array([0.0, 0.0])
                    self.dragging.fixed = True

                if self.dragging_gauss:
                    self.dragging_gauss.pos = np.array(event.pos, dtype=float)
                    self.dragging_gauss.velocity = np.array([0.0, 0.0])
                    self.dragging_gauss.fixed = True

        return True

    def handle_mouse_down(self, pos):
        """Detectar click en partícula"""
        # Sistema LU
        for p in self.system_lu.particles:
            dist = np.linalg.norm(p.pos - np.array(pos))
            if dist < 20 and not p.permanently_fixed:
                self.dragging = p
                return

        # Sistema Gauss
        for p in self.system_gauss.particles:
            dist = np.linalg.norm(p.pos - np.array(pos))
            if dist < 20 and not p.permanently_fixed:
                self.dragging_gauss = p
                return

    def reset(self):
        """Reiniciar"""
        self.system_lu = EquilibriumSystem(use_lu=True)
        self.system_gauss = EquilibriumSystem(use_lu=False)
        self.setup_scene()
        self.frame_times_lu.clear()
        self.frame_times_gauss.clear()
        self.dragging = None
        self.dragging_gauss = None

    def update(self):
        """Actualizar física"""
        if not self.paused:
            # LU
            try:
                self.system_lu.update()
                self.frame_times_lu.append(self.system_lu.solve_time * 1000)
            except:
                self.frame_times_lu.append(0)

            # Gauss
            try:
                self.system_gauss.update()
                self.frame_times_gauss.append(self.system_gauss.solve_time * 1000)
            except:
                self.frame_times_gauss.append(0)

            if len(self.frame_times_lu) > self.max_history:
                self.frame_times_lu.pop(0)
                self.frame_times_gauss.pop(0)

    def draw(self):
        """Renderizar"""
        self.screen.fill(BG_COLOR)

        self.draw_system(self.system_lu, "LU Decomposition", True, self.dragging)
        self.draw_system(
            self.system_gauss, "Gaussian Elimination", False, self.dragging_gauss
        )

        self.draw_ui()

        pygame.display.flip()

    def draw_system(self, system, title, is_lu, dragged):
        """Dibujar sistema"""
        # Título
        title_surf = self.font_large.render(title, True, ACCENT_COLOR)
        x_offset = 100 if is_lu else 650
        self.screen.blit(title_surf, (x_offset, 50))

        # Resortes
        for spring in system.springs:
            try:
                p1_pos = spring.p1.pos.astype(int)
                p2_pos = spring.p2.pos.astype(int)

                distance = np.linalg.norm(spring.p2.pos - spring.p1.pos)
                strain = abs(distance - spring.rest_length) / spring.rest_length
                color_val = min(255, int(strain * 300))
                color = (color_val, 100, 255 - color_val)

                pygame.draw.line(self.screen, color, p1_pos, p2_pos, 2)
            except:
                continue

        # Partículas
        for p in system.particles:
            try:
                pos = p.pos.astype(int)

                if p == dragged:
                    color = DRAG_COLOR
                    radius = 8
                elif p.permanently_fixed:
                    color = (255, 100, 100)
                    radius = 6
                else:
                    color = (100, 255, 100) if is_lu else (255, 200, 100)
                    radius = 5

                pygame.draw.circle(self.screen, color, pos, radius)
            except:
                continue

    def draw_ui(self):
        """Interfaz"""
        y_offset = HEIGHT - 150
        pygame.draw.rect(self.screen, UI_BG, (0, y_offset, WIDTH, 150))

        if len(self.frame_times_lu) > 10:
            avg_lu = np.mean(self.frame_times_lu[-30:])
            avg_gauss = np.mean(self.frame_times_gauss[-30:])
            speedup = avg_gauss / avg_lu if avg_lu > 0.001 else 1.0

            stats = [
                f"LU: {avg_lu:.3f} ms/frame",
                f"Gauss: {avg_gauss:.3f} ms/frame",
                f"Speedup: {speedup:.2f}x",
                f"FPS: {self.clock.get_fps():.1f}",
                "ARRASTRA particulas con el raton",
            ]

            for i, stat in enumerate(stats):
                surf = self.font.render(stat, True, TEXT_COLOR)
                self.screen.blit(surf, (20, y_offset + 20 + i * 25))

        controls = [
            "SPACE: Pausar/Reanudar",
            "R: Reiniciar",
            "I: Info on/off",
            "B: Benchmark",
            "MOUSE: Arrastrar particulas",
        ]

        for i, control in enumerate(controls):
            surf = self.font.render(control, True, TEXT_COLOR)
            self.screen.blit(surf, (WIDTH - 380, y_offset + 20 + i * 25))

        if self.show_info:
            info = "Las particulas CUELGAN por gravedad"
            surf = self.font.render(info, True, ACCENT_COLOR)
            self.screen.blit(surf, (WIDTH // 2 - 180, y_offset + 5))

        self.draw_performance_graph(y_offset)

    def draw_performance_graph(self, y_offset):
        """Dibujar gráfica de rendimiento en tiempo real"""
        graph_x, graph_y = 500, y_offset + 20
        graph_width, graph_height = 250, 80

        # Fondo de la gráfica
        pygame.draw.rect(
            self.screen, (30, 30, 40), (graph_x, graph_y, graph_width, graph_height)
        )

        # Si hay suficientes datos
        if len(self.frame_times_lu) > 1 and len(self.frame_times_gauss) > 1:
            # Obtener los últmos 100 valores
            num_points = min(50, len(self.frame_times_lu))
            lu_data = self.frame_times_lu[-num_points:]
            gauss_data = self.frame_times_gauss[-num_points:]

            # Escala FIJA: siempre usar 5ms como máximo
            max_val = 5.0  # Escala fija en milisegundos

            # Crear puntos para las líneas
            points_lu = []
            points_gauss = []
            for i in range(num_points):
                x_pos = graph_x + (i / num_points) * graph_width
                # Limitar valores para que no se salgan del recuadro del grafico
                y_lu = (
                    graph_y
                    + graph_height
                    - min(lu_data[i] / max_val, 1.0) * graph_height
                )
                y_gauss = (
                    # el /3 es para poner en la misma posición los x de lu y gauss
                    graph_y
                    + graph_height
                    - min(gauss_data[i] / max_val, 3.0) * graph_height / 2.80
                )

                points_lu.append((x_pos, y_lu))
                points_gauss.append((x_pos, y_gauss))

            # Dibujar líneas
            if len(points_lu) > 1:
                pygame.draw.lines(self.screen, (100, 255, 100), False, points_lu, 2)
            if len(points_gauss) > 1:
                pygame.draw.lines(self.screen, (255, 200, 100), False, points_gauss, 2)

        label = self.font.render("Performance", True, TEXT_COLOR)
        self.screen.blit(label, (graph_x + 5, graph_y + 5))

    def run_benchmark(self):
        """Benchmark"""
        print("\n" + "=" * 50)
        print("BENCHMARK - LU vs GAUSS")
        print("=" * 50)

        for n in [10, 20, 30, 50, 100]:
            print(f"\nMatriz {n}x{n}:")
            results = benchmark_solvers(n, num_solves=100)
            print(f"  LU decomp: {results['lu_decomp_time'] * 1000:.2f} ms")
            print(f"  LU solve:  {results['lu_solve_time'] * 1000:.3f} ms")
            print(f"  LU total:  {results['lu_total_time'] * 1000:.2f} ms")
            print(f"  Gauss:     {results['gauss_total_time'] * 1000:.2f} ms")
            print(f"  Speedup:   {results['speedup']:.2f}x")

        print("\n" + "=" * 50)

    def run(self):
        """Loop principal"""
        running = True
        while running:
            self.clock.tick(FPS)
            running = self.handle_events()
            self.update()
            self.draw()

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    print("=" * 60)
    print("Simulador Visual - LU vs Gauss")
    print("=" * 60)
    print("Las particulas CUELGAN por gravedad")
    print("Arrastralas para ver la deformacion")
    print("=" * 60)

    sim = Simulator()
    sim.run()
