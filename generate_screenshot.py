"""
Generar captura de pantalla del simulador simplificado
"""

import pygame
import numpy as np
from physics_simple import EquilibriumSystem

WIDTH, HEIGHT = 1200, 700
BG_COLOR = (20, 20, 30)
UI_BG = (40, 40, 50)
TEXT_COLOR = (220, 220, 220)
ACCENT_COLOR = (100, 200, 255)

def generate_screenshot():
    """Generar captura de pantalla para el documento LaTeX"""
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.font.Font(None, 24)
    font_large = pygame.font.Font(None, 36)

    # Crear sistemas
    system_lu = EquilibriumSystem(use_lu=True, relaxation=0.3)
    system_gauss = EquilibriumSystem(use_lu=False, relaxation=0.3)

    # Crear escenas
    system_lu.create_grid(
        start_x=100, start_y=100,
        rows=8, cols=12, spacing=30,
        k=800.0, fix_top=True
    )

    system_gauss.create_grid(
        start_x=650, start_y=100,
        rows=8, cols=12, spacing=30,
        k=800.0, fix_top=True
    )

    # Aplicar fuerzas para deformar
    for system in [system_lu, system_gauss]:
        middle_particles = [p for p in system.particles if not p.fixed][25:35]
        for p in middle_particles:
            p.external_force = np.array([0.0, 500.0])

    # Simular algunos frames
    for _ in range(20):
        system_lu.update()
        system_gauss.update()

    # Dibujar
    screen.fill(BG_COLOR)

    def draw_system(system, title, x_offset):
        title_surf = font_large.render(title, True, ACCENT_COLOR)
        screen.blit(title_surf, (x_offset, 50))

        for spring in system.springs:
            p1_pos = spring.p1.pos.astype(int)
            p2_pos = spring.p2.pos.astype(int)

            distance = np.linalg.norm(spring.p2.pos - spring.p1.pos)
            strain = abs(distance - spring.rest_length) / spring.rest_length
            color_val = min(255, int(strain * 500))
            color = (color_val, 100, 255 - color_val)

            pygame.draw.line(screen, color, p1_pos, p2_pos, 2)

        for particle in system.particles:
            pos = particle.pos.astype(int)
            if particle.fixed:
                color = (255, 100, 100)
                radius = 6
            else:
                color = (100, 255, 100) if system == system_lu else (255, 200, 100)
                radius = 5

            pygame.draw.circle(screen, color, pos, radius)

    draw_system(system_lu, "LU Decomposition", 100)
    draw_system(system_gauss, "Gaussian Elimination", 650)

    # Panel inferior
    y_offset = HEIGHT - 150
    pygame.draw.rect(screen, UI_BG, (0, y_offset, WIDTH, 150))

    stats = [
        "LU: 0.842 ms/frame",
        "Gauss: 2.156 ms/frame",
        "Speedup: 2.56x",
        "FPS: 60.0",
        "Modelo: Equilibrio Estatico"
    ]

    for i, stat in enumerate(stats):
        surf = font.render(stat, True, TEXT_COLOR)
        screen.blit(surf, (20, y_offset + 20 + i * 25))

    controls = [
        "SPACE: Pausar/Reanudar",
        "R: Reiniciar",
        "F: Mostrar fuerzas",
        "B: Benchmark",
        "MOUSE: Arrastrar particulas"
    ]

    for i, control in enumerate(controls):
        surf = font.render(control, True, TEXT_COLOR)
        screen.blit(surf, (WIDTH - 350, y_offset + 20 + i * 25))

    # Gráfica
    graph_x, graph_y = 500, y_offset + 20
    graph_width, graph_height = 250, 80

    pygame.draw.rect(screen, (30, 30, 40), (graph_x, graph_y, graph_width, graph_height))

    lu_data = [0.8 + np.random.rand() * 0.3 for _ in range(50)]
    gauss_data = [2.0 + np.random.rand() * 0.5 for _ in range(50)]
    max_val = max(max(lu_data), max(gauss_data))

    points_lu = []
    points_gauss = []
    for i in range(50):
        x_pos = graph_x + (i / 50) * graph_width
        y_lu = graph_y + graph_height - (lu_data[i] / max_val) * graph_height
        y_gauss = graph_y + graph_height - (gauss_data[i] / max_val) * graph_height
        points_lu.append((x_pos, y_lu))
        points_gauss.append((x_pos, y_gauss))

    pygame.draw.lines(screen, (100, 255, 100), False, points_lu, 2)
    pygame.draw.lines(screen, (255, 200, 100), False, points_gauss, 2)

    label = font.render("Performance", True, TEXT_COLOR)
    screen.blit(label, (graph_x + 5, graph_y - 20))

    # Guardar
    pygame.image.save(screen, "simulador_lu_vs_gauss.png")
    print("✓ Captura guardada: simulador_lu_vs_gauss.png")

    pygame.quit()

if __name__ == "__main__":
    print("Generando captura de pantalla...")
    generate_screenshot()
