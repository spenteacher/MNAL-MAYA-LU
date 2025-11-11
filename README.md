# Simulador de Equilibrio - Descomposición LU

**Proyecto de Métodos Numéricos**

## Modelo Final Simplificado

Este simulador demuestra la ventaja de la descomposición LU mediante:

1. **Visualización física**: Partículas conectadas por resortes que cuelgan por gravedad
2. **Benchmarks**: Comparación directa LU vs Gauss en sistemas lineales

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
python main_simple.py
```
## Controles

- **SPACE**: Pausar/Reanudar
- **R**: Reiniciar
- **I**: Info on/off
- **B**: **Ejecutar benchmark** (aquí se ve la diferencia LU vs Gauss)
- **MOUSE**: Arrastrar partículas

## Estructura del Proyecto

- `lu_solver.py`: Implementación LU y Gauss (aquí está la comparación)
- `physics_simple.py`: Sistema de partículas (visualización)
- `main_simple.py`: Interfaz Pygame
- `informe-final.tex`: Documento teórico completo

## Benchmark de Ejemplo

Al presionar **B** verás:

```
==================================================
BENCHMARK - LU vs GAUSS
==================================================

Matriz 50x50:
  LU decomp: 1.89 ms
  LU solve:  0.52 ms (promedio)
  LU total:  53.89 ms (100 resoluciones)
  Gauss:     473.0 ms (100 resoluciones)
  Speedup:   8.78x  ← AQUÍ ESTÁ LA VENTAJA
==================================================
```

## Compilar LaTeX

```bash
pdflatex informe-final.tex
```

El documento explica todo el fundamento teórico y muestra por qué LU es superior.
