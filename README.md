# Simulador de Equilibrio - Descomposición LU

**Proyecto de Métodos Numéricos - Sin Ecuaciones Diferenciales**

## Modelo Final Simplificado

Este simulador demuestra la ventaja de la descomposición LU mediante:

1. **Visualización física**: Partículas conectadas por resortes que cuelgan por gravedad
2. **Benchmarks**: Comparación directa LU vs Gauss en sistemas lineales

## ⚠️ Nota Importante sobre el Modelo

Por simplicidad y estabilidad visual, ambos lados del simulador usan **el mismo método físico** (fuerzas directas). La comparación LU vs Gauss se demuestra mediante:

- **Benchmarks en consola** (presiona B): Compara LU vs Gauss en sistemas lineales puros
- **Análisis teórico** en el documento LaTeX
- **Implementación completa** de ambos métodos en `lu_solver.py`

La visualización sirve para **ilustrar el tipo de sistema** donde LU es ventajoso (sistemas que se resuelven repetidamente), mientras que los **benchmarks muestran la diferencia cuantitativa**.

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
python main_simple.py
```

## Generar Imagen para LaTeX

```bash
python generate_screenshot.py
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
- `generate_screenshot.py`: Generar imagen

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

## Matemáticas Requeridas

✅ Álgebra lineal (matrices, sistemas)  
✅ Análisis de complejidad  
✅ Implementación de algoritmos  

❌ NO requiere: EDOs, derivadas, integrales
