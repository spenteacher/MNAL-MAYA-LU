"""
LU Decomposition Solver
Implementación optimizada con pivoteo parcial
"""

import numpy as np
from typing import Tuple
import time

class LUSolver:
    """
    Solver de sistemas lineales usando descomposición LU con pivoteo parcial.
    Optimizado para reutilizar la factorización cuando la matriz no cambia.
    """

    def __init__(self):
        self.L = None
        self.U = None
        self.P = None
        self.decomposition_time = 0
        self.solve_time = 0

    def decompose(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Descomposición PA = LU con pivoteo parcial

        Args:
            A: Matriz cuadrada nxn

        Returns:
            L, U, P: Matrices de la descomposición
        """
        start_time = time.perf_counter()

        n = A.shape[0]
        L = np.eye(n)
        U = A.copy().astype(float)
        P = np.eye(n)

        for k in range(n-1):
            # Pivoteo parcial: buscar el elemento de mayor magnitud
            pivot_row = k + np.argmax(np.abs(U[k:, k]))

            if pivot_row != k:
                # Intercambiar filas
                U[[k, pivot_row]] = U[[pivot_row, k]]
                P[[k, pivot_row]] = P[[pivot_row, k]]
                if k > 0:
                    L[[k, pivot_row], :k] = L[[pivot_row, k], :k]

            # Eliminación gaussiana
            for i in range(k+1, n):
                if U[k, k] != 0:
                    factor = U[i, k] / U[k, k]
                    L[i, k] = factor
                    U[i, k:] -= factor * U[k, k:]

        self.L = L
        self.U = U
        self.P = P
        self.decomposition_time = time.perf_counter() - start_time

        return L, U, P

    def solve(self, b: np.ndarray, reuse_decomposition: bool = False) -> np.ndarray:
        """
        Resolver Ax = b usando la descomposición LU previamente calculada

        Args:
            b: Vector independiente
            reuse_decomposition: Si True, reutiliza L, U, P previamente calculados

        Returns:
            x: Solución del sistema
        """
        start_time = time.perf_counter()

        if not reuse_decomposition or self.L is None:
            raise ValueError("Debe llamar a decompose() primero")

        # Aplicar permutación: Pb
        Pb = self.P @ b

        # Forward substitution: Ly = Pb
        n = len(b)
        y = np.zeros(n)
        for i in range(n):
            y[i] = Pb[i] - np.dot(self.L[i, :i], y[:i])

        # Backward substitution: Ux = y
        x = np.zeros(n)
        for i in range(n-1, -1, -1):
            x[i] = (y[i] - np.dot(self.U[i, i+1:], x[i+1:])) / self.U[i, i]

        self.solve_time = time.perf_counter() - start_time

        return x

    def solve_full(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Resolver Ax = b calculando la descomposición y resolviendo
        """
        self.decompose(A)
        return self.solve(b, reuse_decomposition=True)


class GaussianSolver:
    """
    Solver usando eliminación gaussiana directa (para comparación)
    """

    def __init__(self):
        self.solve_time = 0

    def solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Resolver Ax = b usando eliminación gaussiana con pivoteo
        """
        start_time = time.perf_counter()

        n = len(b)
        Ab = np.column_stack([A.copy().astype(float), b.copy().astype(float)])

        # Eliminación hacia adelante
        for k in range(n-1):
            # Pivoteo
            pivot_row = k + np.argmax(np.abs(Ab[k:, k]))
            if pivot_row != k:
                Ab[[k, pivot_row]] = Ab[[pivot_row, k]]

            # Eliminación
            for i in range(k+1, n):
                if Ab[k, k] != 0:
                    factor = Ab[i, k] / Ab[k, k]
                    Ab[i, k:] -= factor * Ab[k, k:]

        # Sustitución hacia atrás
        x = np.zeros(n)
        for i in range(n-1, -1, -1):
            x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:])) / Ab[i, i]

        self.solve_time = time.perf_counter() - start_time

        return x


def benchmark_solvers(n: int, num_solves: int = 100) -> dict:
    """
    Comparar rendimiento de LU vs Gauss para múltiples resoluciones
    """
    # Crear sistema de prueba (simétrico definido positivo)
    A = np.random.randn(n, n)
    A = A + A.T + n * np.eye(n)

    results = {
        'n': n,
        'num_solves': num_solves,
        'lu_decomp_time': 0,
        'lu_solve_time': 0,
        'lu_total_time': 0,
        'gauss_total_time': 0,
    }

    # Benchmark LU (descomponer una vez, resolver muchas veces)
    lu = LUSolver()
    lu.decompose(A)
    results['lu_decomp_time'] = lu.decomposition_time

    lu_solve_times = []
    for _ in range(num_solves):
        b = np.random.randn(n)
        lu.solve(b, reuse_decomposition=True)
        lu_solve_times.append(lu.solve_time)

    results['lu_solve_time'] = np.mean(lu_solve_times)
    results['lu_total_time'] = results['lu_decomp_time'] + results['lu_solve_time'] * num_solves

    # Benchmark Gauss (resolver desde cero cada vez)
    gauss = GaussianSolver()
    gauss_times = []
    for _ in range(num_solves):
        b = np.random.randn(n)
        gauss.solve(A, b)
        gauss_times.append(gauss.solve_time)

    results['gauss_total_time'] = np.sum(gauss_times)
    results['speedup'] = results['gauss_total_time'] / results['lu_total_time']

    return results
