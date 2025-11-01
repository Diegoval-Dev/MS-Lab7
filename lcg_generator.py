import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kstest

# ============================================================
# Generador Congruencial Lineal (Linear Congruential Generator)
# ============================================================

class LCG:
    def __init__(self, seed: int, a: int, c: int, m: int):
        """
        Inicializa el generador LCG con los parámetros:
        X_{n+1} = (a * X_n + c) mod m
        """
        self.seed = seed
        self.a = a
        self.c = c
        self.m = m
        self.state = seed

    def next(self) -> int:
        """Genera el siguiente número pseudoaleatorio entero."""
        self.state = (self.a * self.state + self.c) % self.m
        return self.state

    def random(self) -> float:
        """Devuelve un número pseudoaleatorio en el intervalo (0, 1)."""
        return self.next() / self.m

    def sample(self, n: int) -> np.ndarray:
        """Genera una muestra de tamaño n de números uniformes en (0,1)."""
        return np.array([self.random() for _ in range(n)])


# ============================================================
# Funciones auxiliares para análisis estadístico y visualización
# ============================================================

def analizar_muestra(muestra: np.ndarray, titulo: str):
    """
    Muestra estadísticas, histograma y prueba de Kolmogorov–Smirnov.
    """
    print("=" * 70)
    print(f"Análisis para {titulo}")
    print("=" * 70)

    # Estadísticos básicos
    media = np.mean(muestra)
    varianza = np.var(muestra)
    minimo = np.min(muestra)
    maximo = np.max(muestra)
    print(f"Media: {media:.4f}")
    print(f"Varianza: {varianza:.4f}")
    print(f"Mínimo: {minimo:.4f}")
    print(f"Máximo: {maximo:.4f}")

    # Prueba de Kolmogorov-Smirnov contra U(0,1)
    ks_stat, p_value = kstest(muestra, 'uniform')
    print(f"Estadístico KS: {ks_stat:.4f}")
    print(f"Valor-p: {p_value:.4f}")
    if p_value > 0.05:
        print("No se rechaza H0: la muestra parece provenir de U(0,1).")
    else:
        print("Se rechaza H0: la muestra NO parece provenir de U(0,1).")

    # Histograma
    plt.figure(figsize=(8, 4))
    sns.histplot(muestra, bins=20, kde=True, color='steelblue')
    plt.title(f"Histograma de muestra - {titulo}")
    plt.xlabel("Valores generados")
    plt.ylabel("Frecuencia")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


# ============================================================
# Ejecución principal
# ============================================================

def main():
    N = 10000

    params = [
        {"a": 1664525, "c": 1013904223, "m": 2**32, "seed": 12345, "nombre": "Conjunto 1 (Parámetros estándar C)"},
        {"a": 1103515245, "c": 12345, "m": 2**31, "seed": 6789, "nombre": "Conjunto 2 (Parámetros ANSI C)"}
    ]

    for p in params:
        lcg = LCG(seed=p["seed"], a=p["a"], c=p["c"], m=p["m"])
        muestra = lcg.sample(N)
        analizar_muestra(muestra, p["nombre"])


if __name__ == "__main__":
    main()
