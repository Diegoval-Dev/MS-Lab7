import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kstest

# ============================================================
# Generador Mersenne Twister
# ============================================================

def generar_muestra_mt(n: int, seed: int = None) -> np.ndarray:
    """
    Genera una muestra de tamaño n con el generador Mersenne Twister.
    La distribución es uniforme en (0,1).
    """
    rng = np.random.default_rng(seed) 
    return rng.random(n)


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

    media = np.mean(muestra)
    varianza = np.var(muestra)
    minimo = np.min(muestra)
    maximo = np.max(muestra)
    print(f"Media: {media:.4f}")
    print(f"Varianza: {varianza:.4f}")
    print(f"Mínimo: {minimo:.4f}")
    print(f"Máximo: {maximo:.4f}")

    ks_stat, p_value = kstest(muestra, 'uniform')
    print(f"Estadístico KS: {ks_stat:.4f}")
    print(f"Valor-p: {p_value:.4f}")
    if p_value > 0.05:
        print("No se rechaza H0: la muestra parece provenir de U(0,1).")
    else:
        print("Se rechaza H0: la muestra NO parece provenir de U(0,1).")

    plt.figure(figsize=(8, 4))
    sns.histplot(muestra, bins=20, kde=True, color='seagreen')
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

    seeds = [42, 2025]

    for i, seed in enumerate(seeds, start=1):
        muestra = generar_muestra_mt(N, seed)
        analizar_muestra(muestra, f"Conjunto {i} (Seed = {seed})")


if __name__ == "__main__":
    main()
