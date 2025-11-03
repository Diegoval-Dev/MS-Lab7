import numpy as np
from scipy.stats import geom, ks_2samp, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import csv
from pathlib import Path

def counts_by_value(x: np.ndarray):
    vals, cnts = np.unique(x, return_counts=True)
    return dict(zip(vals, cnts))

def build_bins(ct: dict, ce: dict, min_per_cell: int = 5):
    all_vals = sorted(set(ct.keys()) | set(ce.keys()))
    bins, current_bin, current_total = [], [], 0
    for v in all_vals:
        c = ct.get(v, 0) + ce.get(v, 0)
        if c >= min_per_cell:
            if current_bin:
                bins.append(current_bin)
                current_bin, current_total = [], 0
            bins.append([v])
        else:
            current_bin.append(v)
            current_total += c
    if current_bin:
        if bins:
            bins[-1].extend(current_bin)
        else:
            bins.append(current_bin)
    return bins

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=5000, help="tamaño de muestra")
    ap.add_argument("--p", type=float, default=0.30, help="parámetro p de Geom(p)")
    ap.add_argument("--seed", type=int, default=2025, help="semilla")
    ap.add_argument("--no-plot", action="store_true", help="no generar gráficas")
    ap.add_argument("--outdir", type=str, default="out_geom", help="carpeta de salida")
    args = ap.parse_args()

    N, p, seed = args.N, args.p, args.seed
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    print("="*70)
    print("Comparación de muestras Geom(p)")
    print(f"N = {N:,}, p = {p}, alpha = 0.05")
    print("="*70)

    # Muestras
    sample_theoretical = geom.rvs(p, size=N, random_state=rng)
    U = rng.random(N)
    sample_empirical = 1 + np.floor(np.log(1.0 - U) / np.log(1.0 - p)).astype(int)

    # Chi-cuadrado (homogeneidad)
    ct, ce = counts_by_value(sample_theoretical), counts_by_value(sample_empirical)
    bins = build_bins(ct, ce, min_per_cell=5)
    contingency = []
    for b in bins:
        contingency.append([
            int(sum(ct.get(v, 0) for v in b)),
            int(sum(ce.get(v, 0) for v in b))
        ])
    contingency = np.array(contingency, dtype=int)
    chi2_stat, chi2_p, chi2_df, _ = chi2_contingency(contingency)

    # KS dos muestras
    ks_stat, ks_p = ks_2samp(sample_theoretical, sample_empirical, alternative='two-sided', mode='auto')

    # Reporte en consola
    print("\nCHI-CUADRADO (homogeneidad)")
    print(f"Bins usados: {len(bins)}")
    print(f"Estadístico χ² = {chi2_stat:.4f}, gl = {chi2_df}, p-value = {chi2_p:.4f}")
    print("Conclusión χ²:", "RECHAZAMOS" if chi2_p < 0.05 else "NO rechazamos", "que provienen de la misma distribución.")

    print("\nKOLMOGOROV–SMIRNOV (dos muestras)")
    print(f"Estadístico D = {ks_stat:.4f}, p-value = {ks_p:.4f}")
    print("Conclusión KS:", "RECHAZAMOS" if ks_p < 0.05 else "NO rechazamos", "que provienen de la misma distribución.")

    # Exportar tabla de frecuencias por valor (útil para el informe)
    max_k = int(max(sample_theoretical.max(), sample_empirical.max()))
    freq_path = outdir / f"frecuencias_N{N}_p{p}.csv"
    with freq_path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["k", "freq_teorica", "freq_empirica"])
        for k in range(1, max_k + 1):
            wr.writerow([k, ct.get(k, 0), ce.get(k, 0)])
    print(f"\nFrecuencias guardadas en: {freq_path}")

    # Gráficas → guardar a archivos (sin plt.show())
    if not args.no_plot:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        sns.histplot(sample_theoretical, stat="probability", discrete=True,
                     bins=range(1, sample_theoretical.max()+2), ax=ax[0])
        ax[0].set_title("Teórica (scipy.stats.geom)"); ax[0].set_xlabel("k")
        sns.histplot(sample_empirical, stat="probability", discrete=True,
                     bins=range(1, sample_empirical.max()+2), ax=ax[1])
        ax[1].set_title("Empírica (transformada integral)"); ax[1].set_xlabel("k")
        fig.suptitle(f"Geom(p={p}) — Distribuciones empíricas")
        plt.tight_layout()
        plot_path = outdir / f"hist_N{N}_p{p}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Figura guardada en: {plot_path}")

if __name__ == "__main__":
    main()
