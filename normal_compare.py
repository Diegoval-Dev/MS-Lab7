import numpy as np
from scipy.stats import norm, ks_2samp, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

def chi2_homogeneity_on_quantile_bins(x1: np.ndarray, x2: np.ndarray, mu: float, sigma: float, k: int):
    
    eps = 1e-6
    probs = np.linspace(eps, 1 - eps, k + 1)
    edges = norm.ppf(probs, loc=mu, scale=sigma)

    h1, _ = np.histogram(x1, bins=edges)
    h2, _ = np.histogram(x2, bins=edges)

    contingency = np.vstack([h1, h2]).T  # B x 2
    chi2_stat, chi2_p, chi2_df, _ = chi2_contingency(contingency)
    return chi2_stat, chi2_p, chi2_df, edges, contingency

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=5000, help="tamaño de muestra")
    ap.add_argument("--mu", type=float, default=0.0, help="media")
    ap.add_argument("--sigma", type=float, default=1.0, help="desviación estándar (>0)")
    ap.add_argument("--seed", type=int, default=2025, help="semilla")
    ap.add_argument("--outdir", type=str, default="out_normal", help="carpeta de salida")
    ap.add_argument("--no-plot", action="store_true", help="no guardar figuras")
    args = ap.parse_args()

    N, mu, sigma, seed = args.N, args.mu, args.sigma, args.seed
    assert sigma > 0, "sigma debe ser > 0"
    rng = np.random.default_rng(seed)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Comparación de muestras Normal(mu, sigma^2)")
    print(f"N = {N:,}, mu = {mu}, sigma = {sigma}, alpha = 0.05")
    print("="*70)

    
    theoretical = norm.rvs(loc=mu, scale=sigma, size=N, random_state=rng)

    
    U = rng.random(N)
    
    eps = 1e-12
    U = np.clip(U, eps, 1 - eps)
    empirical = norm.ppf(U, loc=mu, scale=sigma)

    
    k = max(10, min(40, N // 200))
    chi2_stat, chi2_p, chi2_df, edges, contingency = chi2_homogeneity_on_quantile_bins(
        theoretical, empirical, mu, sigma, k
    )

    
    ks_stat, ks_p = ks_2samp(theoretical, empirical, alternative="two-sided", mode="auto")

    
    print("\nCHI-CUADRADO (homogeneidad)")
    print(f"Bins equiprobables: {k}")
    print(f"Estadístico χ² = {chi2_stat:.4f}, gl = {chi2_df}, p-value = {chi2_p:.4f}")
    print("Conclusión χ²:", "RECHAZAMOS que provienen de la misma distribución." if chi2_p < 0.05
          else "NO rechazamos que provienen de la misma distribución.")

    print("\nKOLMOGOROV–SMIRNOV (dos muestras)")
    print(f"Estadístico D = {ks_stat:.4f}, p-value = {ks_p:.4f}")
    print("Conclusión KS:", "RECHAZAMOS que provienen de la misma distribución." if ks_p < 0.05
          else "NO rechazamos que provienen de la misma distribución.")

    # Figuras (histogramas y QQ)
    if not args.no_plot:
        # Hist overlay
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(theoretical, bins=40, stat="density", alpha=0.5, label="Teórica (scipy)")
        sns.histplot(empirical,   bins=40, stat="density", alpha=0.5, label="Empírica (PPF)")
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 400)
        ax.plot(x, norm.pdf(x, mu, sigma), linewidth=1.5, label="Densidad N(mu, sigma)")
        ax.set_title(f"Normal(mu={mu}, sigma={sigma}) — Histogramas")
        ax.legend()
        fig.tight_layout()
        fig_path = outdir / f"hist_N{N}_mu{mu}_sigma{sigma}.png"
        plt.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"\nFigura de histogramas: {fig_path}")

        # Q-Q plot teórica vs empírica
        q = np.linspace(0.01, 0.99, 200)
        q_the = np.quantile(theoretical, q)
        q_emp = np.quantile(empirical, q)
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        ax2.scatter(q_the, q_emp, s=10, alpha=0.7)
        lims = [min(q_the.min(), q_emp.min()), max(q_the.max(), q_emp.max())]
        ax2.plot(lims, lims, linestyle="--")
        ax2.set_title("Q-Q plot: teórica vs empírica")
        ax2.set_xlabel("Cuantiles muestra teórica")
        ax2.set_ylabel("Cuantiles muestra empírica")
        ax2.set_aspect("equal", "box")
        fig2.tight_layout()
        qq_path = outdir / f"qq_N{N}_mu{mu}_sigma{sigma}.png"
        plt.savefig(qq_path, dpi=150)
        plt.close(fig2)
        print(f"Figura Q-Q: {qq_path}")

if __name__ == "__main__":
    main()
