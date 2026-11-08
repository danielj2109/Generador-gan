"""
visualizar.py
=============
Funciones para generar todas las gráficas e imágenes del proyecto.
Todas reciben el texto de la categoría para que quede ESPECIFICADA en cada figura.
"""

import matplotlib
matplotlib.use("Agg")  # Dibuja en archivos, sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid


def a_imagen(t: torch.Tensor) -> torch.Tensor:
    """Convierte píxeles de [-1, 1] (salida de Tanh) a [0, 1] para dibujar."""
    return (t.detach().cpu() + 1) / 2


def _cuadricula_numpy(imagenes: torch.Tensor, nfilas: int = 8) -> np.ndarray:
    rejilla = make_grid(a_imagen(imagenes), nrow=nfilas, padding=2, pad_value=1.0)
    return rejilla.permute(1, 2, 0).numpy()


# ---------------------------------------------------------------------------
# Cuadrícula de imágenes (se usa en cada época)
# ---------------------------------------------------------------------------
def guardar_cuadricula(imagenes, ruta, titulo, nfilas=8):
    plt.figure(figsize=(6, 6.4))
    plt.imshow(_cuadricula_numpy(imagenes, nfilas), cmap="gray")
    plt.title(titulo, fontsize=11)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(ruta, dpi=110)
    plt.close()


# ---------------------------------------------------------------------------
# FUNCIONES DE PÉRDIDA
# ---------------------------------------------------------------------------
def graficar_perdidas(h, ruta, texto_cat):
    fig, ejes = plt.subplots(1, 2, figsize=(14, 5))

    # Izquierda: pérdida de G y D por iteración (+ media por época)
    ax = ejes[0]
    ax.plot(h["iter_loss_D"], color="tab:red", alpha=0.25, lw=0.8)
    ax.plot(h["iter_loss_G"], color="tab:blue", alpha=0.25, lw=0.8)
    iters_por_epoca = len(h["iter_loss_D"]) / len(h["epoca"])
    x_epoca = [(e - 0.5) * iters_por_epoca for e in h["epoca"]]
    ax.plot(x_epoca, h["loss_D"], "o-", color="tab:red", label="Pérdida Discriminador  L_D")
    ax.plot(x_epoca, h["loss_G"], "o-", color="tab:blue", label="Pérdida Generador  L_G")
    ax.axhline(2 * np.log(2), ls="--", color="tab:red", alpha=0.5, lw=1)
    ax.axhline(np.log(2), ls="--", color="tab:blue", alpha=0.5, lw=1)
    ax.text(0, 2 * np.log(2) + 0.03, "equilibrio L_D = 2·ln2 ≈ 1.386", color="tab:red", fontsize=8)
    ax.text(0, np.log(2) + 0.03, "equilibrio L_G = ln2 ≈ 0.693", color="tab:blue", fontsize=8)
    ax.set_xlabel("Iteración (lote)")
    ax.set_ylabel("Binary Cross-Entropy")
    ax.set_title("Funciones de pérdida (línea fina = cada lote, puntos = media por época)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Derecha: descomposición de la pérdida de D en parte real + parte falsa
    ax = ejes[1]
    ax.plot(h["epoca"], h["loss_D_real"], "o-", color="tab:green",
            label="L_D real  = −log D(x)")
    ax.plot(h["epoca"], h["loss_D_falso"], "o-", color="tab:orange",
            label="L_D falso = −log(1 − D(G(z)))")
    ax.plot(h["epoca"], h["loss_D"], "--", color="tab:red", label="L_D total = real + falso")
    ax.set_xlabel("Época")
    ax.set_ylabel("Binary Cross-Entropy")
    ax.set_title("Pérdida del discriminador separada en sus dos partes")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(f"Funciones de pérdida de la GAN — {texto_cat}", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# VALORES DEL DISCRIMINADOR: D(x) y D(G(z))
# ---------------------------------------------------------------------------
def graficar_valores_D(h, ruta, texto_cat):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(h["epoca"], h["D_x"], "o-", color="tab:green",
            label="D(x): prob. que D asigna a imágenes REALES")
    ax.plot(h["epoca"], h["D_G_z1"], "o-", color="tab:orange",
            label="D(G(z)) antes de actualizar G")
    ax.plot(h["epoca"], h["D_G_z2"], "s--", color="tab:purple",
            label="D(G(z)) después de actualizar G")
    ax.axhline(0.5, ls=":", color="gray")
    ax.text(h["epoca"][0], 0.52, "0.5 = D ya no distingue (equilibrio ideal)", color="gray", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Época")
    ax.set_ylabel("Salida del discriminador (probabilidad de 'real')")
    ax.set_title(f"Valores del discriminador y del generador — {texto_cat}")
    ax.legend(loc="center right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Evolución del generador a lo largo del entrenamiento
# ---------------------------------------------------------------------------
def figura_evolucion(instantaneas, ruta, texto_cat):
    """instantaneas = lista de (epoca, tensor de imágenes)"""
    n = len(instantaneas)
    fig, ejes = plt.subplots(1, n, figsize=(3.2 * n, 3.6))
    for ax, (epoca, imgs) in zip(ejes, instantaneas):
        ax.imshow(_cuadricula_numpy(imgs[:16], nfilas=4), cmap="gray")
        ax.set_title("Inicio (sin entrenar)" if epoca == 0 else f"Época {epoca}")
        ax.axis("off")
    fig.suptitle(f"Evolución del Generador (mismo ruido z fijo) — {texto_cat}",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# RESULTADO FINAL: reales vs generadas
# ---------------------------------------------------------------------------
def figura_resultado_final(reales, falsas, ruta, texto_cat):
    fig, ejes = plt.subplots(1, 2, figsize=(12, 6.4))
    ejes[0].imshow(_cuadricula_numpy(reales[:64]), cmap="gray")
    ejes[0].set_title("Imágenes REALES (dataset)", fontsize=12)
    ejes[1].imshow(_cuadricula_numpy(falsas[:64]), cmap="gray")
    ejes[1].set_title("Imágenes GENERADAS por la GAN", fontsize=12)
    for ax in ejes:
        ax.axis("off")
    fig.suptitle(f"Resultado de la GAN — Categoría usada: {texto_cat}",
                 fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Veredicto del discriminador imagen por imagen
# ---------------------------------------------------------------------------
def figura_veredicto(reales, falsas, p_reales, p_falsas, ruta, texto_cat,
                     media_real=None, media_falsa=None):
    n = len(reales)
    fig, ejes = plt.subplots(2, n, figsize=(1.6 * n, 4.5))
    for i in range(n):
        for fila, (imgs, probs, etiqueta) in enumerate(
                [(reales, p_reales, "REAL"), (falsas, p_falsas, "GENERADA")]):
            ax = ejes[fila, i]
            ax.imshow(a_imagen(imgs[i])[0], cmap="gray")
            p = probs[i]
            ax.set_title(f"D = {p:.2f}", fontsize=10,
                         color="tab:green" if p >= 0.5 else "tab:red")
            ax.axis("off")
            if i == 0:
                ax.text(-6, 14, etiqueta, rotation=90, va="center", ha="center",
                        fontsize=10, weight="bold")
    titulo = f"¿Qué opina el Discriminador? D = prob. de ser real (verde ≥ 0.5) — {texto_cat}"
    if media_real is not None:
        titulo += (f"\nMedia en 64 imágenes:  D(x) reales = {media_real:.2f}   |   "
                   f"D(G(z)) generadas = {media_falsa:.2f}")
    fig.suptitle(titulo, fontsize=11, weight="bold")
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Histogramas de los PESOS de cada capa: inicio vs final
# ---------------------------------------------------------------------------
def histogramas_pesos(pesos_inicio, pesos_final, ruta, titulo):
    """pesos_* = dict {nombre_parametro: tensor}. Solo se dibujan matrices de pesos (weight)."""
    nombres = [n for n in pesos_final if n.endswith("weight") and pesos_final[n].dim() > 1]
    fig, ejes = plt.subplots(1, len(nombres), figsize=(4.2 * len(nombres), 3.6))
    if len(nombres) == 1:
        ejes = [ejes]
    for ax, n in zip(ejes, nombres):
        ini = pesos_inicio[n].flatten().numpy()
        fin = pesos_final[n].flatten().numpy()
        rango = (min(ini.min(), fin.min()), max(ini.max(), fin.max()))
        ax.hist(ini, bins=80, range=rango, alpha=0.55, color="gray", label="Inicio", density=True)
        ax.hist(fin, bins=80, range=rango, alpha=0.55, color="tab:blue", label="Final", density=True)
        ax.set_title(f"{n}\n({fin.size:,} pesos)", fontsize=9)
        ax.legend(fontsize=8)
    fig.suptitle(titulo, fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)
