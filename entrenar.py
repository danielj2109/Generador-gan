"""
entrenar.py
===========
Script principal. Entrena la GAN y genera todos los resultados en resultados/.

Uso:
    python entrenar.py                       # dígito 7, 30 épocas (config.py)
    python entrenar.py --categoria 3         # entrena con el dígito 3
    python entrenar.py --epocas 50

Pasos del script:
    1. Cargar los datos de UNA categoría.
    2. CREAR la red Generadora y la Discriminadora (instancias de modelos.py).
    3. Definir la FUNCIÓN DE PÉRDIDA y los optimizadores.
    4. Bucle de entrenamiento (alternando D y G) mostrando sus valores.
    5. Guardar gráficas, imágenes, parámetros y modelos.
"""

import argparse
import copy
import csv
import json
import os
import time

import torch
import torch.nn as nn

import config
from datos import cargar_categoria
from modelos import (Generador, Discriminador, inicializar_pesos,
                     resumen_capas, estadisticas_pesos)
import visualizar as vis


def copiar_pesos(modelo):
    """Copia de los pesos actuales (para comparar inicio vs final)."""
    return {n: p.detach().cpu().clone() for n, p in modelo.named_parameters()}


def main():
    # -----------------------------------------------------------------------
    # 0) Argumentos y preparación
    # -----------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Entrenar una GAN sobre una categoría de MNIST")
    parser.add_argument("--categoria", type=int, default=config.CATEGORIA, choices=range(10))
    parser.add_argument("--epocas", type=int, default=config.EPOCAS)
    parser.add_argument("--lote", type=int, default=config.TAM_LOTE)
    args = parser.parse_args()

    torch.manual_seed(config.SEMILLA)
    dispositivo = config.DISPOSITIVO
    categoria = args.categoria
    texto_cat = f"{config.DATASET}, dígito {categoria} ('{config.NOMBRES_CATEGORIAS[categoria]}')"

    salida = config.CARPETA_RESULTADOS
    os.makedirs(os.path.join(salida, "evolucion"), exist_ok=True)
    os.makedirs(os.path.join(salida, "modelos"), exist_ok=True)

    print("=" * 70)
    print(f" GAN — Categoría usada: {texto_cat}")
    print(f" Dispositivo: {dispositivo} | Épocas: {args.epocas} | Lote: {args.lote}")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 1) DATOS: solo imágenes reales de la categoría elegida
    # -----------------------------------------------------------------------
    cargador = cargar_categoria(categoria, args.lote)

    # -----------------------------------------------------------------------
    # 2) >>> AQUÍ SE CREA LA RED NEURONAL GENERADORA (G) <<<
    #    La clase está definida en modelos.py -> class Generador
    # -----------------------------------------------------------------------
    G = Generador(dim_latente=config.DIM_LATENTE, filtros=config.FILTROS_G).to(dispositivo)
    G.apply(inicializar_pesos)

    # -----------------------------------------------------------------------
    # 3) >>> AQUÍ SE CREA LA RED NEURONAL DISCRIMINADORA (D) <<<
    #    La clase está definida en modelos.py -> class Discriminador
    # -----------------------------------------------------------------------
    D = Discriminador(filtros=config.FILTROS_D,
                      pendiente_leaky=config.PENDIENTE_LEAKY).to(dispositivo)
    D.apply(inicializar_pesos)

    # Mostrar y guardar los PARÁMETROS de cada red
    tabla_G = resumen_capas(G, torch.randn(2, config.DIM_LATENTE, device=dispositivo))
    tabla_D = resumen_capas(D, torch.randn(2, 1, 28, 28, device=dispositivo))
    print("\n--- PARÁMETROS DE LA RED GENERADORA ---\n" + tabla_G)
    print("\n--- PARÁMETROS DE LA RED DISCRIMINADORA ---\n" + tabla_D + "\n")

    pesos_ini_G, pesos_ini_D = copiar_pesos(G), copiar_pesos(D)
    stats_ini_G, stats_ini_D = estadisticas_pesos(G), estadisticas_pesos(D)

    # -----------------------------------------------------------------------
    # 4) FUNCIÓN DE PÉRDIDA: Binary Cross-Entropy (BCE)
    #
    #    BCE(p, y) = −[ y·log(p) + (1 − y)·log(1 − p) ]
    #
    #    • Pérdida del DISCRIMINADOR (quiere D(x)→1 y D(G(z))→0):
    #         L_D = BCE(D(x), 1) + BCE(D(G(z)), 0)
    #             = −log D(x)  −  log(1 − D(G(z)))
    #
    #    • Pérdida del GENERADOR (versión "no saturante", quiere D(G(z))→1):
    #         L_G = BCE(D(G(z)), 1) = −log D(G(z))
    #
    #    Equilibrio teórico (D(x) = D(G(z)) = 0.5):
    #         L_D = 2·ln2 ≈ 1.386      L_G = ln2 ≈ 0.693
    # -----------------------------------------------------------------------
    criterio = nn.BCELoss()

    # Optimizadores Adam (uno para cada red; cada uno solo mueve SUS pesos)
    opt_G = torch.optim.Adam(G.parameters(), lr=config.LR_G, betas=(config.BETA1, config.BETA2))
    opt_D = torch.optim.Adam(D.parameters(), lr=config.LR_D, betas=(config.BETA1, config.BETA2))

    # Ruido FIJO: siempre las mismas 64 "semillas" para ver cómo mejora G
    ruido_fijo = torch.randn(64, config.DIM_LATENTE, device=dispositivo)

    # Historial de valores para gráficas y CSV
    h = {k: [] for k in ["epoca", "loss_D", "loss_D_real", "loss_D_falso", "loss_G",
                         "D_x", "D_G_z1", "D_G_z2", "iter_loss_D", "iter_loss_G"]}

    # Imagen de G antes de entrenar (época 0)
    with torch.no_grad():
        G.eval()
        instantaneas = [(0, G(ruido_fijo).cpu())]
        G.train()

    # Épocas para la figura de evolución (≈ inicio, 25%, 50%, 75%, final)
    epocas_clave = sorted({1, max(1, args.epocas // 4), max(1, args.epocas // 2),
                           max(1, 3 * args.epocas // 4), args.epocas})

    print(f"{'Época':>6} | {'Loss_D':>7} {'(real':>7} {'+falso)':>8} | {'Loss_G':>7} | "
          f"{'D(x)':>6} | {'D(G(z)) antes→después':>22} | tiempo")
    print("-" * 92)

    # -----------------------------------------------------------------------
    # 5) BUCLE DE ENTRENAMIENTO
    # -----------------------------------------------------------------------
    for epoca in range(1, args.epocas + 1):
        t0 = time.time()
        acum = {k: 0.0 for k in ["loss_D", "loss_D_real", "loss_D_falso", "loss_G",
                                 "D_x", "D_G_z1", "D_G_z2"]}
        n_lotes = 0

        for reales, _ in cargador:
            reales = reales.to(dispositivo)
            b = reales.size(0)
            etiq_real = torch.full((b, 1), config.ETIQUETA_REAL, device=dispositivo)
            etiq_falsa = torch.full((b, 1), config.ETIQUETA_FALSA, device=dispositivo)

            # ===============================================================
            # PASO A: entrenar el DISCRIMINADOR
            #         maximizar log D(x) + log(1 − D(G(z)))
            # ===============================================================
            opt_D.zero_grad()

            # A.1 imágenes REALES -> D debería decir 1
            salida_real = D(reales)
            loss_D_real = criterio(salida_real, etiq_real)          # −log D(x)

            # A.2 imágenes FALSAS -> D debería decir 0
            z = torch.randn(b, config.DIM_LATENTE, device=dispositivo)
            falsas = G(z)
            # .detach(): no propagamos el gradiente hacia G en este paso
            salida_falsa = D(falsas.detach())
            loss_D_falso = criterio(salida_falsa, etiq_falsa)       # −log(1 − D(G(z)))

            loss_D = loss_D_real + loss_D_falso
            loss_D.backward()
            opt_D.step()

            # ===============================================================
            # PASO B: entrenar el GENERADOR
            #         maximizar log D(G(z))  (= engañar a D)
            # ===============================================================
            opt_G.zero_grad()
            # Volvemos a pasar las falsas por D (ya actualizado) SIN detach,
            # y usamos etiqueta REAL: G gana si D cree que son reales.
            salida_falsa_G = D(falsas)
            loss_G = criterio(salida_falsa_G, etiq_real)            # −log D(G(z))
            loss_G.backward()
            opt_G.step()

            # --- VALORES del generador y discriminador en este lote ---------
            acum["loss_D"] += loss_D.item()
            acum["loss_D_real"] += loss_D_real.item()
            acum["loss_D_falso"] += loss_D_falso.item()
            acum["loss_G"] += loss_G.item()
            acum["D_x"] += salida_real.mean().item()        # media de D(x)
            acum["D_G_z1"] += salida_falsa.mean().item()    # D(G(z)) antes de mover G
            acum["D_G_z2"] += salida_falsa_G.mean().item()  # D(G(z)) usado para mover G
            h["iter_loss_D"].append(loss_D.item())
            h["iter_loss_G"].append(loss_G.item())
            n_lotes += 1

        # Medias de la época
        h["epoca"].append(epoca)
        for k, v in acum.items():
            h[k].append(v / n_lotes)

        print(f"{epoca:>6} | {h['loss_D'][-1]:>7.4f} {h['loss_D_real'][-1]:>7.4f} "
              f"{h['loss_D_falso'][-1]:>8.4f} | {h['loss_G'][-1]:>7.4f} | "
              f"{h['D_x'][-1]:>6.3f} | {h['D_G_z1'][-1]:>10.3f} → {h['D_G_z2'][-1]:<9.3f} | "
              f"{time.time() - t0:5.1f}s")

        # Imágenes generadas con el ruido fijo en esta época
        G.eval()
        with torch.no_grad():
            muestras = G(ruido_fijo).cpu()
        G.train()
        vis.guardar_cuadricula(muestras, os.path.join(salida, "evolucion", f"epoca_{epoca:03d}.png"),
                               f"Época {epoca} — {texto_cat}")
        if epoca in epocas_clave:
            instantaneas.append((epoca, muestras))

    # -----------------------------------------------------------------------
    # 6) RESULTADOS
    # -----------------------------------------------------------------------
    print("\nGuardando resultados en la carpeta 'resultados/' ...")

    # 6.1 Funciones de pérdida y valores de D
    vis.graficar_perdidas(h, os.path.join(salida, "perdidas.png"), texto_cat)
    vis.graficar_valores_D(h, os.path.join(salida, "valores_discriminador.png"), texto_cat)
    vis.figura_evolucion(instantaneas, os.path.join(salida, "evolucion_generador.png"), texto_cat)

    # 6.2 Resultado final: reales vs generadas
    G.eval()
    reales_muestra = next(iter(cargador))[0][:64]
    # Para "preguntarle" a D usamos una COPIA en modo train(): así BatchNorm usa
    # las estadísticas del lote, igual que durante el entrenamiento, y los valores
    # coinciden con los D(x) y D(G(z)) impresos en la tabla. (Con D.eval() BatchNorm
    # usa promedios acumulados de lotes reales y falsos mezclados y los valores se distorsionan.)
    D_juez = copy.deepcopy(D).train()
    with torch.no_grad():
        falsas_muestra = G(torch.randn(64, config.DIM_LATENTE, device=dispositivo)).cpu()
        todas_reales = D_juez(reales_muestra.to(dispositivo)).squeeze(1).cpu()
        todas_falsas = D_juez(falsas_muestra.to(dispositivo)).squeeze(1).cpu()
    media_real, media_falsa = todas_reales.mean().item(), todas_falsas.mean().item()
    p_reales, p_falsas = todas_reales[:10].tolist(), todas_falsas[:10].tolist()
    vis.figura_resultado_final(reales_muestra, falsas_muestra,
                               os.path.join(salida, "resultado_final.png"), texto_cat)
    vis.guardar_cuadricula(falsas_muestra, os.path.join(salida, "generadas_final.png"),
                           f"64 imágenes generadas — {texto_cat}")
    vis.figura_veredicto(reales_muestra[:10], falsas_muestra[:10], p_reales, p_falsas,
                         os.path.join(salida, "veredicto_discriminador.png"), texto_cat,
                         media_real, media_falsa)

    # 6.3 Pesos: histogramas inicio vs final
    pesos_fin_G, pesos_fin_D = copiar_pesos(G), copiar_pesos(D)
    vis.histogramas_pesos(pesos_ini_G, pesos_fin_G, os.path.join(salida, "pesos_generador.png"),
                          f"Pesos de la red GENERADORA: inicio vs final — {texto_cat}")
    vis.histogramas_pesos(pesos_ini_D, pesos_fin_D, os.path.join(salida, "pesos_discriminador.png"),
                          f"Pesos de la red DISCRIMINADORA: inicio vs final — {texto_cat}")

    # 6.4 Reporte de parámetros en texto
    with open(os.path.join(salida, "parametros_redes.txt"), "w", encoding="utf-8") as f:
        f.write(f"GAN — Categoría usada: {texto_cat}\n")
        f.write(f"Épocas: {args.epocas} | Lote: {args.lote} | Ruido z: {config.DIM_LATENTE} | "
                f"LR_G: {config.LR_G} | LR_D: {config.LR_D} | Adam betas: "
                f"({config.BETA1}, {config.BETA2})\n\n")
        f.write("=" * 80 + "\nRED NEURONAL GENERADORA (G)\n" + "=" * 80 + "\n")
        f.write(str(G) + "\n\n" + tabla_G + "\n\n")
        f.write("Valores de los pesos AL INICIO:\n" + stats_ini_G + "\n\n")
        f.write("Valores de los pesos AL FINAL:\n" + estadisticas_pesos(G) + "\n\n")
        f.write("=" * 80 + "\nRED NEURONAL DISCRIMINADORA (D)\n" + "=" * 80 + "\n")
        f.write(str(D) + "\n\n" + tabla_D + "\n\n")
        f.write("Valores de los pesos AL INICIO:\n" + stats_ini_D + "\n\n")
        f.write("Valores de los pesos AL FINAL:\n" + estadisticas_pesos(D) + "\n\n")
        f.write("=" * 80 + "\nVEREDICTO DEL DISCRIMINADOR AL FINAL (10 ejemplos)\n" + "=" * 80 + "\n")
        f.write("D(x) reales   : " + ", ".join(f"{p:.3f}" for p in p_reales) + "\n")
        f.write("D(G(z)) falsas: " + ", ".join(f"{p:.3f}" for p in p_falsas) + "\n")
        f.write(f"Media sobre 64 imágenes -> D(x) reales: {media_real:.3f} | "
                f"D(G(z)) generadas: {media_falsa:.3f}\n")

    # 6.5 Historial por época (CSV) y resumen (JSON)
    claves = ["epoca", "loss_D", "loss_D_real", "loss_D_falso", "loss_G", "D_x", "D_G_z1", "D_G_z2"]
    with open(os.path.join(salida, "historial.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(claves)
        for i in range(len(h["epoca"])):
            w.writerow([h[k][i] if k == "epoca" else round(h[k][i], 5) for k in claves])
    with open(os.path.join(salida, "resumen.json"), "w", encoding="utf-8") as f:
        json.dump({"dataset": config.DATASET, "categoria": categoria,
                   "nombre_categoria": config.NOMBRES_CATEGORIAS[categoria],
                   "epocas": args.epocas, "imagenes_entrenamiento": len(cargador.dataset),
                   "parametros_G": sum(p.numel() for p in G.parameters()),
                   "parametros_D": sum(p.numel() for p in D.parameters()),
                   "final": {k: round(h[k][-1], 4) for k in claves[1:]}},
                  f, ensure_ascii=False, indent=2)

    # 6.6 Modelos entrenados
    torch.save({"categoria": categoria, "estado": G.state_dict()},
               os.path.join(salida, "modelos", "generador.pt"))
    torch.save({"categoria": categoria, "estado": D.state_dict()},
               os.path.join(salida, "modelos", "discriminador.pt"))

    print(f"Listo. Categoría usada: {texto_cat}")


if __name__ == "__main__":
    main()
