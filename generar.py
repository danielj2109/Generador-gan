"""
generar.py
==========
Carga el GENERADOR ya entrenado y crea imágenes nuevas de la categoría usada.
El discriminador NO se necesita aquí: una vez entrenada la GAN, solo se usa G.

Uso:
    python generar.py              # 64 imágenes nuevas
    python generar.py --n 16
"""

import argparse
import os

import torch

import config
from modelos import Generador
import visualizar as vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--salida", default=os.path.join(config.CARPETA_RESULTADOS, "nuevas.png"))
    args = parser.parse_args()

    ruta = os.path.join(config.CARPETA_RESULTADOS, "modelos", "generador.pt")
    punto = torch.load(ruta, map_location=config.DISPOSITIVO)
    categoria = punto["categoria"]

    # Se crea la red GENERADORA y se cargan los pesos aprendidos
    G = Generador().to(config.DISPOSITIVO)
    G.load_state_dict(punto["estado"])
    G.eval()

    with torch.no_grad():
        z = torch.randn(args.n, config.DIM_LATENTE, device=config.DISPOSITIVO)  # ruido nuevo
        imagenes = G(z)

    texto_cat = f"{config.DATASET}, dígito {categoria} ('{config.NOMBRES_CATEGORIAS[categoria]}')"
    vis.guardar_cuadricula(imagenes, args.salida, f"Imágenes nuevas — {texto_cat}")
    print(f"Guardado {args.salida} ({args.n} imágenes de la categoría {categoria})")


if __name__ == "__main__":
    main()
