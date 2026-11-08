"""
datos.py
========
Carga MNIST y se queda SOLO con las imágenes de la categoría elegida
(por defecto el dígito 7, ver config.CATEGORIA).
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import config


def cargar_categoria(categoria: int = config.CATEGORIA,
                     tam_lote: int = config.TAM_LOTE) -> DataLoader:
    """
    Devuelve un DataLoader con las imágenes de entrenamiento de MNIST
    que pertenecen a `categoria`.
    """
    # Transformaciones:
    #  - ToTensor: convierte la imagen a tensor con píxeles en [0, 1]
    #  - Normalize(0.5, 0.5): lleva los píxeles a [-1, 1]
    #    (mismo rango que la salida Tanh del generador)
    transformacion = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # Descarga MNIST la primera vez (~11 MB) en la carpeta datos/
    mnist = datasets.MNIST(root=config.CARPETA_DATOS, train=True,
                           download=True, transform=transformacion)

    # FILTRO DE CATEGORÍA: índices de las imágenes cuya etiqueta == categoria
    indices = torch.where(mnist.targets == categoria)[0].tolist()
    subconjunto = Subset(mnist, indices)

    print(f"[datos] Categoría usada: {categoria} ('{config.NOMBRES_CATEGORIAS[categoria]}') "
          f"-> {len(subconjunto)} imágenes reales de entrenamiento")

    # drop_last=True evita un último lote de tamaño 1 (rompe BatchNorm)
    # num_workers=0 para que funcione sin problemas en Windows
    return DataLoader(subconjunto, batch_size=tam_lote, shuffle=True,
                      drop_last=True, num_workers=0)
