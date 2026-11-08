"""
modelos.py
==========
AQUÍ SE CREAN LAS DOS REDES NEURONALES DE LA GAN:

    1. class Generador      -> Red neuronal GENERADORA     (G)
    2. class Discriminador  -> Red neuronal DISCRIMINADORA (D)

Arquitectura tipo DCGAN (Deep Convolutional GAN) reducida para imágenes 28x28.

Idea general de una GAN:
    - G recibe ruido aleatorio z y "dibuja" una imagen falsa G(z).
    - D recibe una imagen y devuelve la probabilidad de que sea REAL.
    - Compiten: D intenta no ser engañado, G intenta engañar a D.
"""

import torch
import torch.nn as nn

import config


# ===========================================================================
# 1) RED NEURONAL GENERADORA  (G)
# ===========================================================================
class Generador(nn.Module):
    """
    ENTRADA : vector de ruido z ~ N(0, 1) de tamaño DIM_LATENTE (100).
    SALIDA  : imagen falsa de 1 x 28 x 28 con píxeles en el rango [-1, 1].
    OBJETIVO: producir imágenes tan realistas que D las clasifique como reales.

    Flujo de formas (con FILTROS_G = 64):
        z (100)
          -> Linear            -> 6272 valores  (=128 * 7 * 7)
          -> reshape           -> 128 x 7 x 7
          -> ConvTranspose2d   -> 64 x 14 x 14   (duplica el tamaño)
          -> ConvTranspose2d   -> 1 x 28 x 28    (duplica el tamaño)
          -> Tanh              -> píxeles en [-1, 1]
    """

    def __init__(self,
                 dim_latente: int = config.DIM_LATENTE,
                 filtros: int = config.FILTROS_G,
                 canales: int = config.CANALES_IMAGEN):
        super().__init__()
        self.dim_latente = dim_latente
        self.filtros = filtros

        # --- Bloque 1: PROYECCIÓN del ruido a un pequeño mapa de 7x7 ---------
        # Linear: cada uno de los 100 números de z se conecta con 6272 salidas.
        #   Parámetros = 100 * 6272 = 627.200 pesos (sin bias, BatchNorm lo sustituye)
        self.proyeccion = nn.Sequential(
            nn.Linear(dim_latente, filtros * 2 * 7 * 7, bias=False),
            # BatchNorm normaliza las activaciones -> entrenamiento más estable.
            #   Parámetros = 2 * 6272 (gamma y beta)
            nn.BatchNorm1d(filtros * 2 * 7 * 7),
            # ReLU: activación no lineal (recomendada en G por DCGAN).
            nn.ReLU(inplace=True),
        )

        # --- Bloque 2: CONVOLUCIONES TRANSPUESTAS (agrandan la imagen) -------
        # kernel_size=4, stride=2, padding=1  => duplica alto y ancho.
        self.convoluciones = nn.Sequential(
            # 128 x 7 x 7  ->  64 x 14 x 14
            #   Parámetros = 128 * 64 * 4 * 4 = 131.072
            nn.ConvTranspose2d(filtros * 2, filtros, kernel_size=4, stride=2,
                               padding=1, bias=False),
            nn.BatchNorm2d(filtros),
            nn.ReLU(inplace=True),

            # 64 x 14 x 14  ->  1 x 28 x 28   (la imagen final)
            #   Parámetros = 64 * 1 * 4 * 4 = 1.024
            nn.ConvTranspose2d(filtros, canales, kernel_size=4, stride=2,
                               padding=1, bias=False),
            # Tanh deja los píxeles en [-1, 1], igual que las imágenes reales
            # normalizadas en datos.py.
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.proyeccion(z)                      # (lote, 6272)
        x = x.view(-1, self.filtros * 2, 7, 7)      # (lote, 128, 7, 7)
        return self.convoluciones(x)                # (lote, 1, 28, 28)


# ===========================================================================
# 2) RED NEURONAL DISCRIMINADORA  (D)
# ===========================================================================
class Discriminador(nn.Module):
    """
    ENTRADA : imagen de 1 x 28 x 28 (real del dataset o falsa del generador).
    SALIDA  : un número entre 0 y 1 = probabilidad de que la imagen sea REAL.
              ~1 -> "creo que es real"     ~0 -> "creo que es falsa"
    OBJETIVO: distinguir correctamente imágenes reales de falsas.

    Flujo de formas (con FILTROS_D = 64):
        imagen 1 x 28 x 28
          -> Conv2d     -> 64 x 14 x 14   (reduce a la mitad)
          -> Conv2d     -> 128 x 7 x 7    (reduce a la mitad)
          -> Flatten    -> 6272 valores
          -> Linear     -> 1 valor
          -> Sigmoid    -> probabilidad en [0, 1]
    """

    def __init__(self,
                 filtros: int = config.FILTROS_D,
                 canales: int = config.CANALES_IMAGEN,
                 pendiente_leaky: float = config.PENDIENTE_LEAKY):
        super().__init__()

        self.red = nn.Sequential(
            # --- Conv 1: 1 x 28 x 28 -> 64 x 14 x 14 -------------------------
            #   Parámetros = 1 * 64 * 4 * 4 = 1.024
            #   (DCGAN no usa BatchNorm en la primera capa de D)
            nn.Conv2d(canales, filtros, kernel_size=4, stride=2, padding=1, bias=False),
            # LeakyReLU deja pasar un poco de gradiente negativo (pendiente 0.2).
            nn.LeakyReLU(pendiente_leaky, inplace=True),

            # --- Conv 2: 64 x 14 x 14 -> 128 x 7 x 7 -------------------------
            #   Parámetros = 64 * 128 * 4 * 4 = 131.072
            nn.Conv2d(filtros, filtros * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(filtros * 2),
            nn.LeakyReLU(pendiente_leaky, inplace=True),

            # --- Clasificador final -----------------------------------------
            nn.Flatten(),                         # 128*7*7 = 6272 valores
            #   Parámetros = 6272 pesos + 1 bias = 6.273
            nn.Linear(filtros * 2 * 7 * 7, 1),
            # Sigmoid convierte el valor en una probabilidad entre 0 y 1.
            nn.Sigmoid(),
        )

    def forward(self, imagen: torch.Tensor) -> torch.Tensor:
        return self.red(imagen)                   # (lote, 1)


# ===========================================================================
# 3) INICIALIZACIÓN DE PESOS (recomendación del paper DCGAN)
# ===========================================================================
def inicializar_pesos(modulo: nn.Module) -> None:
    """
    - Capas Conv / ConvTranspose / Linear: pesos ~ Normal(media=0, desv=0.02)
    - Capas BatchNorm: gamma ~ Normal(1, 0.02), beta = 0
    Se aplica con:  modelo.apply(inicializar_pesos)
    """
    if isinstance(modulo, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(modulo.weight, 0.0, 0.02)
        if modulo.bias is not None:
            nn.init.zeros_(modulo.bias)
    elif isinstance(modulo, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.normal_(modulo.weight, 1.0, 0.02)
        nn.init.zeros_(modulo.bias)


# ===========================================================================
# 4) HERRAMIENTAS PARA "VER" LOS PARÁMETROS DE CADA RED
# ===========================================================================
def resumen_capas(modelo: nn.Module, entrada_ejemplo: torch.Tensor) -> str:
    """
    Devuelve una tabla en texto con cada capa: tipo, forma de salida y nº de
    parámetros entrenables. Se obtiene pasando una entrada de ejemplo por la red.
    """
    filas = []
    ganchos = []

    def registrar(nombre):
        def gancho(mod, _entrada, salida):
            n_param = sum(p.numel() for p in mod.parameters(recurse=False))
            filas.append((nombre, mod.__class__.__name__, tuple(salida.shape[1:]), n_param))
        return gancho

    # Registramos un "gancho" en cada capa hoja (las que no contienen otras capas).
    for nombre, mod in modelo.named_modules():
        if len(list(mod.children())) == 0:
            ganchos.append(mod.register_forward_hook(registrar(nombre)))

    estaba_entrenando = modelo.training
    modelo.eval()
    with torch.no_grad():
        modelo(entrada_ejemplo)
    modelo.train(estaba_entrenando)
    for g in ganchos:
        g.remove()

    lineas = [f"{'Capa':<18}{'Tipo':<18}{'Forma de salida':<20}{'Parámetros':>12}",
              "-" * 68]
    for nombre, tipo, forma, n in filas:
        lineas.append(f"{nombre:<18}{tipo:<18}{str(forma):<20}{n:>12,}")
    total = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    lineas.append("-" * 68)
    lineas.append(f"{'TOTAL parámetros entrenables':<56}{total:>12,}")
    return "\n".join(lineas)


def estadisticas_pesos(modelo: nn.Module) -> str:
    """
    Muestra los VALORES de los pesos de cada capa resumidos en estadísticas:
    media, desviación estándar, mínimo y máximo.
    """
    lineas = [f"{'Parámetro':<28}{'Cantidad':>10}{'Media':>11}{'Desv.':>10}{'Mín':>10}{'Máx':>10}",
              "-" * 79]
    for nombre, p in modelo.named_parameters():
        d = p.detach().float()
        desv = d.std().item() if d.numel() > 1 else 0.0  # un solo valor (p. ej. bias) no tiene desviación
        lineas.append(f"{nombre:<28}{d.numel():>10,}{d.mean().item():>11.4f}"
                      f"{desv:>10.4f}{d.min().item():>10.4f}{d.max().item():>10.4f}")
    return "\n".join(lineas)


# ---------------------------------------------------------------------------
# Prueba rápida:  python modelos.py   -> imprime la estructura de ambas redes
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    G = Generador()
    D = Discriminador()
    print("=== RED GENERADORA ===")
    print(G)
    print(resumen_capas(G, torch.randn(2, config.DIM_LATENTE)))
    print("\n=== RED DISCRIMINADORA ===")
    print(D)
    print(resumen_capas(D, torch.randn(2, 1, 28, 28)))
