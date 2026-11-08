"""
config.py
=========
Todos los HIPERPARÁMETROS del proyecto en un solo lugar.

Si quieres experimentar, cambia los valores aquí (o pásalos por línea de
comandos a entrenar.py, por ejemplo:  python entrenar.py --categoria 3).
"""

import torch

# ---------------------------------------------------------------------------
# 1) DATOS Y CATEGORÍA USADA
# ---------------------------------------------------------------------------
# Usamos el dataset MNIST (dígitos escritos a mano, 28x28 píxeles, escala de grises).
# La GAN se entrena SOLO con una categoría (un dígito), así aprende a dibujar
# únicamente ese número.
DATASET = "MNIST"
CATEGORIA = 7  # <-- CATEGORÍA USADA: el dígito 7

# Nombre en texto de cada categoría (para títulos de gráficas y reportes).
NOMBRES_CATEGORIAS = {
    0: "cero", 1: "uno", 2: "dos", 3: "tres", 4: "cuatro",
    5: "cinco", 6: "seis", 7: "siete", 8: "ocho", 9: "nueve",
}

CANALES_IMAGEN = 1   # 1 canal = escala de grises
TAM_IMAGEN = 28      # las imágenes de MNIST miden 28x28

# ---------------------------------------------------------------------------
# 2) PARÁMETROS DE LA RED GENERADORA (G)
# ---------------------------------------------------------------------------
DIM_LATENTE = 100    # Tamaño del vector de ruido z (la "semilla" de cada imagen)
FILTROS_G = 64       # Filtros base del generador (la capa intermedia usa 2x = 128)

# ---------------------------------------------------------------------------
# 3) PARÁMETROS DE LA RED DISCRIMINADORA (D)
# ---------------------------------------------------------------------------
FILTROS_D = 64           # Filtros base del discriminador (la 2ª conv usa 2x = 128)
PENDIENTE_LEAKY = 0.2    # Pendiente negativa de LeakyReLU (evita neuronas "muertas")

# ---------------------------------------------------------------------------
# 4) PARÁMETROS DE ENTRENAMIENTO
# ---------------------------------------------------------------------------
EPOCAS = 30          # Veces que se recorre todo el conjunto de imágenes del dígito
TAM_LOTE = 64        # Imágenes por lote (batch)
LR_G = 2e-4          # Tasa de aprendizaje del generador (valor clásico de DCGAN)
LR_D = 2e-4          # Tasa de aprendizaje del discriminador
BETA1 = 0.5          # Momento del optimizador Adam (0.5 estabiliza las GAN)
BETA2 = 0.999        # Segundo momento de Adam

# Etiquetas que usa la función de pérdida (BCE):
ETIQUETA_REAL = 1.0  # "esta imagen es REAL"
ETIQUETA_FALSA = 0.0 # "esta imagen es FALSA (la hizo G)"

SEMILLA = 42         # Para que los resultados sean reproducibles

# GPU si existe, si no CPU.
DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 5) CARPETAS
# ---------------------------------------------------------------------------
CARPETA_DATOS = "datos"            # Aquí se descarga MNIST
CARPETA_RESULTADOS = "resultados"  # Aquí se guardan gráficas, imágenes y modelos
