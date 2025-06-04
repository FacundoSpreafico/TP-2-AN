import argparse
import numpy as np
from PIL import Image

from ej3_functions import cargar_imagen_escala_grises, extraer_mensaje


def calcular_indices(imagen: np.ndarray, total_bits: int, redundancia: int):
    """Calcula las posiciones de los coeficientes usados para el mensaje."""
    fft = np.fft.fft2(imagen.astype(np.float32))
    fft_shift = np.fft.fftshift(fft)
    magnitudes = np.abs(fft_shift).flatten()
    h, w = imagen.shape
    indices_planos = np.arange(h * w)
    dc_plano = (h // 2) * w + (w // 2)
    magnitudes[dc_plano] = -1  # Excluir componente DC

    indices_ordenados = indices_planos[np.argsort(-magnitudes)]
    necesarios = total_bits * redundancia
    if necesarios > h * w - 1:
        raise RuntimeError(
            f"Redundancia excesiva. Se necesitan {necesarios} coeficientes."
        )

    seleccionados = indices_ordenados[:necesarios]
    return [(idx // w, idx % w) for idx in seleccionados]


def decodificar(stego_path: str, delta: float, secret_shape, redundancia: int, output_path: str):
    estego = cargar_imagen_escala_grises(stego_path)
    indices = calcular_indices(estego, secret_shape[0] * secret_shape[1], redundancia)
    secreta_rec, _ = extraer_mensaje(
        estego.astype(np.float32), delta, secret_shape, indices, redundancia
    )
    Image.fromarray(secreta_rec.astype(np.uint8)).save(output_path)
    print(f"[INFO] Imagen secreta recuperada guardada en: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Decodificador Ejercicio 3")
    parser.add_argument("stego_image", help="Ruta de la imagen estego en escala de grises")
    parser.add_argument("-o", "--output", default="secreta_decodificada.png", help="Ruta de salida de la imagen secreta")
    parser.add_argument("-d", "--delta", type=float, default=1.0, help="Delta usado durante la codificaci\u00f3n")
    parser.add_argument("-r", "--redundancia", type=int, default=10, help="Redundancia utilizada")
    parser.add_argument("-s", "--secret-size", type=int, default=256, help="Tama\u00f1o del lado de la imagen secreta")
    args = parser.parse_args()

    secret_shape = (args.secret_size, args.secret_size)
    decodificar(args.stego_image, args.delta, secret_shape, args.redundancia, args.output)


if __name__ == "__main__":
    main()
