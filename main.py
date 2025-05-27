import os
from ej1_functions import (
    mostrar_info_capacidad,
    codificar_lsb,
    decodificar_lsb,
    verificar_decodificacion
)

from ej2_functions import (
    fourier_steganography_encode,
    fourier_steganography_decode,
)

from ej3_functions import (
    codificar_fourier_delta,
    decodificar_fourier_delta
)

if __name__ == "__main__":
    # Configuración común
    directorio_actual = os.path.dirname(os.path.abspath(__file__))

    # Ejemplo de uso para LSB Steganography
    print("\n" + "=" * 50)
    print("DEMOSTRACIÓN LSB STEGANOGRAPHY")
    print("=" * 50)

    imagen_original = os.path.join(directorio_actual, "secreta.png")
    imagen_estego = os.path.join(directorio_actual, "estego_lsb.png")
    mensaje = ("La tecnología ha transformado radicalmente la manera en que vivimos, trabajamos y nos comunicamos. " +
               "Gracias a los avances en inteligencia artificial, conectividad y automatización, las empresas pueden " +
               "optimizar sus procesos, mientras que las personas acceden a información en tiempo real. No obstante, " +
               "este progreso exige una adaptación constante y un compromiso ético para garantizar un desarrollo sostenible e inclusivo.")

    if not mostrar_info_capacidad(imagen_original, mensaje):
        exit()

    try:
        codificar_lsb(imagen_original, mensaje, imagen_estego)
    except Exception as e:
        print(f"Error al codificar: {e}")
        exit()
    try:
        mensaje_decodificado = decodificar_lsb(imagen_estego)
        print(f"\nMensaje decodificado: {mensaje_decodificado}")
        verificar_decodificacion(mensaje, mensaje_decodificado)
    except Exception as e:
        print(f"Error al decodificar: {e}")

    # Ejemplo de uso para Fourier Steganography
    print("\n" + "=" * 50)
    print("DEMOSTRACIÓN FOURIER STEGANOGRAPHY")
    print("=" * 50)

    cover_path = os.path.join(directorio_actual, "portadora.jpg")
    secret_path = os.path.join(directorio_actual, "secreta.png")
    stego_path = os.path.join(directorio_actual, "estego_fourier.png")
    recovered_path = os.path.join(directorio_actual, "secreta_recuperada.png")
    SECRET_SIZE = 256

    try:
        fourier_steganography_encode(cover_path, secret_path, stego_path)
        fourier_steganography_decode(stego_path, recovered_path, SECRET_SIZE)
        print(f"[INFO] Imagen secreta recuperada: {recovered_path}")

    except Exception as e:
        print(f"[ERROR] {e}")

    # Ejemplo de uso para Fourier Delta Steganography
    print("\n" + "=" * 50)
    print("DEMOSTRACIÓN FOURIER DELTA STEGANOGRAPHY")
    print("=" * 50)

    stego_delta_path = os.path.join(directorio_actual, "estego_fourier_delta.png")
    recovered_delta_path = os.path.join(directorio_actual, "secreta_recuperada_delta.png")
    DELTA = 10
    dimensiones_secreto = (256, 256)

    try:
        codificar_fourier_delta(cover_path, secret_path, stego_delta_path, DELTA)
        decodificar_fourier_delta(stego_delta_path, recovered_delta_path, dimensiones_secreto, DELTA)
        print(f"[INFO] Imagen secreta recuperada (Delta): {recovered_delta_path}")

    except Exception as e:
        print(f"[ERROR] {e}")
