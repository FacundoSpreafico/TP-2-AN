import os
from tkinter import Image
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr

from constants import (
    DELTA,
    REDUNDANCIA,
    TAMAÑO_PORTA,
    TAMAÑO_SECRETA
)


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
   cargar_imagen_escala_grises,
    insertar_mensaje,
    ajustar_bit_q,
    extraer_mensaje,
    calcular_metricas,
    mostrar_resultados
)

if __name__ == "__main__":
    # Configuración común
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    print("\n" + "=" * 50)
    print("EJERCICIO 1 - LSB STEGANOGRAPHY")
    print("=" * 50)

    imagen_original = os.path.join(directorio_actual, "portadora.png")
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
    print("EJERCICIO 2 - FOURIER STEGANOGRAPHY")
    print("=" * 50)

    cover_path = os.path.join(directorio_actual, "portadora.png")
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
        
        
    print("\n" + "=" * 50)
    print("EJERCICIO 3 - FOURIER STEGANOGRAPHY")
    print("=" * 50)
    # Configuración
    try:
        print("=== CARGANDO IMÁGENES ===")
        portadora = cargar_imagen_escala_grises("portadora.png", TAMAÑO_PORTA)
        secreta = cargar_imagen_escala_grises("secreta.png", TAMAÑO_SECRETA)
        
        print("\n=== INSERTANDO MENSAJE ===")
        estego_float, indices, _ = insertar_mensaje(
            portadora, secreta, DELTA, REDUNDANCIA
        )
        estego_uint8 = np.clip(estego_float, 0, 255).astype(np.uint8)
        Image.fromarray(estego_uint8).save("estego_delta.png")
        print("Imagen estego guardada: 'estego_delta.png'")
        
        print("\n=== EXTRAYENDO MENSAJE ===")
        secreta_rec, _ = extraer_mensaje(
            estego_float, DELTA, secreta.shape, indices, REDUNDANCIA
        )
        Image.fromarray(secreta_rec.astype(np.uint8)).save("secreta_recuperada_delta.png")
        print("Imagen secreta recuperada: 'secreta_recuperada_delta.png'")
        
        print("\n=== CALCULANDO MÉTRICAS ===")
        precision, psnr_val = calcular_metricas(secreta, secreta_rec, portadora, estego_uint8)
        print(f"Precisión de bits: {precision*100:.2f}%")
        print(f"PSNR: {psnr_val:.2f} dB")
        
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        raise