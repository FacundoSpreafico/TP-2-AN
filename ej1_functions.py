from PIL import Image, ImageChops, UnidentifiedImageError
import numpy as np
import matplotlib.pyplot as plt
import math
import os

from constants import (
    BITS_POR_PIXEL
)
# ==============================================
# Funciones para Esteganografía con LSB Ejercicio1.py)
# ==============================================

def texto_a_binario(texto):
    """Convierte un texto a su representación binaria (8 bits por carácter)"""
    binario = []
    for char in texto:
        bin_char = bin(ord(char))[2:].zfill(8)
        binario.extend([int(bit) for bit in bin_char])
    return binario

def binario_a_texto(binario):
    """Convierte una secuencia binaria de vuelta a texto"""
    texto = []
    for i in range(0, len(binario), 8):
        byte = binario[i:i + 8]
        if len(byte) < 8:
            break
        char = chr(int(''.join(map(str, byte)), 2))
        texto.append(char)
    return ''.join(texto)

def calcular_capacidad_lsb(imagen_path, bits_pp=BITS_POR_PIXEL):
    """Calcula la capacidad máxima de mensaje para la imagen en LSB"""
    try:
        img = Image.open(imagen_path)
        ancho, alto = img.size
        capacidad_bits = ancho * alto * bits_pp
        capacidad_bytes = capacidad_bits // 8
        capacidad_caracteres = capacidad_bytes - 1  # Restamos 1 para el marcador '&'
        return capacidad_caracteres, capacidad_bits
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontró el archivo de imagen: {imagen_path}")
    except UnidentifiedImageError:
        raise UnidentifiedImageError(f"No se pudo identificar la imagen: {imagen_path}")
    except Exception as e:
        raise Exception(f"Error al procesar la imagen: {str(e)}")

def mostrar_conversion_mensaje(mensaje):
    """Muestra la conversión paso a paso del mensaje"""
    print("\n--- Conversión del mensaje ---")
    print(f"Mensaje original: {mensaje}")
    print("ASCII: ", end="")
    ascii_values = [ord(c) for c in mensaje]
    print(ascii_values)

    print("Binario: ", end="")
    for c in mensaje:
        print(f"{ord(c):08b}", end=" ")
    print()

    pixel_ejemplo = 189
    bit_ejemplo = 1
    print(f"\nEjemplo modificación LSB:\nPixel original: {pixel_ejemplo} ({pixel_ejemplo:08b})")
    pixel_modificado = (pixel_ejemplo & 0xFE) | bit_ejemplo
    print(f"Pixel modificado: {pixel_modificado} ({pixel_modificado:08b})")

def analizar_estego_imagen(original_path, estego_path):
    """Realiza análisis detallado de la imagen estego"""
    try:
        original = Image.open(original_path).convert("L")
        estego = Image.open(estego_path).convert("L")

        if original.size != estego.size:
            raise ValueError("Las imágenes deben tener las mismas dimensiones")

        diff = ImageChops.difference(original, estego)
        diff_array = np.array(diff)

        total_pixeles = original.size[0] * original.size[1]
        pixeles_modificados = np.sum(diff_array > 0)
        porcentaje_modificado = (pixeles_modificados / total_pixeles) * 100

        print("\n--- Análisis de la imagen estego ---")
        print(f"Dimensiones: {original.size[0]}x{original.size[1]}")
        print(f"Total de píxeles: {total_pixeles:,}")
        print(f"Píxeles modificados: {pixeles_modificados:,} ({porcentaje_modificado:.6f}%)")
        print(f"Relación modificación: 1 cada {int(total_pixeles / pixeles_modificados)} píxeles")

        # Visualización (histogramas e imágenes)
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.hist(np.array(original).ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7)
        plt.title("Histograma Original")

        plt.subplot(1, 3, 2)
        plt.hist(np.array(estego).ravel(), bins=256, range=(0, 256), color='green', alpha=0.7)
        plt.title("Histograma Estego")

        plt.subplot(1, 3, 3)
        plt.hist(diff_array.ravel(), bins=range(0, 3), align='left', color='red', alpha=0.7, rwidth=0.8)
        plt.title("Histograma de Diferencias")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"\n❌ Error durante el análisis: {str(e)}")
        raise

def codificar_lsb(imagen_path, mensaje, salida_path, bits_pp = BITS_POR_PIXEL):
    """Codifica un mensaje en una imagen usando LSB"""
    capacidad_caracteres, _ = calcular_capacidad_lsb(imagen_path, bits_pp)
    print(f"\nCapacidad máxima de la imagen: {capacidad_caracteres} caracteres (usando {bits_pp} LSB por píxel)")

    mostrar_conversion_mensaje(mensaje)

    img = Image.open(imagen_path).convert('L')
    img_array = np.array(img)
    mensaje += '&'
    binario = texto_a_binario(mensaje)

    if len(binario) > img_array.size * bits_pp:
        raise ValueError(f"El mensaje es demasiado largo para la imagen. Máximo: {capacidad_caracteres} caracteres")

    flat_img = img_array.flatten()
    for i in range(len(binario)):
        pixel_idx = i // bits_pp
        bit_pos = i % bits_pp
        if pixel_idx >= len(flat_img):
            break
        mask = 255 ^ (1 << bit_pos)
        flat_img[pixel_idx] = (flat_img[pixel_idx] & mask) | (binario[i] << bit_pos)

    img_estego = flat_img.reshape(img_array.shape)
    Image.fromarray(img_estego).save(salida_path)
    print(f"\nMensaje codificado en {salida_path}")
    print(f"Total de bits del mensaje: {len(binario)}")
    print(f"Total de píxeles modificados: {math.ceil(len(binario) / bits_pp)}")

def decodificar_lsb(imagen_path, bits_pp = BITS_POR_PIXEL):
    """Decodifica un mensaje de una imagen usando LSB"""
    img = Image.open(imagen_path).convert('L')
    img_array = np.array(img)

    bits = []
    for pixel in img_array.flatten():
        for bit_pos in range(bits_pp):
            bits.append((pixel >> bit_pos) & 1)

    mensaje_binario = []
    marcador = [0, 0, 1, 0, 0, 1, 1, 0]  # '&' en binario

    for i in range(0, len(bits), 8):
        byte = bits[i:i + 8]
        if len(byte) < 8:
            break
        if byte == marcador:
            break
        mensaje_binario.extend(byte)

    return binario_a_texto(mensaje_binario)

def mostrar_info_capacidad(imagen_path, mensaje):
    """Muestra información sobre la capacidad de la imagen"""
    print(f"\n{'=' * 50}")
    print(f"Capacidad de la imagen {imagen_path}:")

    try:
        if not os.path.exists(imagen_path):
            raise FileNotFoundError(f"El archivo {imagen_path} no existe")

        capacidad_caracteres, capacidad_bits = calcular_capacidad_lsb(imagen_path)
        print(f"- Máximo caracteres (1 LSB por píxel): {capacidad_caracteres}")
        print(f"- Máximo bits (1 LSB por píxel): {capacidad_bits}")
        print(f"Longitud del mensaje a codificar: {len(mensaje)} caracteres")

        if len(mensaje) > capacidad_caracteres:
            print("\n ADVERTENCIA: El mensaje es más largo que la capacidad de la imagen")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return False

    print(f"{'=' * 50}")
    return True

def verificar_decodificacion(mensaje_original, mensaje_decodificado):
    """Verifica si el mensaje decodificado coincide con el original"""
    if mensaje_original == mensaje_decodificado:
        print("\n✅ El mensaje decodificado coincide exactamente con el original!")
    else:
        print("\n❌ Hubo un error en la decodificación.")
        print(f"Diferencia: {len(mensaje_original) - len(mensaje_decodificado)} caracteres")
