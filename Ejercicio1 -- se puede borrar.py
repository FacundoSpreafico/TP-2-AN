import os
from PIL import Image, ImageChops, UnidentifiedImageError
import numpy as np
import matplotlib.pyplot as plt
import math


def texto_a_binario(texto):
    """Convierte un texto a su representación binaria (8 bits por carácter)"""
    binario = []
    for char in texto:
        # Convertir cada carácter a su valor ASCII y luego a binario de 8 bits
        bin_char = bin(ord(char))[2:].zfill(8)
        binario.extend([int(bit) for bit in bin_char])
    return binario


def binario_a_texto(binario):
    """Convierte una secuencia binaria de vuelta a texto"""
    texto = []
    # Agrupar los bits en grupos de 8
    for i in range(0, len(binario), 8):
        byte = binario[i:i + 8]
        if len(byte) < 8:
            break  # Ignorar bits incompletos al final
        # Convertir el byte binario a carácter ASCII
        char = chr(int(''.join(map(str, byte)), 2))
        texto.append(char)
    return ''.join(texto)

def calcular_capacidad(imagen_path, bits_por_pixel=1):
    """Calcula la capacidad máxima de mensaje para la imagen"""
    img = Image.open(imagen_path)
    ancho, alto = img.size
    capacidad_bits = ancho * alto * bits_por_pixel
    capacidad_bytes = capacidad_bits // 8
    capacidad_caracteres = capacidad_bytes - 1  # Restamos 1 para el marcador '&'
    return capacidad_caracteres, capacidad_bits

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

    # Mostrar ejemplo de modificación de píxel
    print("\nEjemplo modificación LSB:")
    pixel_ejemplo = 189  # Valor de ejemplo
    bit_ejemplo = 1  # Bit de ejemplo
    print(f"Pixel original: {pixel_ejemplo} ({pixel_ejemplo:08b})")
    pixel_modificado = (pixel_ejemplo & 0xFE) | bit_ejemplo
    print(f"Pixel modificado: {pixel_modificado} ({pixel_modificado:08b})")


def analizar_estego_imagen(original_path, estego_path):
    """Realiza análisis detallado de la imagen estego"""
    try:
        # Abrir imágenes con manejo de errores
        try:
            original = Image.open(original_path).convert("L")
            estego = Image.open(estego_path).convert("L")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"No se encontró el archivo: {str(e)}")
        except Exception as e:
            raise Exception(f"Error al abrir imágenes: {str(e)}")

        # Verificar que las imágenes tengan el mismo tamaño
        if original.size != estego.size:
            raise ValueError("Las imágenes deben tener las mismas dimensiones")

        # Calcular diferencias
        diff = ImageChops.difference(original, estego)
        diff_array = np.array(diff)

        # Estadísticas
        total_pixeles = original.size[0] * original.size[1]
        pixeles_modificados = np.sum(diff_array > 0)
        porcentaje_modificado = (pixeles_modificados / total_pixeles) * 100

        print("\n--- Análisis de la imagen estego ---")
        print(f"Dimensiones: {original.size[0]}x{original.size[1]}")
        print(f"Total de píxeles: {total_pixeles:,}")
        print(f"Píxeles modificados: {pixeles_modificados:,} ({porcentaje_modificado:.6f}%)")
        print(f"Relación modificación: 1 cada {int(total_pixeles / pixeles_modificados)} píxeles")

        # Mostrar histogramas
        plt.figure(figsize=(15, 5))

        # Histograma original
        plt.subplot(1, 3, 1)
        plt.hist(np.array(original).ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7)
        plt.title("Histograma Original")
        plt.xlabel("Valor de píxel")
        plt.ylabel("Frecuencia")

        # Histograma estego
        plt.subplot(1, 3, 2)
        plt.hist(np.array(estego).ravel(), bins=256, range=(0, 256), color='green', alpha=0.7)
        plt.title("Histograma Estego")
        plt.xlabel("Valor de píxel")

        # Histograma diferencias (ampliado para mejor visualización)
        plt.subplot(1, 3, 3)
        plt.hist(diff_array.ravel(), bins=range(0, 3), align='left', color='red', alpha=0.7, rwidth=0.8)
        plt.title("Histograma de Diferencias")
        plt.xlabel("Diferencia")
        plt.xticks([0, 1])
        plt.tight_layout()
        plt.show()

        # Mostrar imágenes comparativas
        plt.figure(figsize=(15, 5))

        # Imagen original
        plt.subplot(1, 3, 1)
        plt.imshow(original, cmap='gray', vmin=0, vmax=255)
        plt.title("Imagen Original")
        plt.axis('off')

        # Imagen estego
        plt.subplot(1, 3, 2)
        plt.imshow(estego, cmap='gray', vmin=0, vmax=255)
        plt.title("Imagen Estego")
        plt.axis('off')

        # Diferencias (resaltadas)
        plt.subplot(1, 3, 3)
        plt.imshow(diff_array, cmap='hot', interpolation='nearest')
        plt.title("Mapa de Diferencias")
        plt.colorbar(label='Magnitud de cambio')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"\n❌ Error durante el análisis: {str(e)}")
        raise

def codificar_lsb(imagen_path, mensaje, salida_path, bits_por_pixel=1):
    """Codifica un mensaje en una imagen usando LSB"""
    # Mostrar información de capacidad
    capacidad_caracteres, _ = calcular_capacidad(imagen_path, bits_por_pixel)
    print(f"\nCapacidad máxima de la imagen: {capacidad_caracteres} caracteres (usando {bits_por_pixel} LSB por píxel)")

    # Mostrar conversión del mensaje
    mostrar_conversion_mensaje(mensaje)

    # Abrir la imagen y convertir a escala de grises
    img = Image.open(imagen_path).convert('L')
    img_array = np.array(img)

    # Convertir el mensaje a binario y agregar marcador de fin '&'
    mensaje += '&'
    binario = texto_a_binario(mensaje)

    # Verificar si el mensaje cabe en la imagen
    if len(binario) > img_array.size * bits_por_pixel:
        raise ValueError(f"El mensaje es demasiado largo para la imagen. Máximo: {capacidad_caracteres} caracteres")

    # Codificar el mensaje en los LSBs
    flat_img = img_array.flatten()
    for i in range(len(binario)):
        # Calcular índice de píxel y bit dentro del píxel
        pixel_idx = i // bits_por_pixel
        bit_pos = i % bits_por_pixel

        if pixel_idx >= len(flat_img):
            break

        # SOLUCIÓN: Usar máscara positiva
        mask = 255 ^ (1 << bit_pos)  # Máscara siempre positiva
        flat_img[pixel_idx] = (flat_img[pixel_idx] & mask) | (binario[i] << bit_pos)

    # Reconstruir la imagen y guardarla
    img_estego = flat_img.reshape(img_array.shape)
    Image.fromarray(img_estego).save(salida_path)
    print(f"\nMensaje codificado en {salida_path}")
    print(f"Total de bits del mensaje: {len(binario)}")
    print(f"Total de píxeles modificados: {math.ceil(len(binario) / bits_por_pixel)}")

def decodificar_lsb(imagen_path, bits_por_pixel=1):
    """Decodifica un mensaje de una imagen usando LSB"""
    # Abrir la imagen estego
    img = Image.open(imagen_path).convert('L')
    img_array = np.array(img)

    # Extraer los bits según la cantidad de bits por píxel
    bits = []
    for pixel in img_array.flatten():
        for bit_pos in range(bits_por_pixel):
            bits.append((pixel >> bit_pos) & 1)

    # Buscar el marcador de fin '&' (00100110 en binario)
    mensaje_binario = []
    marcador = [0, 0, 1, 0, 0, 1, 1, 0]  # '&' en binario

    for i in range(0, len(bits), 8):
        byte = bits[i:i + 8]
        if len(byte) < 8:
            break

        # Verificar si es el marcador de fin
        if byte == marcador:
            break

        mensaje_binario.extend(byte)

    # Convertir los bits a texto
    mensaje = binario_a_texto(mensaje_binario)
    return mensaje

def calcular_capacidad(imagen_path, bits_por_pixel=1):
    """Calcula la capacidad máxima de mensaje para la imagen"""
    try:
        img = Image.open(imagen_path)
        ancho, alto = img.size
        capacidad_bits = ancho * alto * bits_por_pixel
        capacidad_bytes = capacidad_bits // 8
        capacidad_caracteres = capacidad_bytes - 1  # Restamos 1 para el marcador '&'
        return capacidad_caracteres, capacidad_bits
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontró el archivo de imagen: {imagen_path}")
    except UnidentifiedImageError:
        raise UnidentifiedImageError(
            f"No se pudo identificar la imagen: {imagen_path}. Verifica que sea un archivo de imagen válido.")
    except Exception as e:
        raise Exception(f"Error al procesar la imagen: {str(e)}")

def mostrar_info_capacidad(imagen_path, mensaje):
    """Muestra información sobre la capacidad de la imagen con manejo de errores"""
    print(f"\n{'=' * 50}")
    print(f"Capacidad de la imagen {imagen_path}:")

    try:
        if not os.path.exists(imagen_path):
            raise FileNotFoundError(f"El archivo {imagen_path} no existe en el directorio actual")

        capacidad_caracteres, capacidad_bits = calcular_capacidad(imagen_path)
        print(f"- Máximo caracteres (1 LSB por píxel): {capacidad_caracteres}")
        print(f"- Máximo bits (1 LSB por píxel): {capacidad_bits}")
        print(f"Longitud del mensaje a codificar: {len(mensaje)} caracteres")

        if len(mensaje) > capacidad_caracteres:
            print("\n⚠️ ADVERTENCIA: El mensaje es más largo que la capacidad de la imagen")
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

if __name__ == "__main__":
    # Configuración - ahora con rutas absolutas para mejor control
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    imagen_original = os.path.join(directorio_actual, "secreta.png")
    imagen_estego = os.path.join(directorio_actual, "estego.png")

    mensaje = ("La tecnología ha transformado radicalmente la manera en que vivimos, trabajamos y nos comunicamos. " +
               "Gracias a los avances en inteligencia artificial, conectividad y automatización, las empresas pueden " +
               "optimizar sus procesos, mientras que las personas acceden a información en tiempo real. No obstante, " +
               "este progreso exige una adaptación constante y un compromiso ético para garantizar un desarrollo sostenible e inclusivo.")

    # 1. Mostrar capacidad de la imagen (con verificación)
    if not mostrar_info_capacidad(imagen_original, mensaje):
        exit()  # Salir si hay error con la imagen

    # 2. Codificar el mensaje
    try:
        print("\nCodificando mensaje...")
        codificar_lsb(imagen_original, mensaje, imagen_estego)
    except ValueError as e:
        print(f"Error al codificar: {e}")
        exit()
    except Exception as e:
        print(f"Error inesperado al codificar: {str(e)}")
        exit()

    # Resto del código (análisis, decodificación, verificación) se mantiene igual
    try:
        # 3. Analizar imagen estego
        analizar_estego_imagen(imagen_original, imagen_estego)

        # 4. Decodificar el mensaje
        print("\nDecodificando mensaje...")
        mensaje_decodificado = decodificar_lsb(imagen_estego)
        print(f"\nMensaje decodificado: {mensaje_decodificado}")

        # 5. Verificación
        verificar_decodificacion(mensaje, mensaje_decodificado)
    except Exception as e:
        print(f"Error en el proceso de análisis/decodificación: {str(e)}")