import os
from PIL import Image
import numpy as np

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

if __name__ == "__main__":

    # Configuración - ahora con rutas absolutas para mejor control
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    imagen_decodificada = os.path.join(directorio_actual, "nico.png")

    try:

        # 4. Decodificar el mensaje
        print("\nDecodificando mensaje...")
        mensaje_decodificado = decodificar_lsb(imagen_decodificada)
        print(f"\nMensaje decodificado: {mensaje_decodificado}")

    except Exception as e:
        print(f"Error en el proceso de análisis/decodificación: {str(e)}")