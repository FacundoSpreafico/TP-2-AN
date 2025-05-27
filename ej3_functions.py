import numpy as np
from PIL import Image

def codificar_fourier_delta(ruta_imagen_portadora, ruta_imagen_secreta, ruta_salida, delta):
    imagen_portadora = np.array(Image.open(ruta_imagen_portadora).convert('L'))
    imagen_secreta = np.array(Image.open(ruta_imagen_secreta).convert('L'))

    # Transformada de Fourier
    transformada = np.fft.fftshift(np.fft.fft2(imagen_portadora))

    bits_secretos = (imagen_secreta > 127).astype(np.uint8).flatten()

    filas, columnas = transformada.shape
    coordenadas = [(f, c) for f in range(filas) for c in range(columnas)]

    transformada_modificada = transformada.copy()

    for i, bit in enumerate(bits_secretos):
        if i >= len(coordenadas):
            break
        f, c = coordenadas[i]
        a, b = transformada[f, c].real, transformada[f, c].imag

        signo_a = 1 if a >= 0 else -1
        signo_b = 1 if b >= 0 else -1

        q_a = abs(round(a / delta))
        q_b = abs(round(b / delta))

        # Ajustar paridad según bit secreto
        if bit == 0:
            q_a += q_a % 2
            q_b += q_b % 2
        else:
            q_a += (q_a + 1) % 2
            q_b += (q_b + 1) % 2

        transformada_modificada[f, c] = complex(signo_a * q_a * delta, signo_b * q_b * delta)

    # Transformada inversa
    imagen_estego = np.abs(np.fft.ifft2(np.fft.ifftshift(transformada_modificada)))
    imagen_estego = np.clip(imagen_estego, 0, 255).astype(np.uint8)

    Image.fromarray(imagen_estego).save(ruta_salida)
    print(f"[DELTA TF2D] Imagen estego guardada en: {ruta_salida}")


def decodificar_fourier_delta(ruta_imagen_estego, ruta_salida, dimensiones_secreto, delta):
    imagen_estego = np.array(Image.open(ruta_imagen_estego).convert('L'))

    transformada = np.fft.fftshift(np.fft.fft2(imagen_estego))

    filas, columnas = transformada.shape
    coordenadas = [(f, c) for f in range(filas) for c in range(columnas)]

    total_pixeles = dimensiones_secreto[0] * dimensiones_secreto[1]
    bits_extraidos = []

    for i in range(total_pixeles):
        if i >= len(coordenadas):
            break
        f, c = coordenadas[i]

        a, b = transformada[f, c].real, transformada[f, c].imag
        q_a = abs(round(a / delta))
        bit = q_a % 2
        bits_extraidos.append(bit)

    bits_extraidos += [0] * (total_pixeles - len(bits_extraidos))
    imagen_secreta = np.array(bits_extraidos).reshape(dimensiones_secreto)
    imagen_secreta = (imagen_secreta * 255).astype(np.uint8)

    Image.fromarray(imagen_secreta).save(ruta_salida)
    print(f"[DELTA TF2D] Imagen secreta recuperada guardada en: {ruta_salida}")
