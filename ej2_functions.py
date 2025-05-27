import os
import math
import numpy as np
from PIL import Image, ImageChops, UnidentifiedImageError
import matplotlib.pyplot as plt

from constants import SECRET_SIZE

# ==============================================
# Funciones para Esteganografía con TF2D (Ejercicio2.py)
# ==============================================

def fourier_steganography_encode(cover_img_path, secret_img_path, output_path):
    """Codifica una imagen secreta en los coeficientes de Fourier de una imagen portadora"""
    cover_img = np.array(Image.open(cover_img_path).convert('L'))
    secret_img = np.array(Image.open(secret_img_path).convert('L'))

    fshift = np.fft.fftshift(np.fft.fft2(cover_img))
    secret_bits = (secret_img > 127).astype(np.uint8).flatten()

    rows, cols = fshift.shape
    center_row, center_col = rows // 2, cols // 2
    radius = min(center_row, center_col)
    coords = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if np.sqrt((r - center_row) ** 2 + (c - center_col) ** 2) < radius
    ]

    modified = fshift.copy()
    for i in range(min(len(secret_bits), len(coords))):
        r, c = coords[i]
        val = modified[r, c]
        bit = secret_bits[i]
        a, b = (-abs(val.real), -abs(val.imag)) if bit else (abs(val.real), abs(val.imag))
        modified[r, c] = complex(a, b)

    stego_img = np.abs(np.fft.ifft2(np.fft.ifftshift(modified)))
    stego_img = np.clip(stego_img, 0, 255).astype(np.uint8)
    Image.fromarray(stego_img).save(output_path)
    print(f"[FOURIER] Imagen estego guardada en: {output_path}")


def fourier_steganography_decode(stego_img_path, output_path, secret_size= SECRET_SIZE):
    """Decodifica una imagen secreta de los coeficientes de Fourier"""
    secret_shape = (secret_size, secret_size)
    total_pixels = secret_shape[0] * secret_shape[1]

    stego_img = np.array(Image.open(stego_img_path).convert('L'))
    fshift = np.fft.fftshift(np.fft.fft2(stego_img))

    rows, cols = fshift.shape
    center_row, center_col = rows // 2, cols // 2
    radius = min(center_row, center_col)
    coords = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if np.sqrt((r - center_row) ** 2 + (c - center_col) ** 2) < radius
    ]

    extracted_bits = []
    for i in range(total_pixels):
        if i >= len(coords):
            break
        r, c = coords[i]
        a, b = fshift[r, c].real, fshift[r, c].imag
        extracted_bits.append(1 if (a < 0 or b < 0) else 0)

    extracted_bits += [0] * (total_pixels - len(extracted_bits))
    secret_img = np.array(extracted_bits[:total_pixels]).reshape(secret_shape)
    secret_img = (secret_img * 255).astype(np.uint8)
    Image.fromarray(secret_img).save(output_path)
    print(f"[FOURIER] Imagen secreta recuperada guardada en: {output_path}")
