import numpy as np
import math
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

def binarizar_imagen(img):
    """
    Convierte una imagen (0..255) en bits {0,1} usando umbral 127. Devuelve un array de bits (aplanado).
    """
    return (img > 127).astype(np.uint8).flatten()


def redimensionar_secreta_a_capacidad(secret_img, capacity_bits):
    """
    Redimensiona la imagen secreta para que quepa en la capacidad disponible.
    """
    h_s, w_s = secret_img.shape
    num_pixels = h_s * w_s
    if num_pixels <= capacity_bits:
        return secret_img

    lado = int(math.floor(math.sqrt(capacity_bits)))
    if lado < 1:
        raise RuntimeError("Carrier demasiado pequeño para ocultar ni un solo bit.")

    # Convertir a PIL Image para redimensionar
    secret_pil = Image.fromarray(secret_img)
    resized = secret_pil.resize((lado, lado), Image.Resampling.LANCZOS)
    return np.array(resized)


def fourier_steganography_encode(carrier_img, secret_img):
    """
    Embedding usando cambio de signo en coeficientes de magnitud mínima con PIL.
    """
    h, w = carrier_img.shape
    ch, cw = h // 2, w // 2

    # 1) Binarizar la imagen secreta
    bits_initial = binarizar_imagen(secret_img)
    bits_secret = bits_initial.size

    # 2) FFT2D + FFTSHIFT de la portadora
    fft = np.fft.fft2(carrier_img.astype(np.float32))
    fft_shift = np.fft.fftshift(fft)
    real = np.real(fft_shift).astype(np.float32)
    imag = np.imag(fft_shift).astype(np.float32)

    # 3) Crear lista de coeficientes (excluyendo DC)
    mag = np.abs(fft_shift).flatten()
    idx_flat = np.arange(h * w)
    dc_flat = ch * w + cw
    mag[dc_flat] = np.inf  # Excluir DC

    coef_list = [(mag_val, flat) for mag_val, flat in zip(mag, idx_flat) if flat != dc_flat]
    coef_list.sort(key=lambda x: x[0])  # Ordenar por magnitud

    # 5) Calcular capacidad
    capacity_bits = len(coef_list) // 2

    # Redimensionar secreta si es necesario
    if bits_secret > capacity_bits:
        secret_resized = redimensionar_secreta_a_capacidad(secret_img, capacity_bits)
        bits_initial = binarizar_imagen(secret_resized)
        bits_secret = bits_initial.size
    else:
        secret_resized = secret_img

    num_coef_needed = bits_secret

    if num_coef_needed > len(coef_list):
        raise RuntimeError("Capacidad insuficiente para los bits secretos.")

    # Obtener índices de los coeficientes a modificar
    selected = [flat for (_, flat) in coef_list[:num_coef_needed]]
    indices_used = [(flat // w, flat % w) for flat in selected]

    # Modificar coeficientes
    for k in range(num_coef_needed):
        bit = int(bits_initial[k])
        i, j = indices_used[k]
        sign = -1.0 if bit else 1.0

        # Modificar coeficiente y su simétrico
        real[i, j] = sign * abs(real[i, j])
        imag[i, j] = sign * abs(imag[i, j])

        # Simetría Hermitiana
        sym_i, sym_j = (h - i) % h, (w - j) % w
        real[sym_i, sym_j] = real[i, j]
        imag[sym_i, sym_j] = -imag[i, j]

    # Reconstruir imagen estego
    new_fft_shift = real + 1j * imag
    new_fft = np.fft.ifftshift(new_fft_shift)
    stego_complex = np.fft.ifft2(new_fft)
    stego_float = np.real(stego_complex).astype(np.float32)

    return stego_float, indices_used, secret_resized.shape


def fourier_steganography_decode(stego_float, secret_shape, indices_used):
    """
    Decodificación de la imagen secreta usando PIL.
    """
    fft_rec = np.fft.fft2(stego_float.astype(np.float32))
    fft_shift = np.fft.fftshift(fft_rec)
    real_rec = np.real(fft_shift).astype(np.float32)

    total_bits = secret_shape[0] * secret_shape[1]
    bits_rec = np.zeros(total_bits, dtype=np.uint8)

    for k in range(total_bits):
        if k >= len(indices_used):
            break
        i, j = indices_used[k]
        bits_rec[k] = 0 if real_rec[i, j] >= 0 else 1

    recovered = (bits_rec[:total_bits].reshape(secret_shape) * 255).astype(np.uint8)
    return recovered
