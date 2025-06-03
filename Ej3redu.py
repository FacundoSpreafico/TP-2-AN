import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr

def load_grayscale_image(path, size=None):
    """Carga una imagen en escala de grises y opcionalmente la redimensiona."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo abrir la imagen en: {path}")
    if size is not None:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img

def embed_bit_in_q(q, bit):
    """Ajusta q para que su paridad guarde `bit`."""
    if (q % 2) != bit:
        q = q + 1 if bit == 1 else q - 1
    return q

def esteganografia_fourier_complex_redundant(carrier_img, secret_img, delta, redundancy=3):
    """
    Inserta cada bit de secret_img con redundancia en componentes real e imaginaria.
    Devuelve:
    - stego_float: imagen estego en float32
    - indices: posiciones de los coeficientes usados
    - magnitudes_pre: magnitudes antes de la inserción
    """
    # 1) Binarizar secret_img (0/1)
    _, secret_bw = cv2.threshold(secret_img, 127, 1, cv2.THRESH_BINARY)
    bits = secret_bw.flatten().astype(np.uint8)
    total_bits = bits.size

    # 2) FFT2D + FFTSHIFT de la portadora
    fft = np.fft.fft2(carrier_img.astype(np.float32))
    fft_shifted = np.fft.fftshift(fft)
    real = np.real(fft_shifted).astype(np.float32)
    imag = np.imag(fft_shifted).astype(np.float32)
    h, w = carrier_img.shape

    # 3) Seleccionar coeficientes por magnitud (excluyendo DC)
    mag_complex = np.abs(fft_shifted).flatten()
    idx_flat = np.arange(h * w)
    dc_flat = (h // 2) * w + (w // 2)
    mag_complex[dc_flat] = -1  # excluir DC

    sorted_flat = idx_flat[np.argsort(-mag_complex)]
    needed = total_bits * redundancy
    if needed > h * w - 1:
        raise RuntimeError(f"Redundancia demasiado alta. Se necesitan {needed} coeficientes.")
    selected_flat = sorted_flat[:needed]
    indices = [(flat // w, flat % w) for flat in selected_flat]
    magnitudes_pre = np.array([mag_complex[flat] for flat in selected_flat], dtype=np.float32)

    # 4) Insertar bits con redundancia
    for k in range(total_bits):
        bit_val = int(bits[k])
        for r in range(redundancy):
            idx = k * redundancy + r
            i, j = indices[idx]

            # Modificar parte real
            a = real[i, j]
            signa = np.sign(a)
            q_r = int(round(abs(a) / delta))
            q_r = embed_bit_in_q(q_r, bit_val)
            real[i, j] = signa * (q_r * delta)

            # Modificar parte imaginaria
            b = imag[i, j]
            signb = np.sign(b)
            q_i = int(round(abs(b) / delta))
            q_i = embed_bit_in_q(q_i, bit_val)
            imag[i, j] = signb * (q_i * delta)

    # 5) Reconstruir imagen estego
    new_fft = real + 1j * imag
    new_fft_ishift = np.fft.ifftshift(new_fft)
    stego_complex = np.fft.ifft2(new_fft_ishift)
    return np.real(stego_complex).astype(np.float32), indices, magnitudes_pre

def extraer_mensaje_complex_redundant(stego_img_float, delta, secret_shape, indices, redundancy=3):
    """Extrae el mensaje usando votación mayoritaria sobre inserciones redundantes."""
    total_bits = secret_shape[0] * secret_shape[1]

    # 1) FFT de la imagen estego
    fft = np.fft.fft2(stego_img_float)
    fft_shifted = np.fft.fftshift(fft)
    real_rec = np.real(fft_shifted)
    imag_rec = np.imag(fft_shifted)

    # 2) Recuperar bits con votación mayoritaria
    bits_rec = np.zeros(total_bits, dtype=np.uint8)
    for k in range(total_bits):
        votes = 0
        for r in range(redundancy):
            idx = k * redundancy + r
            i, j = indices[idx]

            # Votos de parte real e imaginaria
            q_r = int(round(abs(real_rec[i, j]) / delta))
            q_i = int(round(abs(imag_rec[i, j]) / delta))
            votes += (q_r % 2) + (q_i % 2)

        bits_rec[k] = 1 if votes >= redundancy else 0

    return bits_rec.reshape(secret_shape) * 255, bits_rec
    

if __name__ == "__main__":
    
    try:
        # 1) Cargar imágenes
        carrier = load_grayscale_image("portadora.png", size=(1300, 1300))
        secret = load_grayscale_image("secreta.png", size=(256, 256))

        # 2) Parámetros
        delta = 10             # Factor de escala para la inserción
        redundancy = 10          # Número de inserciones redundantes por bit: en 5 pareciera que anda bien.

        # 3) Insertar mensaje
        print("=== Codificación ===")
        stego_float, indices, _ = esteganografia_fourier_complex_redundant(
            carrier, secret, delta, redundancy
        )
        stego_uint8 = np.clip(stego_float, 0, 255).astype(np.uint8)
        cv2.imwrite("estego.png", stego_uint8)
        print("[✔] Imagen esteganografiada guardada como 'estego.png'")

        # 4) Extraer mensaje
        print("\n=== Decodificación ===")
        mensaje_recuperado, bits_rec = extraer_mensaje_complex_redundant(
            stego_float, delta, secret.shape, indices, redundancy
        )
        cv2.imwrite("secreta_recuperada.png", mensaje_recuperado)
        print("[✔] Imagen secreta recuperada guardada como 'secreta_recuperada.png'")

        # 5) Métricas
        _, secret_bw = cv2.threshold(secret, 127, 1, cv2.THRESH_BINARY)
        accuracy = np.mean(secret_bw.flatten() == bits_rec)
        psnr_val = psnr(carrier, stego_uint8)
        
        print(f"\n🔍 Precisión de bits: {accuracy*100:.2f}%")
        print(f"🔍 PSNR (portadora vs estego): {psnr_val:.2f} dB")

        # 6) Mostrar resultados
        plt.figure(figsize=(12, 4))
        ax1 = plt.subplot(1, 3, 1)
        ax1.imshow(carrier, cmap='gray')
        ax1.set_title('Portadora')
        ax2 = plt.subplot(1, 3, 2)
        ax2.imshow(stego_uint8, cmap='gray')
        ax2.set_title(f'Estego (δ={delta})')
        ax3 = plt.subplot(1, 3, 3)
        ax3.imshow(mensaje_recuperado, cmap='gray')
        ax3.set_title('Secreto Recuperado')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"[ERROR] {str(e)}")