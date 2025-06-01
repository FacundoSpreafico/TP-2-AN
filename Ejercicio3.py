import numpy as np
from PIL import Image

def codificar_fourier_delta(cover_img_path, secret_img_path, output_path, delta=10, secret_size=64):
    # Cargar y redimensionar imágenes
    cover_img = np.array(Image.open(cover_img_path).convert('L'))
    secret_img = np.array(Image.open(secret_img_path).convert('L'))
    secret_img = Image.fromarray(secret_img).resize((secret_size, secret_size), Image.LANCZOS)
    secret_img = np.array(secret_img)
    
    # Binarizar la imagen secreta
    secret_bits = (secret_img > 127).astype(np.uint8).flatten()
    required_bits = secret_size * secret_size
    
    # FFT de la portadora
    fshift = np.fft.fftshift(np.fft.fft2(cover_img))

    # Seleccionar zona circular central
    rows, cols = fshift.shape
    center_row, center_col = rows // 2, cols // 2
    radius = min(center_row, center_col)
    coords = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if np.sqrt((r - center_row) ** 2 + (c - center_col) ** 2) < radius
    ]

    available_bits = len(coords)
    if available_bits < required_bits:
        raise ValueError(f"Capacidad insuficiente: Se necesitan {required_bits} bits, solo hay {available_bits} disponibles")

    modified = fshift.copy()
    for i in range(required_bits):
        r, c = coords[i]
        a, b = modified[r, c].real, modified[r, c].imag
        signo_a = 1 if a >= 0 else -1
        signo_b = 1 if b >= 0 else -1

        q_a = abs(round(a / delta))
        q_b = abs(round(b / delta))
        bit = secret_bits[i]

        # Modificación basada en paridad del bit
        if bit == 0:
            if q_a % 2 == 1: q_a += 1  # Forzar par
            if q_b % 2 == 1: q_b += 1
        else:
            if q_a % 2 == 0: q_a += 1  # Forzar impar
            if q_b % 2 == 0: q_b += 1

        modified[r, c] = complex(signo_a * q_a * delta, signo_b * q_b * delta)

    # Reconstrucción de la imagen
    stego_img = np.abs(np.fft.ifft2(np.fft.ifftshift(modified)))
    stego_img = np.clip(stego_img, 0, 255).astype(np.uint8)
    Image.fromarray(stego_img).save(output_path)
    print(f"[DELTA-QIM] Imagen estego guardada en: {output_path}")
    return secret_size

def decodificar_fourier_delta(stego_img_path, output_path, secret_size=64, delta=10):
    total_bits = secret_size * secret_size
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

    bits = []
    for i in range(total_bits):
        r, c = coords[i]
        val = fshift[r, c]
        
        # Extraer información de ambas componentes
        q_real = abs(round(val.real / delta))
        q_imag = abs(round(val.imag / delta))
        
        # Usar redundancia: ambas componentes deben contener el mismo bit
        bit_real = q_real % 2
        bit_imag = q_imag % 2
        
        # Consenso entre componentes (prioriza real si hay discrepancia)
        bits.append(bit_real if bit_real == bit_imag else bit_real)

    secret_img = np.array(bits).reshape((secret_size, secret_size)).astype(np.uint8)
    secret_img = (secret_img * 255).astype(np.uint8)
    Image.fromarray(secret_img).save(output_path)
    print(f"[DELTA-QIM] Imagen secreta recuperada guardada en: {output_path}")

if __name__ == "__main__":
    portadora = "portadora.png"
    secreta = "secreta.png"
    estego_delta = "estego_fourier_delta.png"
    secreta_delta_recuperada = "secreta_recuperada_delta.png"
    secret_size = 256  # Tamaño reducido para prueba inicial
    delta = 20  # Valor delta aumentado para robustez

    try:
        codificar_fourier_delta(
            portadora, secreta, estego_delta, delta=delta, secret_size=secret_size
        )
        decodificar_fourier_delta(
            estego_delta, secreta_delta_recuperada, secret_size=secret_size, delta=delta
        )
        print("\nResultados:")
        print(f"- Imagen recuperada guardada en: {secreta_delta_recuperada}")
    except Exception as e:
        print(f"[ERROR] {e}")