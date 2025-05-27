import numpy as np
from PIL import Image

def fourier_steganography_encode(cover_img_path, secret_img_path, output_path):
    # Cargar imágenes
    cover_img = np.array(Image.open(cover_img_path).convert('L'))
    secret_img = np.array(Image.open(secret_img_path).convert('L'))

    # FFT de la imagen cubierta
    fshift = np.fft.fftshift(np.fft.fft2(cover_img))

    # Convertir la imagen secreta en bits binarios (1 y 0)
    secret_bits = (secret_img > 127).astype(np.uint8).flatten()

    # Obtener coordenadas circulares desde el centro
    rows, cols = fshift.shape
    center_row, center_col = rows // 2, cols // 2
    radius = min(center_row, center_col)
    coords = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if np.sqrt((r - center_row)**2 + (c - center_col)**2) < radius
    ]

    # Insertar todos los bits posibles
    modified = fshift.copy()
    for i in range(min(len(secret_bits), len(coords))):
        r, c = coords[i]
        val = modified[r, c]
        a, b = val.real, val.imag
        bit = secret_bits[i]

        if bit == 1:
            a, b = -abs(a), -abs(b)  # Bit 1: negativo
        else:
            a, b = abs(a), abs(b)     # Bit 0: positivo

        modified[r, c] = complex(a, b)

    # Transformada inversa para obtener imagen estego
    stego_img = np.abs(np.fft.ifft2(np.fft.ifftshift(modified)))
    stego_img = np.clip(stego_img, 0, 255).astype(np.uint8)

    # Guardar imagen estego
    Image.fromarray(stego_img).save(output_path)
    print(f"[INFO] Imagen estego guardada en: {output_path}")


def fourier_steganography_decode(stego_img_path, output_path, secret_size=64):
    # Asumimos un tamaño fijo para la imagen secreta (ej: 64x64)
    # ¡Debes conocer este valor de antemano!
    secret_shape = (secret_size, secret_size)
    total_pixels = secret_shape[0] * secret_shape[1]

    # Cargar imagen estego
    stego_img = np.array(Image.open(stego_img_path).convert('L'))

    # FFT
    fshift = np.fft.fftshift(np.fft.fft2(stego_img))

    # Extraer coordenadas circulares
    rows, cols = fshift.shape
    center_row, center_col = rows // 2, cols // 2
    radius = min(center_row, center_col)
    coords = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if np.sqrt((r - center_row)**2 + (c - center_col)**2) < radius
    ]

    # Extraer bits (asumimos que todos los coords fueron usados)
    extracted_bits = []
    for i in range(total_pixels):
        if i >= len(coords):
            break  # Si no hay suficientes bits, terminar
        r, c = coords[i]
        a, b = fshift[r, c].real, fshift[r, c].imag
        bit = 1 if (a < 0 or b < 0) else 0
        extracted_bits.append(bit)

    # Rellenar con ceros si faltan bits
    extracted_bits += [0] * (total_pixels - len(extracted_bits))

    # Reconstruir imagen
    secret_img = np.array(extracted_bits[:total_pixels]).reshape(secret_shape)
    secret_img = (secret_img * 255).astype(np.uint8)

    # Guardar imagen recuperada
    Image.fromarray(secret_img).save(output_path)
    print(f"[INFO] Imagen secreta recuperada guardada en: {output_path}")


if __name__ == "__main__":
    cover_path = "portadora.jpg"
    secret_path = "secreta.png"
    stego_path = "stego_image.png"
    recovered_path = "recovered_secret.png"

    # Tamaño de la imagen secreta (DEBES CONOCERLO)
    SECRET_SIZE = 256  # Ejemplo: 64x64

    try:
        fourier_steganography_encode(cover_path, secret_path, stego_path)
        fourier_steganography_decode(stego_path, recovered_path, SECRET_SIZE)

        print(f"[INFO] Mostrando imagen recuperada: {recovered_path}")
        Image.open(recovered_path).show()
    except Exception as e:
        print(f"[ERROR] {e}")