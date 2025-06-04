import numpy as np
from PIL import Image

def fourier_steganography_encode(cover_img_path, secret_img_path, output_path, secret_size=64):
    cover_img = np.array(Image.open(cover_img_path).convert('L'))
    secret_img = np.array(Image.open(secret_img_path).convert('L'))
    secret_img = Image.fromarray(secret_img).resize((secret_size, secret_size), Image.LANCZOS)
    secret_img = np.array(secret_img)

    # Binarizar la imagen secreta
    secret_img_bin = (secret_img > 127).astype(np.uint8)
    secret_bits = secret_img_bin.flatten()
    required_bits = secret_size * secret_size

    # Transformada de Fourier y shift
    fshift = np.fft.fftshift(np.fft.fft2(cover_img))
    rows, cols = fshift.shape
    center_row, center_col = rows // 2, cols // 2
    radius = min(center_row, center_col)

    # Coordenadas dentro del círculo, evitando el centro
    coords = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if (r != center_row or c != center_col) and
           (r < center_row or (r == center_row and c <= center_col)) and
           np.sqrt((r - center_row)**2 + (c - center_col)**2) < radius
    ]

    if len(coords) < required_bits:
        raise ValueError(f"Capacidad insuficiente: {len(coords)} bits disponibles, se requieren {required_bits}")

    modified = fshift.copy()
    for i in range(required_bits):
        r, c = coords[i]
        bit = secret_bits[i]
        val = modified[r, c]
        a, b = val.real, val.imag

        # Codificación por signo
        a = -abs(a) if bit else abs(a)
        b = -abs(b) if bit else abs(b)
        modified[r, c] = complex(a, b)

        # Reflejar para mantener simetría hermítica
        rr = (2 * center_row - r) % rows
        cc = (2 * center_col - c) % cols
        modified[rr, cc] = np.conj(modified[r, c])

    # Transformada inversa
    stego_img = np.abs(np.fft.ifft2(np.fft.ifftshift(modified)))
    stego_img = np.clip(stego_img, 0, 255).astype(np.uint8)
    Image.fromarray(stego_img).save(output_path)
    return secret_size


def fourier_steganography_decode(stego_img_path, output_path, secret_size):
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
        if (r != center_row or c != center_col) and
           (r < center_row or (r == center_row and c <= center_col)) and
           np.sqrt((r - center_row)**2 + (c - center_col)**2) < radius
    ]

    extracted_bits = []
    for i in range(total_bits):
        r, c = coords[i]
        val = fshift[r, c]
        a, b = val.real, val.imag
        if a * b < 0:
            bit = 0  # signos opuestos → ruido
        elif a < 0 and b < 0:
            bit = 1
        else:
            bit = 0
        extracted_bits.append(bit)

    secret_img = np.array(extracted_bits).reshape((secret_size, secret_size)).astype(np.uint8) * 255
    Image.fromarray(secret_img).save(output_path)





if __name__ == "__main__":
    # Configuración
    portadora = "portadora.png"
    secreta = "secreta.png"
    estego = "estego_fourier.png"
    secreta_recuperada = "secreta_recuperada.png"

    try:
        # Codificar (retorna tamaño usado para decodificación)
        secret_size = fourier_steganography_encode(
            portadora,
            secreta,
            estego,
            secret_size=256  # Cambiá este valor según el tamaño deseado de la secreta
        )

        # Decodificar usando el mismo tamaño
        fourier_steganography_decode(
            estego,
            secreta_recuperada,
            secret_size=256
        )

        # Mostrar resultados
        print("\nResultados:")
        print(f"- Tamaño de secreto usado: {secret_size}x{secret_size}")
        print(f"- Imagen recuperada guardada en: {secreta_recuperada}")

        # Mostrar imágenes
        Image.open(secreta_recuperada).show()

    except Exception as e:
        print(f"[ERROR] {e}")
