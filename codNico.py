import cv2
import numpy as np
import matplotlib.pyplot as plt

def img_to_bits(img):
    """Convierte una imagen en escala de grises a una cadena de bits."""
    flat = img.flatten()
    bits = ''.join([format(val, '08b') for val in flat])
    return bits

def bits_to_img(bits, shape):
    """Convierte una cadena de bits a una imagen en escala de grises."""
    vals = [int(bits[i:i+8], 2) for i in range(0, len(bits), 8)]
    arr = np.array(vals, dtype=np.uint8).reshape(shape)
    return arr

# Cargar imágenes en escala de grises
cover = cv2.imread('portadora.jpg', cv2.IMREAD_GRAYSCALE)
secret = cv2.imread('secreta.png', cv2.IMREAD_GRAYSCALE)
secret = cv2.resize(secret, (cover.shape[1], cover.shape[0]))

# --- LSB clásico (por si lo necesitas) ---

cover_clean = cover & 0b11111100
secret_bits = secret >> 6
stego_lsb = cover_clean | secret_bits
cv2.imwrite('estego_lsb.png', stego_lsb)  # Guarda la imagen con la secreta oculta
recuperada_lsb = (stego_lsb & 0b00000011) << 6

# Redimensionar la imagen secreta para que quepa en la de cobertura
max_bytes = cover.size * 2 // 8  # 2 bits por píxel, 8 bits por byte
while secret.size > max_bytes:
    secret = cv2.resize(secret, (secret.shape[1] // 2, secret.shape[0] // 2))

# --- TF2D según el enunciado ---
bits = img_to_bits(secret)
max_bits = cover.size * 2  # 2 bits por cada píxel (real e imaginaria)
if len(bits) > max_bits:
    raise ValueError("La imagen secreta es demasiado grande para ocultar en la de cobertura.")

# Transformada de Fourier 2D de la imagen de cobertura
f = np.fft.fft2(cover)
fshift = np.fft.fftshift(f)
real = np.real(fshift)
imag = np.imag(fshift)

# Ocultar los bits en los signos de las componentes reales e imaginarias
idx = 0
for i in range(real.shape[0]):
    for j in range(real.shape[1]):
        if idx >= len(bits):
            break
        # Real
        bit = int(bits[idx])
        real[i, j] = abs(real[i, j]) if bit == 0 else -abs(real[i, j])
        idx += 1
        if idx >= len(bits):
            break
        # Imaginaria
        bit = int(bits[idx])
        imag[i, j] = abs(imag[i, j]) if bit == 0 else -abs(imag[i, j])
        idx += 1
    if idx >= len(bits):
        break

# Reconstruir la imagen estego (Fourier)
f_mod = real + 1j * imag
f_ishift = np.fft.ifftshift(f_mod)
img_stego_fourier = np.fft.ifft2(f_ishift)
img_stego_fourier = np.abs(img_stego_fourier)
img_stego_fourier = np.clip(img_stego_fourier, 0, 255).astype(np.uint8)

# Decodificación: extraer los bits de los signos
f2 = np.fft.fft2(img_stego_fourier)
f2shift = np.fft.fftshift(f2)
real2 = np.real(f2shift)
imag2 = np.imag(f2shift)

bits_rec = ''
idx = 0
for i in range(real2.shape[0]):
    for j in range(real2.shape[1]):
        if idx >= len(bits):
            break
        bits_rec += '0' if real2[i, j] >= 0 else '1'
        idx += 1
        if idx >= len(bits):
            break
        bits_rec += '0' if imag2[i, j] >= 0 else '1'
        idx += 1
    if idx >= len(bits):
        break

# Reconstruir la imagen secreta recuperada
secret_rec = bits_to_img(bits_rec, secret.shape)

# Mostrar resultados
plt.figure(figsize=(12, 8))
plt.subplot(2,3,1)
plt.title("Cobertura")
plt.imshow(cover, cmap='gray')
plt.axis('off')
plt.subplot(2,3,2)
plt.title("Secreta Original")
plt.imshow(secret, cmap='gray')
plt.axis('off')
plt.subplot(2,3,3)
plt.title("Estego LSB")
plt.imshow(stego_lsb, cmap='gray')
plt.axis('off')
plt.subplot(2,3,4)
plt.title("Recuperada LSB")
plt.imshow(recuperada_lsb, cmap='gray')
plt.axis('off')
plt.subplot(2,3,5)
plt.title("Estego Fourier")
plt.imshow(img_stego_fourier, cmap='gray')
plt.axis('off')
plt.subplot(2,3,6)
plt.title("Recuperada Fourier")
plt.imshow(secret_rec, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()

secret_shape = (64, 64)  # Por ejemplo, (64, 64)
num_bits = secret_shape[0] * secret_shape[1] * 8
