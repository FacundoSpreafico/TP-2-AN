import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr

# -------------------------
# Funciones de procesamiento
# -------------------------
def cargar_imagen_escala_grises(ruta, tamaño=None):
    """Carga una imagen en escala de grises y opcionalmente la redimensiona"""
    img = Image.open(ruta).convert('L')
    if tamaño is not None:
        img = img.resize(tamaño, Image.LANCZOS)
    return np.array(img)

def ajustar_bit_q(q, bit):
    """Ajusta la paridad de q para almacenar el bit especificado"""
    if (q % 2) != bit:
        q = q + 1 if bit == 1 else q - 1
    return q

# -------------------------
# Funciones de esteganografía
# -------------------------
def insertar_mensaje(portadora, secreta, delta, redundancia=3):
    """
    Inserta un mensaje en el dominio de Fourier con redundancia
    Devuelve:
    - estego: imagen con mensaje oculto (float32)
    - indices: posiciones de los coeficientes modificados
    - magnitudes: magnitudes originales de los coeficientes
    """
    # Binarización del mensaje secreto
    secreta_bin = np.where(secreta > 127, 1, 0)
    bits = secreta_bin.flatten().astype(np.uint8)
    total_bits = bits.size

    # Transformada de Fourier
    fft = np.fft.fft2(portadora.astype(np.float32))
    fft_desplazado = np.fft.fftshift(fft)
    real = np.real(fft_desplazado).astype(np.float32)
    imag = np.imag(fft_desplazado).astype(np.float32)
    h, w = portadora.shape

    # Selección de coeficientes por magnitud (excluyendo DC)
    magnitudes = np.abs(fft_desplazado).flatten()
    indices_planos = np.arange(h * w)
    dc_plano = (h // 2) * w + (w // 2)
    magnitudes[dc_plano] = -1  # Excluir componente DC

    # Ordenar coeficientes y seleccionar los más significativos
    indices_ordenados = indices_planos[np.argsort(-magnitudes)]
    necesarios = total_bits * redundancia
    
    if necesarios > h * w - 1:
        raise RuntimeError(f"Redundancia excesiva. Se necesitan {necesarios} coeficientes.")
    
    seleccionados = indices_ordenados[:necesarios]
    indices = [(idx // w, idx % w) for idx in seleccionados]
    magnitudes_orig = magnitudes[seleccionados].astype(np.float32)

    # Inserción redundante de bits
    for k in range(total_bits):
        bit = bits[k]
        for r in range(redundancia):
            idx = k * redundancia + r
            i, j = indices[idx]

            # Modificar componente real
            a = real[i, j]
            signo_a = np.sign(a)
            q_r = int(round(abs(a) / delta))
            q_r = ajustar_bit_q(q_r, bit)
            real[i, j] = signo_a * (q_r * delta)

            # Modificar componente imaginaria
            b = imag[i, j]
            signo_b = np.sign(b)
            q_i = int(round(abs(b) / delta))
            q_i = ajustar_bit_q(q_i, bit)
            imag[i, j] = signo_b * (q_i * delta)

    # Reconstrucción de la imagen estego
    nuevo_fft = real + 1j * imag
    fft_inverso = np.fft.ifftshift(nuevo_fft)
    estego_complejo = np.fft.ifft2(fft_inverso)
    return np.real(estego_complejo).astype(np.float32), indices, magnitudes_orig

def extraer_mensaje(estego_float, delta, forma_secreta, indices, redundancia=3):
    """Extrae el mensaje oculto usando votación mayoritaria"""
    total_bits = forma_secreta[0] * forma_secreta[1]

    # Transformada de Fourier de la imagen estego
    fft = np.fft.fft2(estego_float)
    fft_desplazado = np.fft.fftshift(fft)
    real_rec = np.real(fft_desplazado)
    imag_rec = np.imag(fft_desplazado)

    # Recuperación con votación mayoritaria
    bits_rec = np.zeros(total_bits, dtype=np.uint8)
    for k in range(total_bits):
        votos = 0
        for r in range(redundancia):
            idx = k * redundancia + r
            i, j = indices[idx]

            # Recuperar bit de componentes real e imaginaria
            q_r = int(round(abs(real_rec[i, j]) / delta))
            q_i = int(round(abs(imag_rec[i, j]) / delta))
            votos += (q_r % 2) + (q_i % 2)

        bits_rec[k] = 1 if votos >= redundancia else 0

    return bits_rec.reshape(forma_secreta) * 255, bits_rec

# -------------------------
# Funciones auxiliares
# -------------------------
def calcular_metricas(original, recuperado, portadora, estego):
    """Calcula precisión de bits y PSNR"""
    original_bin = np.where(original > 127, 1, 0)
    recuperado_bin = np.where(recuperado > 127, 1, 0)
    precision = np.mean(original_bin.flatten() == recuperado_bin.flatten())
    psnr_val = psnr(portadora, estego, data_range=255)
    return precision, psnr_val

def mostrar_resultados(portadora, estego, secreta_rec, precision, psnr_val):
    """Muestra resultados gráficos y métricas"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(portadora, cmap='gray')
    plt.title('Imagen Portadora')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(estego, cmap='gray')
    plt.title(f'Estego (PSNR: {psnr_val:.2f} dB)')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(secreta_rec, cmap='gray')
    plt.title(f'Secreto Recuperado\nPrecisión: {precision*100:.2f}%')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# -------------------------
# Bloque principal
# -------------------------
def main():
    try:
        # Configuración
        DELTA = 10
        REDUNDANCIA = 10
        TAMAÑO_PORTA = (1300, 1300)
        TAMAÑO_SECRETA = (256, 256)
        
        print("=== CARGANDO IMÁGENES ===")
        portadora = cargar_imagen_escala_grises("portadora.png", TAMAÑO_PORTA)
        secreta = cargar_imagen_escala_grises("secreta.png", TAMAÑO_SECRETA)
        
        print("\n=== INSERTANDO MENSAJE ===")
        estego_float, indices, _ = insertar_mensaje(
            portadora, secreta, DELTA, REDUNDANCIA
        )
        estego_uint8 = np.clip(estego_float, 0, 255).astype(np.uint8)
        Image.fromarray(estego_uint8).save("estego.png")
        print("Imagen estego guardada: 'estego.png'")
        
        print("\n=== EXTRAYENDO MENSAJE ===")
        secreta_rec, _ = extraer_mensaje(
            estego_float, DELTA, secreta.shape, indices, REDUNDANCIA
        )
        Image.fromarray(secreta_rec.astype(np.uint8)).save("secreta_recuperada.png")
        print("Imagen secreta recuperada: 'secreta_recuperada.png'")
        
        print("\n=== CALCULANDO MÉTRICAS ===")
        precision, psnr_val = calcular_metricas(secreta, secreta_rec, portadora, estego_uint8)
        print(f"Precisión de bits: {precision*100:.2f}%")
        print(f"PSNR: {psnr_val:.2f} dB")
        
        print("\n=== MOSTRANDO RESULTADOS ===")
        mostrar_resultados(portadora, estego_uint8, secreta_rec, precision, psnr_val)
        
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        raise

if __name__ == "__main__":
    main()