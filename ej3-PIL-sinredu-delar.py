import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr

class Constantes:
    """Clase para almacenar constantes de configuración"""
    # Tamaños de imágenes
    TAMANIO_PORTADORA = (1300, 1300)
    TAMANIO_SECRETA = (256, 256)
    
    # Parámetros de esteganografía
    DELTA = 10  # Factor de escalamiento para inserción
    UMBRAL_BINARIZACION = 127  # Umbral para binarizar la imagen secreta
    
    # Rutas de archivos
    RUTA_PORTADORA = "portadora.png"
    RUTA_SECRETA = "secreta.png"
    RUTA_ESTEGO = "estego.png"
    RUTA_SECRETA_RECUPERADA = "secreta_recuperada.png"

def cargar_imagen(ruta, tamaño=None):
    """
    Carga una imagen en escala de grises y la redimensiona.
    
    Args:
        ruta: Ruta del archivo de imagen
        tamaño: Tupla (ancho, alto) para redimensionar (opcional)
    
    Returns:
        Array numpy con la imagen en escala de grises
    """
    img = Image.open(ruta).convert('L')
    if tamaño:
        img = img.resize(tamaño, Image.LANCZOS)
    return np.array(img)

def ajustar_paridad_q(q, bit):
    """
    Ajusta la paridad de q para almacenar un bit.
    
    Args:
        q: Valor a modificar
        bit: Bit a almacenar (0 o 1)
    
    Returns:
        Valor modificado con la paridad deseada
    """
    if (q % 2) != bit:
        return q + 1 if bit == 1 else q - 1
    return q

def incrustar_mensaje(imagen_portadora, imagen_secreta, delta):
    """
    Incrusta un mensaje secreto en el dominio de Fourier de la imagen portadora.
    
    Proceso:
    1. Binariza la imagen secreta
    2. Calcula la FFT de la portadora
    3. Selecciona coeficientes de alta frecuencia
    4. Modifica componentes real e imaginaria para almacenar bits
    
    Args:
        imagen_portadora: Array numpy con imagen portadora
        imagen_secreta: Array numpy con imagen secreta
        delta: Factor de escalamiento para inserción
    
    Returns:
        stego_float: Imagen estego en flotante
        indices: Posiciones de coeficientes modificados
    """
    # 1. Binarizar imagen secreta
    bits_secretos = np.where(imagen_secreta > Constantes.UMBRAL_BINARIZACION, 1, 0)
    bits_planos = bits_secretos.flatten()
    total_bits = bits_planos.size
    
    # 2. Transformada de Fourier
    fft = np.fft.fft2(imagen_portadora.astype(np.float32))
    fft_desplazado = np.fft.fftshift(fft)
    parte_real = np.real(fft_desplazado)
    parte_imag = np.imag(fft_desplazado)
    alto, ancho = imagen_portadora.shape
    
    # 3. Seleccionar coeficientes (excluyendo componente DC)
    magnitud = np.abs(fft_desplazado).flatten()
    indices_planos = np.arange(alto * ancho)
    pos_dc = (alto // 2) * ancho + (ancho // 2)
    magnitud[pos_dc] = -1  # Excluir componente DC
    
    # Ordenar de mayor a menor magnitud
    indices_ordenados = indices_planos[np.argsort(-magnitud)]
    coeficientes_seleccionados = indices_ordenados[:total_bits]
    indices = [(idx // ancho, idx % ancho) for idx in coeficientes_seleccionados]
    
    # 4. Modificar coeficientes para almacenar bits
    for k, bit in enumerate(bits_planos):
        i, j = indices[k]
        
        # Modificar componente real
        valor_real = parte_real[i, j]
        signo_real = np.sign(valor_real)
        q_real = int(round(abs(valor_real) / delta))
        q_real = ajustar_paridad_q(q_real, bit)
        parte_real[i, j] = signo_real * (q_real * delta)
        
        # Modificar componente imaginario
        valor_imag = parte_imag[i, j]
        signo_imag = np.sign(valor_imag)
        q_imag = int(round(abs(valor_imag) / delta))
        q_imag = ajustar_paridad_q(q_imag, bit)
        parte_imag[i, j] = signo_imag * (q_imag * delta)
    
    # 5. Reconstruir imagen estego
    fft_modificado = parte_real + 1j * parte_imag
    fft_inverso = np.fft.ifftshift(fft_modificado)
    imagen_estego = np.fft.ifft2(fft_inverso)
    return np.real(imagen_estego).astype(np.float32), indices

def extraer_mensaje(imagen_estego, delta, forma_secreta, indices):
    """
    Extrae el mensaje secreto de la imagen estego.
    
    Proceso:
    1. Calcula la FFT de la imagen estego
    2. Para cada coeficiente modificado:
        a. Obtiene componentes real e imaginario
        b. Usa el componente con mayor magnitud para extraer el bit
    
    Args:
        imagen_estego: Imagen con mensaje oculto
        delta: Factor de escalamiento usado en inserción
        forma_secreta: Dimensiones originales de la imagen secreta
        indices: Posiciones de coeficientes modificados
    
    Returns:
        Imagen secreta recuperada
        Array de bits recuperados
    """
    total_bits = forma_secreta[0] * forma_secreta[1]
    
    # 1. Transformada de Fourier
    fft = np.fft.fft2(imagen_estego)
    fft_desplazado = np.fft.fftshift(fft)
    parte_real = np.real(fft_desplazado)
    parte_imag = np.imag(fft_desplazado)
    
    # 2. Recuperar bits
    bits_recuperados = np.zeros(total_bits, dtype=np.uint8)
    
    for k in range(total_bits):
        i, j = indices[k]
        real = parte_real[i, j]
        imag = parte_imag[i, j]
        
        # Usar el componente con mayor magnitud (más robusto)
        if abs(real) > abs(imag):
            q = int(round(abs(real) / delta))
            bit = q % 2
        else:
            q = int(round(abs(imag) / delta))
            bit = q % 2
            
        bits_recuperados[k] = bit
    
    # Convertir bits a imagen
    imagen_recuperada = bits_recuperados.reshape(forma_secreta) * 255
    return imagen_recuperada, bits_recuperados

def mostrar_resultados(portadora, estego, secreta_recuperada, precision, psnr_val):
    """Muestra las imágenes y métricas de resultados"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(portadora, cmap='gray')
    plt.title('Imagen Portadora')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(estego, cmap='gray')
    plt.title(f'Imagen Estego (δ={Constantes.DELTA})\nPSNR: {psnr_val:.2f} dB')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(secreta_recuperada, cmap='gray')
    plt.title(f'Secreto Recuperado\nPrecisión: {precision:.2%}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def ejecutar_esteganografia():
    """Función principal que ejecuta todo el proceso de esteganografía"""
    try:
        print("=== CARGANDO IMÁGENES ===")
        portadora = cargar_imagen(
            Constantes.RUTA_PORTADORA, 
            Constantes.TAMANIO_PORTADORA
        )
        secreta = cargar_imagen(
            Constantes.RUTA_SECRETA, 
            Constantes.TAMANIO_SECRETA
        )
        print(f"Portadora: {portadora.shape} | Secreta: {secreta.shape}")
        
        print("\n=== INCORPORANDO MENSAJE ===")
        estego_float, indices = incrustar_mensaje(
            portadora, 
            secreta, 
            Constantes.DELTA
        )
        estego_uint8 = np.clip(estego_float, 0, 255).astype(np.uint8)
        Image.fromarray(estego_uint8).save(Constantes.RUTA_ESTEGO)
        print(f"[✓] Mensaje incorporado. Imagen guardada en '{Constantes.RUTA_ESTEGO}'")
        
        print("\n=== EXTRAYENDO MENSAJE ===")
        secreta_recuperada, bits_rec = extraer_mensaje(
            estego_float,
            Constantes.DELTA,
            secreta.shape,
            indices
        )
        Image.fromarray(secreta_recuperada.astype(np.uint8)).save(
            Constantes.RUTA_SECRETA_RECUPERADA
        )
        print(f"[✓] Mensaje extraído. Imagen guardada en '{Constantes.RUTA_SECRETA_RECUPERADA}'")
        
        print("\n=== EVALUANDO RESULTADOS ===")
        bits_originales = np.where(secreta > Constantes.UMBRAL_BINARIZACION, 1, 0).flatten()
        precision = np.mean(bits_originales == bits_rec)
        psnr_val = psnr(portadora, estego_uint8, data_range=255)
        
        print(f"Precisión de bits: {precision:.2%}")
        print(f"Calidad (PSNR): {psnr_val:.2f} dB")
        
        mostrar_resultados(portadora, estego_uint8, secreta_recuperada, precision, psnr_val)
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")

if __name__ == "__main__":
    ejecutar_esteganografia()