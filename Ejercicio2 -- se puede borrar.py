import numpy as np
from PIL import Image

def fourier_steganography_encode(cover_img_path, secret_img_path, output_path, secret_size=64):
    # Cargar y redimensionar imágenes
    cover_img = np.array(Image.open(cover_img_path).convert('L'))
    secret_img = np.array(Image.open(secret_img_path).convert('L'))
    secret_img = Image.fromarray(secret_img).resize((secret_size, secret_size), Image.LANCZOS)
    secret_img = np.array(secret_img)
    
    # Binarizar la imagen secreta y aplanar (1 bit por píxel)
    secret_img_bin = (secret_img > 127).astype(np.uint8)
    secret_bits = secret_img_bin.flatten()
    required_bits = secret_size * secret_size  # Solo 1 bit por píxel
    
    # FFT de la imagen portadora
    fshift = np.fft.fftshift(np.fft.fft2(cover_img))
    
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
    
    # Verificar capacidad
    available_bits = len(coords)
    if available_bits < required_bits:
        raise ValueError(f"Capacidad insuficiente: Se necesitan {required_bits} bits, solo hay {available_bits} disponibles")
    
    # Insertar bits modificando el signo de la parte real e imaginaria
    modified = fshift.copy()
    for i in range(required_bits):
        r, c = coords[i]
        val = modified[r, c]
        a, b = val.real, val.imag
        bit = secret_bits[i]
        
        if bit == 1:
            a = -abs(a)  # Bit 1: ambos negativos
            b = -abs(b)
        else:
            a = abs(a)   # Bit 0: ambos positivos
            b = abs(b)
        modified[r, c] = complex(a, b)
    
    # Transformada inversa para obtener imagen estego
    stego_img = np.abs(np.fft.ifft2(np.fft.ifftshift(modified)))
    stego_img = np.clip(stego_img, 0, 255).astype(np.uint8)
    
    # Guardar imagen estego
    Image.fromarray(stego_img).save(output_path)
    print(f"[INFO] Imagen estego guardada en: {output_path}")
    return secret_size  # Retornar tamaño para decodificación

def fourier_steganography_decode(stego_img_path, output_path, secret_size):
    # Calcular total de bits necesarios
    total_bits = secret_size * secret_size  # Solo 1 bit por píxel
    
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
    
    # Extraer bits usando el signo de la parte real e imaginaria
    extracted_bits = []
    for i in range(total_bits):
        r, c = coords[i]
        a = fshift[r, c].real
        b = fshift[r, c].imag
        bit = 1 if (a < 0 or b < 0) else 0
        extracted_bits.append(bit)
    
    # Dar forma a la imagen secreta binaria
    secret_img = np.array(extracted_bits[:secret_size * secret_size]).reshape((secret_size, secret_size)).astype(np.uint8)
    # Multiplicar por 255 para que sea visible en blanco y negro
    secret_img = (secret_img * 255).astype(np.uint8)
    
    # Guardar imagen recuperada
    Image.fromarray(secret_img).save(output_path)
    print(f"[INFO] Imagen secreta recuperada guardada en: {output_path}")

if __name__ == "__main__":
    # Configuración
    portadora = "portadora.png"
    secreta = "secreta.png"
    estego = "estego.png"
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
