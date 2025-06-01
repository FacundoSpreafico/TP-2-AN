def fourier_steganography_encode(cover_img_path, secret_img_path, output_path):
    cover_img = np.array(Image.open(cover_img_path).convert('L'))
    secret_img = np.array(Image.open(secret_img_path).convert('L'))
    secret_bits = (secret_img > 127).astype(np.uint8).flatten()
    
    fshift = np.fft.fftshift(np.fft.fft2(cover_img))
    rows, cols = fshift.shape
    center_row, center_col = rows // 2, cols // 2
    radius = min(center_row, center_col)
    
    # Generar coordenadas solo en semiplano superior (incluyendo eje central)
    coords = []
    for r in range(center_row + 1):  # Recorrer hasta la fila central
        for c in range(cols):
            dist = np.sqrt((r - center_row)**2 + (c - center_col)**2)
            if dist < radius:
                # Para la fila central, tomar solo mitad izquierda
                if r < center_row or (r == center_row and c <= center_col):
                    coords.append((r, c))
    
    modified = fshift.copy()
    for i in range(min(len(secret_bits), len(coords))):
        r, c = coords[i]
        r_conj = 2 * center_row - r
        c_conj = 2 * center_col - c
        
        val = modified[r, c]
        a, b = val.real, val.imag
        bit = secret_bits[i]
        
        # Modificar solo componente real
        new_real = -abs(a) if bit == 1 else abs(a)
        
        # Actualizar coeficiente y su conjugado
        modified[r, c] = complex(new_real, b)
        if (r, c) != (r_conj, c_conj):  # Evitar centro
            modified[r_conj, c_conj] = complex(new_real, -b)
    
    # Reconstruir imagen
    stego_img = np.real(np.fft.ifft2(np.fft.ifftshift(modified)))
    stego_img = np.clip(stego_img, 0, 255).astype(np.uint8)
    Image.fromarray(stego_img).save(output_path)

def fourier_steganography_decode(stego_img_path, output_path, secret_size):
    stego_img = np.array(Image.open(stego_img_path).convert('L'))
    fshift = np.fft.fftshift(np.fft.fft2(stego_img))
    rows, cols = fshift.shape
    center_row, center_col = rows // 2, cols // 2
    radius = min(center_row, center_col)
    total_pixels = secret_size * secret_size
    
    # Misma región que en codificación
    coords = []
    for r in range(center_row + 1):
        for c in range(cols):
            dist = np.sqrt((r - center_row)**2 + (c - center_col)**2)
            if dist < radius:
                if r < center_row or (r == center_row and c <= center_col):
                    coords.append((r, c))
    
    extracted_bits = []
    for i in range(min(total_pixels, len(coords))):
        r, c = coords[i]
        a = fshift[r, c].real
        extracted_bits.append(1 if a < 0 else 0)
    
    # Rellenar con ceros si es necesario
    extracted_bits += [0] * (total_pixels - len(extracted_bits))
    secret_img = np.array(extracted_bits[:total_pixels]).reshape((secret_size, secret_size))
    secret_img = (secret_img * 255).astype(np.uint8)
    Image.fromarray(secret_img).save(output_path)