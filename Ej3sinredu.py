from PIL import Image
import numpy as np

def embed_secret_image(carrier_path, secret_path, delta=0.1, output_path="estego.png"):
    carrier = Image.open(carrier_path).convert("L")
    secret = Image.open(secret_path).convert("L")
    secret = secret.resize(carrier.size)

    carrier_array = np.array(carrier, dtype=np.float32)
    secret_array = np.array(secret, dtype=np.float32)

    # FFT de ambas
    F_carrier = np.fft.fft2(carrier_array)
    F_secret = np.fft.fft2(secret_array)

    # Incrustar todo el espectro complejo de la secreta escalado por delta
    F_stego = F_carrier + delta * F_secret

    # Imagen esteganografiada
    stego_array = np.fft.ifft2(F_stego)
    stego_array = np.real(stego_array)  # Parte real, evita artefactos complejos
    stego_img = np.clip(stego_array, 0, 255).astype(np.uint8)

    Image.fromarray(stego_img).save(output_path)
    print(f"[✔] Imagen esteganografiada generada: {output_path}")
    return stego_img


def extract_secret_image(original_path, stego_path, delta=0.1, output_path="secreta_extraida.png"):
    original = Image.open(original_path).convert("L")
    stego = Image.open(stego_path).convert("L")

    original_array = np.array(original, dtype=np.float32)
    stego_array = np.array(stego, dtype=np.float32)

    # FFT de ambas
    F_original = np.fft.fft2(original_array)
    F_stego = np.fft.fft2(stego_array)

    # Extraer el espectro de la secreta
    F_secret_est = (F_stego - F_original) / delta

    # Inversa para recuperar la secreta
    secret_array = np.fft.ifft2(F_secret_est)
    secret_array = np.real(secret_array)
    secret_img = np.clip(secret_array, 0, 255).astype(np.uint8)

    Image.fromarray(secret_img).save(output_path)
    print(f"[✔] Imagen secreta extraída: {output_path}")
    return secret_img


if __name__ == "__main__":
    portadora = "portadora.png"
    secreta = "secreta.png"
    estego_output = "estego.png"
    secreta_extraida = "secreta_recuperada.png"
    delta = 0.05                                  # Puedes probar con valores más pequeños si hay distorsión

    try:
        print("=== Codificación ===")
        embed_secret_image(portadora, secreta, delta=delta, output_path=estego_output)

        print("\n=== Decodificación ===")
        extract_secret_image(portadora, estego_output, delta=delta, output_path=secreta_extraida)
    except Exception as e:
        print(f"[ERROR] {str(e)}")
