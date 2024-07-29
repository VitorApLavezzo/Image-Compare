import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import os

def load_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def resize_images(imageA, imageB):
    h_min = min(imageA.shape[0], imageB.shape[0])
    w_min = min(imageA.shape[1], imageB.shape[1])
    
    imageA_resized = cv2.resize(imageA, (w_min, h_min))
    imageB_resized = cv2.resize(imageB, (w_min, h_min))
    
    return imageA_resized, imageB_resized

def compare_images(imageA, imageB):
    imageA_resized, imageB_resized = resize_images(imageA, imageB)
    
    mse_value = np.sum((imageA_resized.astype("float") - imageB_resized.astype("float")) ** 2)
    mse_value /= float(imageA_resized.shape[0] * imageA_resized.shape[1])
    
    ssim_value, _ = ssim(imageA_resized, imageB_resized, full=True)
    
    return mse_value, ssim_value

def process_csv(csv_file):
    df = pd.read_csv(csv_file)
    threshold = 0.01
    results = []

    for index, row in df.iterrows():
        imageA_path = row['imagem_antiga']
        imageB_path = row['imagem_nova']

        imageA = load_image(imageA_path)
        imageB = load_image(imageB_path)

        if imageA is None or imageB is None:
            continue

        mse_value, ssim_value = compare_images(imageA, imageB)
        results.append([imageA_path, imageB_path, mse_value, ssim_value])

    results_df = pd.DataFrame(results, columns=['imagem_antiga', 'imagem_nova', 'MSE', 'SSIM'])
    return results_df

def main():
    csv_files = [f for f in os.listdir() if f.endswith('.csv')]
    if not csv_files:
        print("Nenhum arquivo CSV encontrado no diretório atual.")
        return
    
    print("Arquivos CSV disponíveis:")
    for i, file in enumerate(csv_files):
        print(f"{i+1}. {file}")
    
    try:
        file_index = int(input("Selecione o número do arquivo CSV que deseja utilizar: ")) - 1
        if file_index < 0 or file_index >= len(csv_files):
            print("Seleção inválida.")
            return
    except ValueError:
        print("Entrada inválida.")
        return
    
    csv_file = csv_files[file_index]
    print(f"Processando {csv_file}...")

    results_df = process_csv(csv_file)

    output_file = 'resultados_comparacao.csv'
    results_df.to_csv(output_file, index=False)
    print(f"Resultados salvos em {output_file}")

if __name__ == "__main__":
    main()
