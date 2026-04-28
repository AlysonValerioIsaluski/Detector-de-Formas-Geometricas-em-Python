import cv2
import numpy as np
from pathlib import Path as _P
import math

def detectorCirculos(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    altura, largura = img.shape

    # Calcula borda da imagem para eliminar a poluição para a detecção de círculos
    borrada = cv2.GaussianBlur(img, (5, 5), 0, sigmaY=0)
    bordas = cv2.Canny(borrada, 50, 150, apertureSize=5, L2gradient=True)
    
    # Define os raios que possíveis círculos terão na imagem
    raio_min = 6
    raio_max = 30
    num_raios = raio_max - raio_min + 1

    # Define acumulador de votos à partir dos 3 parâmetros do círculo (ponto central (2 dim) e raio)
    acumulador3D = np.zeros((altura, largura, num_raios), dtype=np.float32)

    print("Pré-Calculando tabela de operações trigonométricas...")
    # Implementação do Look Up Table com deslocamentos dos pontos das bordas de cada círculo possível calculados antes da detecção acontecer
    lut_offsets = {}
    
    for raio in range(raio_min, raio_max + 1):
        lut_offsets[raio] = []
        for ang in range(0, 360):
            dx = round(raio * math.cos(math.radians(ang)))
            dy = round(raio * math.sin(math.radians(ang)))
            lut_offsets[raio].append((dx, dy))
        
        lut_offsets[raio] = list(set(lut_offsets[raio]))

    print("Encontrando círculos.....")
    # Passa em cada pixel da imagem das bordas, e vota em óssíveis centros que podem ter gerado aquela borda
    bordas_y, bordas_x = np.nonzero(bordas == 255)

    for idx in range(len(bordas_y)):
        i = bordas_y[idx]
        j = bordas_x[idx]
        for raio in range(raio_min, raio_max+1):
            raio_indice = raio - raio_min
            for dx, dy in lut_offsets[raio]:
                x_centro = j + dx
                y_centro = i + dy
                if x_centro < 0 or x_centro >= largura or y_centro < 0 or y_centro >= altura:
                    continue
                        
                acumulador3D[y_centro, x_centro, raio_indice] += 1

    print("Marcando certroides dos círculos...")
    maxVote = np.max(acumulador3D)
    # Eurístico para determinar valores que representam centros de círculos enquanto não gera falsos positivos
    # Valor MAIS ALTO = círculo deve ser mais perfeito para ser aceito.
    # Valor MAIS BAIXO = tem mais liberdade de aceitação, porém pode haver mais falsos positivos
    resultado = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # Ideal: 0.7 - 0.75
    limiar = maxVote*0.7
    border_min, border_max = -4, 4
    acumulador2D = np.max(acumulador3D, axis=2)

    # Desenha retângulo vermelho em cada círculo na imagem, usando limiar como parâmetro de aceitação
    for i in range(altura):
        for j in range(largura):
            if acumulador2D[i, j] > limiar:
                for x in range(border_min, border_max+1):
                    for y in range(border_min, border_max+1):
                        py = i + y
                        px = j + x
                        if (px < 0 or px >= largura or py < 0 or py >= altura):
                            continue
                        if (x == border_min or x == border_max or y == border_min or y == border_max):
                            resultado[py, px] = (50, 0, 255)

    # Retorna imagem de entrada, com retângulos desenhados ao redor do centro dos círculos encontrados
    return resultado

def detectorElipses(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    return img

ALLOWED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp', '.avif'}

project_root = _P('.').resolve()
folder_path = str(project_root / 'Imagens')

p = _P(folder_path)
n_images = sum(1 for item in _P(folder_path).iterdir() if item.is_file())
# rglob para incluir subpastas; filtra por extensão
files = sorted([f for f in p.rglob('*') if f.suffix.lower() in ALLOWED_EXTS])
results = []
names = []

for i, f in enumerate(files):
    print(f"Processando Imagem {i+1}: {f.name}")
    try:
        img = detectorCirculos(str(f))
    except Exception as e:
        print(f"Ignorado {f}: {e}")
        continue
    
    if img is None:
        # Se a função não conseguiu ler/processar o arquivo, avisa e pula
        print(f"Falha ao processar {f}")
        continue

    names.append(f.name)
    results.append(img)

for i in range(len(results)):
    if results[i] is not None:
        cv2.imwrite(f"Resultados/circulos_{names[i]}", results[i])
        print(f"Resultados da imagem {names[i]} exportados com sucesso!")