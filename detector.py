import cv2
import numpy as np
from pathlib import Path as _P
import math

def detectorCirculos(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    altura, largura = img.shape

    borrada = cv2.GaussianBlur(img, (5, 5), 0, sigmaY=0)

    bordas = cv2.Canny(borrada, 50, 150, apertureSize=5, L2gradient=True)
    
    nRaios = 30
    acumuladores = []
    for i in range(nRaios):
        acumuladores.append(np.zeros((altura, largura), dtype=np.float32))


    ultimoX, ultimoY, maxVote = -1, -1, -1

    for i in range(altura):
        for j in range(largura):
            if bordas[i, j] == 255:
                print(f"centro: ({j},{i})")
                for raio in range(1, nRaios+1):
                    for ang in range (0, 180):
                        xLength = math.floor(raio * math.cos(ang))
                        yLength = math.floor(raio * math.sin(ang))
                        x = j + xLength
                        y = i + yLength
                        if x == ultimoX and y == ultimoY or x < 0 or x >= largura or y < 0 or y >= altura:
                            continue
                        ultimoX = x
                        ultimoY = y
                        if bordas[y, x] == 255:
                            acumuladores[raio-1][i, j] += 1
                    if acumuladores[raio-1][i, j] > maxVote:
                        maxVote = acumuladores[raio-1][i, j]
                    
                    ultimoX, ultimoY = -1, -1

    print(f"\nNormalizando Acumuladores...")

    div = maxVote / 255

    for i in range(altura):
        for j in range(largura):
            if bordas[i, j] == 255:
                for raio in range(1, nRaios+1):
                    acumuladores[raio-1][i, j] = math.floor(acumuladores[raio-1][i, j] / div)

    return acumuladores

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
    
    img = detectorCirculos(str(f))
    '''
    try:
    except Exception as e:
        print(f"Ignorado {f}: {e}")
        continue
    '''
    if img is None:
        # Se a função não conseguiu ler/processar o arquivo, avisa e pula
        print(f"Falha ao processar {f}")
        continue

    names.append(f.name)
    results.append(img)

for i in range(n_images):
    if results[i] is not None:
        # Exporta imagens
        for j, acumulador in enumerate(results[i]):
            cv2.imwrite(f"Resultados/img_{i+1}_raio_{j+1}.png", acumulador)
        print(f"Resultados da imagem {names[i]} exportados com sucesso!")