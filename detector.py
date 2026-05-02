import cv2
import numpy as np
from pathlib import Path as _P
import math
import os

def detectorCirculos(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    nome_arquivo = os.path.basename(img_path)
    altura, largura = img.shape

    # Calcula borda da imagem para eliminar a poluição para a detecção de círculos
    borrada = cv2.GaussianBlur(img, (5, 5), 0, sigmaY=0)
    bordas = cv2.Canny(borrada, 50, 150, apertureSize=5, L2gradient=True)

    cv2.imwrite(f"Resultados/bordas_{nome_arquivo}", bordas)
    
    # Define os raios que possíveis círculos terão na imagem
    raio_min = 5
    raio_max = 50
    num_raios = raio_max - raio_min + 1

    # Define acumulador de votos à partir dos 3 parâmetros do círculo (ponto central (2 dim) e raio)
    print(f"Alocando Matriz 3D ({altura}x{largura}x{num_raios})...")
    acumulador3D = np.zeros((altura, largura, num_raios), dtype=np.float32)

    print("Pré-Calculando tabela de operações trigonométricas...")
    # Implementação do Look Up Table com deslocamentos dos pontos das bordas de cada círculo possível calculados antes da detecção acontecer
    lut_offsets = {}
    
    for raio in range(raio_min, raio_max + 1):
        lut_offsets[raio] = []
        for ang in range(0, 360, 2):
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

    print("Desenhando círculos...")
    maxVote = np.max(acumulador3D)

    acumulador2D = np.max(acumulador3D, axis=2)
    # Eurístico para determinar valores que representam centros de círculos enquanto não gera falsos positivos
    # Valor MAIS ALTO = círculo deve ser mais perfeito para ser aceito.
    # Valor MAIS BAIXO = tem mais liberdade de aceitação, porém pode haver mais falsos positivos
    limiar = maxVote*0.6
    clear_min, clear_max = -15, 15
    resultado = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # Desenha círculo vermelho em cada círculo na imagem, usando limiar como parâmetro de aceitação
    while True:
        py, px = np.unravel_index(np.argmax(acumulador2D), acumulador2D.shape)
        maxVote = acumulador2D[py, px]
        
        if maxVote < limiar:
            break

        idx_raio = np.argmax(acumulador3D[py, px, :])
        raio_real = idx_raio + raio_min

        cv2.circle(resultado, (px, py), raio_real, (50, 0, 255), 2)
        
        # Limpa área de pico global para evitar repetições
        for x in range(clear_min, clear_max+1):
            for y in range(clear_min, clear_max+1):
                cur_x = px + x
                cur_y = py + y
                if (0 <= cur_x < largura and 0 <= cur_y < altura):
                    acumulador2D[cur_y, cur_x] = 0

    # Retorna imagem de entrada, com retângulos desenhados ao redor do centro dos círculos encontrados
    return resultado

def detectorElipses(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # Redução de resolução para otimizar tempo de processamento
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    altura, largura = img.shape[:2]
    
    # Gera a imagem de borda limpa
    borrada = cv2.GaussianBlur(img, (5, 5), 0)
    bordas = cv2.Canny(borrada, 50, 150)

    # Limites estritamente curtos para otimização
    a_min, a_max = 5, 25  # Raio maior (A)
    b_min, b_max = 2, 25  # Raio menor (B)
    passo_ang = 10        # Avalia a rotação da elipse de 10 em 10 graus
    
    num_a = a_max - a_min + 1
    num_b = b_max - b_min + 1
    num_ang = 180 // passo_ang
    
    print(f"Alocando Matriz 5D ({altura}x{largura}x{num_a}x{num_b}x{num_ang})...")
    try:
        # Usa uint8 em vez de float32 para poupar RAM (1 byte por célula em vez de 4)
        acumulador5D = np.zeros((altura, largura, num_a, num_b, num_ang), dtype=np.uint8)
    except MemoryError:
        print("Memória RAM esgotada. Reduza o tamanho da imagem ou os parâmetros da elipse.")
        return img

    print("Pré-Calculando tabela de operações trigonométricas...")
    lut_offsets = {}
    for a in range(a_min, a_max + 1, 2):
        for b in range(b_min, b_max + 1, 2):
            for ang_rot_idx in range(num_ang):
                ang_rot = math.radians(ang_rot_idx * passo_ang)
                chave = (a, b, ang_rot_idx)
                lut_offsets[chave] = []
                
                # Varrer os 360 graus do contorno da elipse
                for t in range(0, 360, 10):
                    rad_t = math.radians(t)
                    # Equações paramétricas da elipse com rotação
                    dx = round(a * math.cos(rad_t) * math.cos(ang_rot) - b * math.sin(rad_t) * math.sin(ang_rot))
                    dy = round(a * math.cos(rad_t) * math.sin(ang_rot) + b * math.sin(rad_t) * math.cos(ang_rot))
                    lut_offsets[chave].append((dx, dy))
                
                # Remove votos sobrepostos causados pelo round()
                lut_offsets[chave] = list(set(lut_offsets[chave]))

    print("Encontrando elipses.....")
    bordas_y, bordas_x = np.nonzero(bordas == 255)
    
    for idx in range(len(bordas_y)):
        i, j = bordas_y[idx], bordas_x[idx]
        for a in range(a_min, a_max + 1, 2):
            idx_a = a - a_min
            for b in range(b_min, b_max + 1, 2):
                idx_b = b - b_min
                for ang_rot_idx in range(num_ang):
                    for dx, dy in lut_offsets[(a, b, ang_rot_idx)]:
                        x_centro = j + dx
                        y_centro = i + dy
                        
                        if 0 <= x_centro < largura and 0 <= y_centro < altura:
                            acumulador5D[y_centro, x_centro, idx_a, idx_b, ang_rot_idx] += 1

    print("Desenhando elipses...")
    # Achata as 3 dimensões extras (raios e ângulo) para buscar apenas as localizações espaciais mais fortes
    acumulador2D = np.max(acumulador5D, axis=(2, 3, 4))
    
    maxVote = np.max(acumulador2D)
    limiar = maxVote * 0.6

    clear_min, clear_max = -15, 15
    
    resultado = cv2.imread(img_path, cv2.IMREAD_COLOR)
    resultado = cv2.resize(resultado, (largura, altura))
    while True:
        py, px = np.unravel_index(np.argmax(acumulador2D), acumulador2D.shape)
        maxVote = acumulador2D[py, px]
        
        if maxVote < limiar:
            break

        # Recupera os índices dos parâmetros (a, b, angulo) com mais votos NESTE centro exato
        fatia_parametros = acumulador5D[py, px]
        idx_a, idx_b, idx_ang = np.unravel_index(np.argmax(fatia_parametros), fatia_parametros.shape)
        
        # Converte os índices do Numpy de volta para os valores geométricos reais da elipse
        raio_a = idx_a + a_min 
        raio_b = idx_b + b_min
        angulo_graus = idx_ang * passo_ang

        # Desenha a elipse com processamento nativo (OpenCV)
        cv2.ellipse(resultado, (px, py), (raio_a, raio_b), angulo_graus, 0, 360, (50, 0, 255), 2)

        # Limpa área de pico global para evitar repetições
        for x in range(clear_min, clear_max+1):
            for y in range(clear_min, clear_max+1):
                cur_x = px + x
                cur_y = py + y
                if (0 <= cur_x < largura and 0 <= cur_y < altura):
                    acumulador2D[cur_y, cur_x] = 0

    return resultado

ALLOWED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp', '.avif'}

project_root = _P('.').resolve()
folder_path_circ = str(project_root / 'ImagensCirculos')
folder_path_elip = str(project_root / 'ImagensElipses')

p_circ = _P(folder_path_circ)
p_elip = _P(folder_path_elip)

n_images_circ = sum(1 for item in _P(folder_path_circ).iterdir() if item.is_file())
n_images_elip = sum(1 for item in _P(folder_path_elip).iterdir() if item.is_file())

# rglob para incluir subpastas; filtra por extensão
files_circ = sorted([f for f in p_circ.rglob('*') if f.suffix.lower() in ALLOWED_EXTS])
files_elip = sorted([f for f in p_elip.rglob('*') if f.suffix.lower() in ALLOWED_EXTS])
results_circ = []
results_elip = []
names_circ = []
names_elip = []

for i, f in enumerate(files_circ):
    print(f"Processando Círculos da Imagem {i+1}: {f.name}")
    try:
        img = detectorCirculos(str(f))
    except Exception as e:
        print(f"Ignorado {f}: {e}")
        continue
    
    if img is None:
        # Se a função não conseguiu ler/processar o arquivo, avisa e pula
        print(f"Falha ao processar {f}")
        continue

    names_circ.append(f.name)
    results_circ.append(img)

for i, f in enumerate(files_elip):
    print(f"Processando Elipses da Imagem {i+1}: {f.name}")
    try:
        img = detectorElipses(str(f))
    except Exception as e:
        print(f"Ignorado {f}: {e}")
        continue
    
    if img is None:
        # Se a função não conseguiu ler/processar o arquivo, avisa e pula
        print(f"Falha ao processar {f}")
        continue

    names_elip.append(f.name)
    results_elip.append(img)

for i in range(len(results_circ)):
    if results_circ[i] is not None:
        cv2.imwrite(f"Resultados/circulos_{names_circ[i]}", results_circ[i])
        print(f"Resultados da imagem {names_circ[i]} exportados com sucesso!")

for i in range(len(results_elip)):
    if results_elip[i] is not None:
        cv2.imwrite(f"Resultados/elipses_{names_elip[i]}", results_elip[i])
        print(f"Resultados da imagem {names_elip[i]} exportados com sucesso!")