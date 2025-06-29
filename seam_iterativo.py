import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt

# As funções de cálculo de energia (calculate_pixel_energy) e remoção de costura (remove_seam)
# são as mesmas da implementação Top-Down, pois são utilitários.

def calculate_pixel_energy(image_array, r, c, M, N):
    '''
    Calcula a medida de perturbação (energia) para o pixel (r, c).
    '''
    pixel_left = image_array[r, c-1].astype(float) if c > 0 else image_array[r, c].astype(float)
    pixel_right = image_array[r, c+1].astype(float) if c < N - 1 else image_array[r, c].astype(float)
    pixel_up = image_array[r-1, c].astype(float) if r > 0 else image_array[r, c].astype(float)
    pixel_down = image_array[r+1, c].astype(float) if r < M - 1 else image_array[r, c].astype(float)

    delta_x_sq = np.sum((pixel_left - pixel_right)**2)
    delta_y_sq = np.sum((pixel_up - pixel_down)**2)

    energy = delta_x_sq + delta_y_sq
    
    return energy

def remove_seam(image_array, seam):
    '''
    Cria uma nova imagem removendo os pixels da costura encontrada.
    '''
    M, N, _ = image_array.shape
    new_image_array = np.zeros((M, N - 1, 3), dtype=np.uint8) 

    seam_pixels_set = set(seam)

    for r in range(M):
        pixels_to_keep = []
        for c in range(N):
            if (r, c) not in seam_pixels_set:
                pixels_to_keep.append(image_array[r, c, :])
        
        # Isso garante que a linha tem o tamanho correto.
        new_image_array[r, :, :] = np.array(pixels_to_keep, dtype=np.uint8)
        
    return new_image_array

# --- Algoritmo de Programação Dinâmica Iterativo (Bottom-Up) ---

def find_optimal_seam_bottom_up_single_pass(image_data):
    '''
    Encontra UMA costura vertical de menor perturbação na imagem usando DP Iterativo (Bottom-Up).
    '''
    M, N, _ = image_data.shape

    # 1. Pré-calcular o mapa de energia para a imagem atual
    energy_map = create_energy_map(image_data, M, N) # Nova função para clareza

    # Matriz DP para armazenar os custos acumulados mínimos
    cost_matrix = np.zeros((M, N), dtype=float)
    # Matriz para armazenar o caminho (qual coluna veio da linha acima)
    # Valores: -1 (esquerda), 0 (centro), 1 (direita)
    path_matrix = np.zeros((M, N), dtype=int) 

    # 2. Inicializar a primeira linha da matriz de custos
    # O custo acumulado para a primeira linha é a própria energia do pixel
    cost_matrix[0, :] = energy_map[0, :]

    # 3. Preencher a matriz de custos (DP) de cima para baixo
    for r in range(1, M): # Começa da segunda linha (índice 1)
        for c in range(N):
            # Encontrar os custos acumulados dos três vizinhos possíveis na linha anterior
            # Lidar com as bordas da imagem:
            cost_left = cost_matrix[r-1, c-1] if c > 0 else float('inf')
            cost_center = cost_matrix[r-1, c]
            cost_right = cost_matrix[r-1, c+1] if c < N - 1 else float('inf')

            # Encontrar o mínimo e determinar de onde ele veio
            min_prev_cost = min(cost_left, cost_center, cost_right)
            
            # Atualizar a matriz de caminhos
            if min_prev_cost == cost_left:
                path_matrix[r, c] = c - 1 # Veio do vizinho esquerdo
            elif min_prev_cost == cost_center:
                path_matrix[r, c] = c # Veio do vizinho central
            else: # min_prev_cost == cost_right
                path_matrix[r, c] = c + 1 # Veio do vizinho direito
            
            # Calcular o custo acumulado para o pixel atual
            cost_matrix[r, c] = energy_map[r, c] + min_prev_cost
            
    # 4. Encontrar o custo mínimo na última linha da matriz de custos
    min_total_cost = np.min(cost_matrix[M-1, :])
    # Encontrar a coluna do pixel que deu o custo mínimo na última linha
    last_col = np.argmin(cost_matrix[M-1, :])

    # 5. Reconstruir a costura de baixo para cima usando path_matrix
    seam = []
    current_col = last_col
    for r in range(M - 1, -1, -1): # Itera da última linha até a primeira
        seam.append((r, current_col))
        if r > 0: # Para a linha 0, não há pixel anterior na costura
            current_col = path_matrix[r, current_col]
            
    seam.reverse() # Inverter para que a costura esteja do topo para baixo
    
    return min_total_cost, seam

def create_energy_map(image_array, M, N):
    energy_map_result = np.zeros((M, N), dtype=float)
    for r in range(M):
        for c in range(N):
            energy_map_result[r, c] = calculate_pixel_energy(image_array, r, c, M, N)
    return energy_map_result

if __name__ == "__main__":
    image_path = "imagem.jpg" # Sua imagem da praia
    num_seams_to_remove = [5, 10, 15, 20, 25]
    tempos_execucao = []
    img_pil_original = Image.open(image_path).convert("RGB")
    current_image_data = np.array(img_pil_original)
    
    original_M, original_N, _ = current_image_data.shape
    print(f"Imagem '{image_path}' carregada. Dimensões originais: {original_M}x{original_N}")
    
    img_pil_original.save("original_image_bottom_up_test.png")
    print("Imagem original salva como 'original_image_bottom_up_test.png'")

    for i in num_seams_to_remove:
        current_image_data = np.array(img_pil_original)
        total_start_time = time.perf_counter()
        print(f"\nIniciando remoção de {i} costuras (Bottom-Up)...")
        for j in range(i):
            print(f"  Removendo costura {j + 1}/{i}. Dimensão atual: {current_image_data.shape[0]}x{current_image_data.shape[1]}")
            
            # A função de encontrar costura iterativa
            cost, seam = find_optimal_seam_bottom_up_single_pass(current_image_data)
            
            current_image_data = remove_seam(current_image_data, seam)
            
        total_end_time = time.perf_counter()
        tempos_execucao.append(total_end_time - total_start_time)

j = 0
for i in tempos_execucao:
    print(f"tempos de execução iterativo numero de cortes {num_seams_to_remove[j]}: {i:.6}")
    j += 1