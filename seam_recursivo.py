import numpy as np
from PIL import Image
import sys
import time

# Aumentar o limite de recursão padrão para lidar com imagens grandes
sys.setrecursionlimit(20000) 

# --- Funções de Cálculo de Energia ---

def calculate_pixel_energy(image_array, r, c, M, N):
    '''
    Calcula a medida de perturbação (energia) para o pixel (r, c)
    baseado em seus vizinhos horizontais e verticais, conforme a fórmula do problema.
    '''
    # Garantir que os valores dos pixels sejam tratados como float para evitar overflow
    
    # Vizinho esquerdo (A[r, c-1])
    pixel_left = image_array[r, c-1].astype(float) if c > 0 else image_array[r, c].astype(float)
    # Vizinho direito (A[r, c+1])
    pixel_right = image_array[r, c+1].astype(float) if c < N - 1 else image_array[r, c].astype(float)

    # Vizinho superior (A[r-1, c])
    pixel_up = image_array[r-1, c].astype(float) if r > 0 else image_array[r, c].astype(float)
    # Vizinho inferior (A[r+1, c])
    pixel_down = image_array[r+1, c].astype(float) if r < M - 1 else image_array[r, c].astype(float)

    # Calcular a diferença horizontal ao quadrado (sum of squared differences for RGB)
    delta_x_sq = np.sum((pixel_left - pixel_right)**2)

    # Calcular a diferença vertical ao quadrado
    delta_y_sq = np.sum((pixel_up - pixel_down)**2)

    energy = delta_x_sq + delta_y_sq
    
    return energy

# --- Variáveis Globais para as Funções DP (redefinidas a cada iteração no loop principal) ---
current_image_array = None
current_M = 0
current_N = 0
energy_map = None 

memo_custo = {}
memo_path = {}

# --- Funções DP Top-Down (adaptadas para serem reentrantes) ---

def get_precalculated_pixel_energy(linha, coluna):
    '''Retorna a energia do pixel do mapa de energia pré-calculado.'''
    # current_M e current_N são atualizados antes de cada chamada no loop
    if not (0 <= linha < current_M and 0 <= coluna < current_N):
        raise IndexError(f"Acesso inválido ao energy_map ({linha}, {coluna})")
    return energy_map[linha, coluna]

def calculate_min_accumulated_cost_recursive(linha, coluna):
    '''
    Calcula a menor perturbação acumulada de uma costura vertical que termina no pixel (linha, coluna).
    Utiliza memoização para evitar recálculos.
    '''
    # current_N é atualizado antes de cada chamada no loop
    if not (0 <= coluna < current_N):
        return float('inf')
    
    if linha == 0:
        return get_precalculated_pixel_energy(linha, coluna)
    
    if (linha, coluna) in memo_custo:
        return memo_custo[(linha, coluna)]
    
    d_atual = get_precalculated_pixel_energy(linha, coluna)

    custo_prev_esq = calculate_min_accumulated_cost_recursive(linha - 1, coluna - 1)
    custo_prev_centro = calculate_min_accumulated_cost_recursive(linha - 1, coluna)
    custo_prev_dir = calculate_min_accumulated_cost_recursive(linha - 1, coluna + 1)
    
    minimo_pertubacao_prev = min(custo_prev_esq, custo_prev_centro, custo_prev_dir)

    if minimo_pertubacao_prev == custo_prev_centro:
        memo_path[(linha, coluna)] = coluna
    elif minimo_pertubacao_prev == custo_prev_esq:
        memo_path[(linha, coluna)] = coluna - 1
    else: 
        memo_path[(linha, coluna)] = coluna + 1
            
    memo_custo[(linha, coluna)] = d_atual + minimo_pertubacao_prev
    
    return memo_custo[(linha, coluna)]

def find_optimal_seam_top_down_single_pass(image_data_input):
    '''
    Encontra UMA costura vertical de menor perturbação na imagem usando DP Top-Down.
    Esta função agora recebe image_data como argumento e não usa globais para M, N diretamente.
    '''
    global current_image_array, current_M, current_N, energy_map, memo_custo, memo_path
    
    current_image_array = image_data_input # Use a imagem passada para esta iteração
    current_M, current_N, _ = image_data_input.shape

    # Resetar memoização para cada execução da função (para uma nova costura)
    memo_custo = {} 
    memo_path = {}
    
    # 1. Pré-calcular o mapa de energia para a imagem atualizada
    energy_map = np.zeros((current_M, current_N), dtype=float)
    for r in range(current_M):
        for c in range(current_N):
            energy_map[r, c] = calculate_pixel_energy(current_image_array, r, c, current_M, current_N)

    menor_pertubacao_total = float('inf')
    ultima_coluna_da_costura = -1

    # 2. Calcular o custo acumulado para cada pixel na última linha
    for j in range(current_N):
        pert_atual = calculate_min_accumulated_cost_recursive(current_M - 1, j)
        if pert_atual < menor_pertubacao_total:
            menor_pertubacao_total = pert_atual
            ultima_coluna_da_costura = j
            
    # 3. Reconstruir a costura
    seam = []
    current_col = ultima_coluna_da_costura
    for r in range(current_M - 1, -1, -1):
        seam.append((r, current_col))
        if r > 0: 
            current_col = memo_path[(r, current_col)]
            
    seam.reverse() 
    
    return menor_pertubacao_total, seam

# --- Função para remover a costura da imagem ---
def remove_seam(image_array, seam):
    '''
    Cria uma nova imagem removendo os pixels da costura encontrada.
    '''
    M, N, _ = image_array.shape
    new_image_array = np.zeros((M, N - 1, 3), dtype=np.uint8) 

    # Criar um conjunto para busca rápida dos pixels da costura
    seam_pixels_set = set(seam)

    for r in range(M):
        pixels_to_keep = []
        for c in range(N):
            if (r, c) not in seam_pixels_set:
                pixels_to_keep.append(image_array[r, c, :])
        
        # Certifique-se de que temos o número correto de pixels para a nova linha
        if len(pixels_to_keep) != N - 1:
            # Isso não deve acontecer se a costura for válida e tiver exatamente um pixel por linha
            raise ValueError(f"Número incorreto de pixels para manter na linha {r} após remoção da costura.")
            
        new_image_array[r, :, :] = np.array(pixels_to_keep, dtype=np.uint8)
        
    return new_image_array

# --- Bloco Principal para Carregar e Processar a Imagem ---

if __name__ == "__main__":
    # --- 1. ESCOLHA SUA IMAGEM AQUI ---
    image_path = "imagem.jpg" # Sua imagem da praia

    # --- 2. DEFINA QUANTAS COLUNAS DESEJA REMOVER ---
    # Experimente com um número pequeno primeiro, tipo 50 ou 100, para ver o efeito.
    num_seams_to_remove = 15 # Remova 100 colunas

    # Carrega a imagem original
    img_pil_original = Image.open(image_path).convert("RGB")
    current_image_data = np.array(img_pil_original)
    
    original_M, original_N, _ = current_image_data.shape
    print(f"Imagem '{image_path}' carregada. Dimensões originais: {original_M}x{original_N}")
    
    # Salva a imagem original para comparação
    img_pil_original.save("original_image_multiple_seams.png")
    print("Imagem original salva como 'original_image_multiple_seams.png'")

    total_start_time = time.perf_counter()
    
    # --- Loop para Remover Múltiplas Costuras ---
    for i in range(num_seams_to_remove):
        print(f"\nRemovendo costura {i + 1}/{num_seams_to_remove}...")
        
        # A função de encontrar costura agora recebe a imagem atualizada
        cost, seam = find_optimal_seam_top_down_single_pass(current_image_data)
        
        # Atualiza a imagem para a próxima iteração
        current_image_data = remove_seam(current_image_data, seam)
        
        print(f"  Custo da costura: {cost:.2f}. Nova dimensão: {current_image_data.shape[0]}x{current_image_data.shape[1]}")

    total_end_time = time.perf_counter()

    print(f"\n--- Processo de Seam Carving Concluído ---")
    print(f"Total de {num_seams_to_remove} costuras removidas.")
    print(f"Tempo total de execução: {total_end_time - total_start_time:.6f} segundos")
    print(f"Dimensões finais da imagem: {current_image_data.shape[0]}x{current_image_data.shape[1]}")

    # Salvar a imagem final redimensionada
    final_img_pil = Image.fromarray(current_image_data)
    final_img_pil.save(f"carved_image_top_down_final_{num_seams_to_remove}_seams.png")
    print(f"Imagem final redimensionada salva como 'carved_image_top_down_final_{num_seams_to_remove}_seams.png'")
