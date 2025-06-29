import matplotlib.pyplot as plt
import numpy as np

## Aqui eu rodei ambos os códigos, e apenas coloquei os resultados de cada.
cortes = [5, 10, 15, 20, 25]

tempos_execucao_iterativo = [18.2159, 35.0913, 52.3688, 69.5413, 86.6206]

tempos_execucao_recursivo = [19.1805, 37.3538, 56.3538, 74.8667, 91.6232]

plt.plot(cortes, tempos_execucao_iterativo, label='Iterativo', color='blue', linestyle='-', linewidth=2, marker='o')
plt.plot(cortes, tempos_execucao_recursivo, label='Recursivo', color='red', linestyle='--', linewidth=2, marker='o')

plt.title('Comparação do Seam Costure Recursivo e Iterativo')
plt.xlabel('Número de Cortes')
plt.ylabel('Tempos de execução')

# 6. Adicionar Grade e Legenda
plt.grid(True, linestyle=':', alpha=0.7) # Adiciona uma grade fina e semi-transparente
plt.legend() # Mostra a legenda, usando os 'label's definidos em plt.plot()

# 7. Exibir o gráfico
plt.show()