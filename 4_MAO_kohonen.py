import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Função SOM simples
# -----------------------
def train_som(data, grid_size=(5,5), epochs=50, lr=0.1, sigma=None):
    """
    data: matriz n x d
    grid_size: tupla (linhas, colunas) da grade
    """
    n, d = data.shape
    rows, cols = grid_size
    m = rows * cols
    
    # Inicializa pesos aleatórios para cada neurônio (d-dimensional)
    W = np.random.randn(m, d)
    
    # Coordenadas da grade (para cálculo de distância no grid)
    neuron_coords = np.array([[i,j] for i in range(rows) for j in range(cols)])
    
    if sigma is None:
        sigma = max(rows, cols) / 2.0
    
    for epoch in range(epochs):
        for x in data:
            # Distâncias Euclidianas entre input e pesos
            distances = np.linalg.norm(W - x, axis=1)
            winner_idx = np.argmin(distances)
            
            # Distância dos outros neurônios ao vencedor na grade
            grid_dist = np.linalg.norm(neuron_coords - neuron_coords[winner_idx], axis=1)
            h = np.exp(-(grid_dist**2) / (2 * sigma**2))  # Função de vizinhança
            
            # Atualiza pesos
            W += lr * h[:, np.newaxis] * (x - W)
    
    return W, neuron_coords

# -----------------------
# Dados de entrada 10D
# -----------------------
np.random.seed(42)
data = np.random.randn(500, 10)  # 500 amostras, 10 dimensões

# -----------------------
# Treina SOM
# -----------------------
weights, coords = train_som(data, grid_size=(5,5), epochs=30, lr=0.1)

# -----------------------
# Visualização da grade 2D (apenas coordenadas da grade)
# -----------------------
plt.figure(figsize=(6,6))
plt.scatter(coords[:,0], coords[:,1], c='red', marker='s', s=200)
plt.title("Grade 2D do SOM (entrada 10D)")
plt.xlabel("Linha da grade")
plt.ylabel("Coluna da grade")
plt.grid(True)
plt.show()
