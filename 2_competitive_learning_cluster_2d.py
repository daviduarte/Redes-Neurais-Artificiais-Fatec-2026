import numpy as np
import matplotlib.pyplot as plt

# ----- Visualização -----
def visualize_clusters(X, W):
    plt.figure(figsize=(8,6))
    plt.scatter(X[:,0], X[:,1], alpha=0.3, label="Dados")
    plt.scatter(W[:,0], W[:,1], c="red", marker="x", s=200, label="Centroides")
    plt.title("Clusters aprendidos")
    plt.legend()
    plt.show()

# ----- Dados artificiais -----
np.random.seed(42)
#      shape (n_samples, n_features)
data1 = np.random.randn(100, 2) + np.array([2, 2])
data2 = np.random.randn(100, 2) + np.array([-2, -2])
data3 = np.random.randn(100, 2) + np.array([2, -2])
X = np.vstack([data1, data2, data3])

# ----- Parâmetros da rede -----
n_clusters = 3   # número de "neurônios vencedores"
n_features = X.shape[1]
eta = 0.01        # taxa de aprendizado
epochs = 20

# ----- Inicialização dos pesos -----
W = np.random.randn(n_clusters, n_features)

# ----- Treinamento -----
for epoch in range(epochs):
    for x in X:
        # Calcula distâncias euclidianas de x a cada neurônio
        distances = np.linalg.norm(W - x, axis=1)
        # Índice do neurônio vencedor (menor distância)
        winner = np.argmin(distances)
        # Atualiza apenas o neurônio vencedor
        W[winner] += eta * (x - W[winner])
    # Visualização a cada época
    visualize_clusters(X, W)
    

