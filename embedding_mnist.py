"""
Carrega o checkpoint da CNN treinada no MNIST, extrai embeddings de
1 imagem de referência + 10 imagens candidatas e ordena as candidatas
pela similaridade com a referência.

Embedding = saída da camada fc1 (vetor de 128 dimensões), ou seja,
a representação logo antes do classificador final (fc2).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms

CHECKPOINT = "checkpoints/cnn_mnist_melhor.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor()  # mesmo pré-processamento usado no treino
])

# ============================
# 1. Modelo (mesma arquitetura do treino)
# ============================
# A classe foi copiada aqui porque importar treino_mnist.py executaria o
# download/divisão do dataset no nível do módulo. Se mudar a arquitetura
# no treino, mude aqui também, senão o load_state_dict falha.
class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def extrair_embedding(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))  # (B, 128)
        return x

    def forward(self, x):
        return self.fc2(self.extrair_embedding(x))

# ============================
# 2. Carregar checkpoint
# ============================
def carregar_modelo(caminho=CHECKPOINT):
    ckpt = torch.load(caminho, map_location=DEVICE)
    model = CNNMnist().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Checkpoint carregado: época {ckpt['epoch']+1}, "
          f"acurácia val {ckpt['val_acc']*100:.2f}%")
    return model

# ============================
# 3. Imagens -> tensores
# ============================
def carregar_imagem(caminho):
    """Carrega uma imagem de arquivo no formato esperado pela rede (1,28,28).
    Obs.: o MNIST tem dígito branco em fundo preto. Se suas imagens forem
    dígito preto em fundo branco, inverta com PIL.ImageOps.invert(img)."""
    img = Image.open(caminho).convert("L").resize((28, 28))
    return transform(img)

# ============================
# 4. Embeddings
# ============================
@torch.no_grad()
def calcular_embeddings(model, imagens):
    """imagens: tensor (N,1,28,28) ou lista de tensores (1,28,28).
    Retorna tensor (N,128)."""
    if isinstance(imagens, (list, tuple)):
        imagens = torch.stack(imagens)
    return model.extrair_embedding(imagens.to(DEVICE)).cpu()

# ============================
# 5. Similaridade e ranking
# ============================
def ordenar_por_similaridade(emb_referencia, emb_candidatas, metrica="cosseno"):
    """
    emb_referencia: (128,)   emb_candidatas: (N,128)
    Retorna (indices_ordenados, scores_ordenados), do mais parecido
    para o menos parecido.

    metrica:
      - "produto_escalar": <a, b>. Sensível à norma dos vetores: um
        embedding com norma grande tende a "ganhar" de todos.
      - "cosseno": produto escalar entre vetores normalizados (L2).
        Considera só a direção; costuma ser a melhor escolha.
      - "euclidiana": distância L2 (menor = mais parecido); o score
        retornado é a distância negativa para manter "maior = melhor".
    """
    ref = emb_referencia.unsqueeze(0)  # (1,128)

    if metrica == "produto_escalar":
        scores = emb_candidatas @ emb_referencia
    elif metrica == "cosseno":
        scores = F.normalize(emb_candidatas, dim=1) @ F.normalize(ref, dim=1).squeeze(0)
    elif metrica == "euclidiana":
        scores = -torch.cdist(emb_candidatas, ref).squeeze(1)
    else:
        raise ValueError(f"Métrica desconhecida: {metrica}")

    scores_ordenados, indices = torch.sort(scores, descending=True)
    return indices.tolist(), scores_ordenados.tolist()

# ============================
# 6. Main (exemplo com o conjunto de teste oficial do MNIST)
# ============================
def main():
    model = carregar_modelo()

    # Usa o split de teste oficial (train=False), que a rede nunca viu.
    # Para usar arquivos próprios, troque por:
    #   referencia = carregar_imagem("ref.png")
    #   candidatas = [carregar_imagem(p) for p in lista_de_caminhos]
    teste = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(teste), generator=g)[:11].tolist()

    referencia, rotulo_ref = teste[idx[0]]
    candidatas = [teste[i][0] for i in idx[1:]]
    rotulos = [teste[i][1] for i in idx[1:]]

    emb_ref = calcular_embeddings(model, [referencia])[0]  # (128,)
    emb_cand = calcular_embeddings(model, candidatas)      # (10,128)
    print(f"Embedding referência: {tuple(emb_ref.shape)} | candidatas: {tuple(emb_cand.shape)}")

    for metrica in ["cosseno", "produto_escalar", "euclidiana"]:
        indices, scores = ordenar_por_similaridade(emb_ref, emb_cand, metrica)
        print(f"\n=== Ranking ({metrica}) — referência é o dígito {rotulo_ref} ===")
        for pos, (i, s) in enumerate(zip(indices, scores), start=1):
            print(f"{pos:2d}. imagem {i} (dígito {rotulos[i]})  score = {s:.4f}")

if __name__ == "__main__":
    main()
