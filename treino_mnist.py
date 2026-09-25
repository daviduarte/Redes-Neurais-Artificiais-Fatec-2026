import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_MELHOR = os.path.join(CHECKPOINT_DIR, "cnn_mnist_melhor.pt")
CHECKPOINT_ULTIMO = os.path.join(CHECKPOINT_DIR, "cnn_mnist_ultimo.pt")

# ============================
# 1. Transformações e dataset MNIST
# ============================
transform = transforms.Compose([
    transforms.ToTensor()  # mantém formato (1,28,28) para CNN
])

# Baixa o MNIST
full_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

# ----------------------------
# Divisão 80% treino, 10% validação, 10% teste
# ----------------------------
total_len = len(full_dataset)  # 60.000
train_len = int(0.8 * total_len)
val_len = int(0.1 * total_len)
test_len = total_len - train_len - val_len

# Semente fixa para que a divisão seja reprodutível
gerador = torch.Generator().manual_seed(42)
train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_len, val_len, test_len], generator=gerador
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ============================
# 2. Modelo CNN
# ============================
class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        # Conv + Pooling
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # (1,28,28) -> (32,28,28)
        self.pool1 = nn.MaxPool2d(2, 2)                         # -> (32,14,14)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # -> (64,14,14)
        self.pool2 = nn.MaxPool2d(2, 2)                         # -> (64,7,7)

        # Fully Connected
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes do MNIST

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)  # achata
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ============================
# 3. Função para calcular acurácia e loss
# ============================
def avaliar_modelo(model, dataloader, criterion):
    total_loss = 0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    model.train()
    avg_loss = total_loss / len(dataloader)
    acc = correct / total
    return avg_loss, acc

# ============================
# 4. Checkpoint
# ============================
def salvar_checkpoint(caminho, model, optimizer, epoch, val_loss, val_acc):
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_acc": val_acc,
    }, caminho)

# ============================
# 5. Loop de treino
# ============================
def train_looping(model, train_loader, val_loader, criterion, writer):
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    epochs = 10
    melhor_val_loss = float("inf")

    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss, val_acc = avaliar_modelo(model, val_loader, criterion)
        print(f"Época {epoch+1}, Loss treino: {total_loss/len(train_loader):.4f}, "
              f"Loss val: {val_loss:.4f}, Acurácia val: {val_acc*100:.2f}%")

        writer.add_scalars("Train/", {"Loss Treino": total_loss/len(train_loader), "Loss Validação": val_loss}, epoch)
        writer.add_scalar("Train/Acuracia Validação", val_acc, epoch)

        # Checkpoint da última época (permite retomar o treino)
        salvar_checkpoint(CHECKPOINT_ULTIMO, model, optimizer, epoch, val_loss, val_acc)

        # Checkpoint do melhor modelo segundo a loss de validação
        if val_loss < melhor_val_loss:
            melhor_val_loss = val_loss
            salvar_checkpoint(CHECKPOINT_MELHOR, model, optimizer, epoch, val_loss, val_acc)
            print(f"  -> Novo melhor modelo salvo em {CHECKPOINT_MELHOR}")

    return val_loss, val_acc

# ============================
# 6. Main
# ============================
def main():
    writer = SummaryWriter(log_dir="runs/mnist_cnn")
    model = CNNMnist()
    criterion = nn.CrossEntropyLoss()

    val_loss, val_acc = train_looping(model, train_loader, val_loader, criterion, writer)

    # Avaliação final no teste usando o melhor checkpoint
    ckpt = torch.load(CHECKPOINT_MELHOR, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"\nMelhor checkpoint: época {ckpt['epoch']+1} (loss val {ckpt['val_loss']:.4f})")

    test_loss, test_acc = avaliar_modelo(model, test_loader, criterion)
    print(f"\n=== Avaliação final no conjunto de teste ===")
    print(f"Loss teste: {test_loss:.4f}")
    print(f"Acurácia teste: {test_acc*100:.2f}%")

    writer.close()

if __name__ == "__main__":
    main()
