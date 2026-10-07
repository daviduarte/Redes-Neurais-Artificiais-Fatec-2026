import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

# =====================
# 1. Dataset mínimo
# =====================


class MiniNewsDataset(Dataset):
    def __init__(self):
        # Textos e rótulos (0=Esporte, 1=Política, 2=Tecnologia)
        self.samples = [
            ("time venceu o jogo", 0),
            ("gol no ultimo minuto", 0),
            ("novo smartphone lançado", 2),
            ("lançamento de app", 2),
            ("presidente assinou a lei", 1),
            ("debate político acirrado", 1),
            ("senador perdeu o embate por ser muito fraco", 1)
        ]

        # Construir vocabulário
        tokens = []
        for text, _ in self.samples:
            words = text.split()
            for word in words:
                tokens.append(word)

        tokens = set(tokens)    #remove duplicatas
        self.vocab = {word: i+2 for i, word in enumerate(tokens)}
        self.vocab["<PAD>"] = 0
        self.vocab["<UNK>"] = 1
        self.itos = {i: w for w, i in self.vocab.items()}
        print("Vocabulário: ")
        print(self.vocab)
        print(self.itos)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        # Converter palavras em IDs
        ids = [self.vocab.get(w, self.vocab["<UNK>"]) for w in text.split()]
        #print("Texto:", text, "-> IDs:", ids, "Rótulo:", label)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# =====================
# 2. Collate para padding
# =====================
def collate_batch(batch):
    texts, labels = zip(*batch)

    # pad_sequence ajusta todas as sequências para o mesmo comprimento, 
    # preenchendo com zeros (padding_value=0) onde a sequência é menor 
    # que a maior do batch.    
    texts = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return texts, labels

dataset = MiniNewsDataset()
loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_batch)

# =====================
# 3. Modelo RNN simples
# =====================
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        # Embedding estabelece um vetor (representação, embedding) para cada palavra. 
        # Cada linha da matriz corresponde a um índice do vocabulário.
        # Cada coluna é uma dimensão do embedding (embed_dim).
        # Essa representação vetorial é aprendida durante o treinamento.
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        #print("Antes do embedding:", x)
        x = self.embedding(x)
        #print("Depois do embedding:", x)
        #exit()
        _, hidden = self.rnn(x)
        out = self.fc(hidden.squeeze(0))
        return out

vocab_size = len(dataset.vocab)
print("Vocab size:", vocab_size)
num_classes = 3
model = RNNClassifier(vocab_size, embed_dim=10, hidden_dim=16, num_classes=num_classes)

# =====================
# 4. Treino rápido
# =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(20):
    model.train()
    total_loss = 0
    for texts, labels in loader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

# =====================
# 5. Teste rápido
# =====================
model.eval()
with torch.no_grad():
    for texts, labels in loader:
        texts, labels = texts.to(device), labels.to(device)
        preds = model(texts).argmax(dim=1)
        print("Preds:", preds.cpu().numpy(), "Labels:", labels.cpu().numpy())
