import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path
from CNN.sequential_CNN import load_and_preprocess_data_cnn
from AutoEncoder.greedy_layer_wise_classifier import create_data_loaders


class ParallelCNN(nn.Module):
    def __init__(self, channels1=16, channels2=48, dropout_p=0.3):
        super().__init__()

        # === ПЕРВЫЙ ПАРАЛЛЕЛЬНЫЙ БЛОК (маленькое ядро) ===
        self.branch1 = nn.Sequential(
            nn.Conv2d(1, channels1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.1)
        )

        # === ВТОРОЙ ПАРАЛЛЕЛЬНЫЙ БЛОК (большое ядро) ===
        self.branch2 = nn.Sequential(
            nn.Conv2d(1, channels2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(channels2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.1)
        )

        # Общий классификатор после конкатенации
        total_channels = channels1 + channels2  # 16 + 48 = 64

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # ← магия: любой размер → 1×1
            nn.Flatten(),
            nn.Linear(total_channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        # x: [B, 1, H, W]
        out1 = self.branch1(x)  # → [B, 16, H//2, W//2]
        out2 = self.branch2(x)  # → [B, 48, H//2, W//2]

        # Конкатенация по канальному измерению
        out = torch.cat([out1, out2], dim=1)  # → [B, 64, H//2, W//2]

        out = self.classifier(out)
        return out


# Обучение и тест — полностью как в Sequential_CNN (оставляем без изменений)
def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001, device='cpu'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = correct = total = 0

        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

        # Validation
        model.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, pred = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (pred == labels).sum().item()

        history['train_loss'].append(running_loss / len(train_loader))
        history['train_acc'].append(correct / total)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_correct / val_total)

        print(f"Epoch [{epoch + 1:2d}/{num_epochs}] "
              f"Train Acc: {correct / total:.4f} | Val Acc: {val_correct / val_total:.4f}")

    return history


def test_model(model, test_loader, device='cpu'):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            out = model(data)
            pred = torch.argmax(out, dim=1)
            preds.extend(pred.cpu().numpy())
            labels.extend(label.numpy())

    acc = accuracy_score(labels, preds)
    print(f"\nТЕСТ: Accuracy = {acc:.4f}")
    print(classification_report(labels, preds, target_names=['NO', 'YES'], digits=4))
    return acc


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {DEVICE}")

    target_size = (64, 64)

    X, y = load_and_preprocess_data_cnn(target_size=target_size)
    train_loader, val_loader, test_loader, _, _, _ = create_data_loaders(X, y, batch_size=16)

    model = ParallelCNN(channels1=16, channels2=48, dropout_p=0.3)
    print(model)
    print(f"Параметров: {sum(p.numel() for p in model.parameters()):,}")

    history = train_model(model, train_loader, val_loader, num_epochs=50, device=DEVICE)
    test_model(model, test_loader, device=DEVICE)

    # Сохранение
    Path("CNN/models").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"models/ParallelCNN_{target_size[0]}x{target_size[1]}.pth")

    # График
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.legend()
    plt.grid()
    plt.title('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Val')
    plt.legend()
    plt.grid()
    plt.title('Accuracy')
    plt.tight_layout()
    plt.savefig(f"images/parallel_learning_curves_{target_size[0]}x{target_size[1]}.png")
    plt.show()
