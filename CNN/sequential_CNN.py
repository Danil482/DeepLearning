import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path
from AutoEncoder.greedy_layer_wise_classifier import load_and_preprocess_data as old_load
from AutoEncoder.greedy_layer_wise_classifier import create_data_loaders


def load_and_preprocess_data_cnn(target_size=(64, 64)):
    """
    Обёртка над старой функцией, но возвращает данные в формате [N, 1, H, W] для CNN
    """
    print("Загрузка данных для CNN (с правильной формой тензора)...")
    X_flat, y = old_load(target_size=target_size)  # → [N, 4096], [N]

    # Один раз превращаем в изображение
    N = X_flat.shape[0]
    X_img = X_flat.view(N, 1, target_size[0], target_size[1])  # → [N, 1, 64, 64]

    print(f"Данные преобразованы: {X_flat.shape} → {X_img.shape}")
    return X_img, y


# --------------------- МОДЕЛЬ ---------------------
class SequentialCNN(nn.Module):
    def __init__(self, num_channels=32, dropout2d_p=0.1, dropout_p=0.1):
        super(SequentialCNN, self).__init__()

        self.features = nn.Sequential(
            # Один сверточный блок
            nn.Conv2d(in_channels=1, out_channels=num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),  # 64→32
            nn.Dropout2d(p=dropout2d_p),

            # Глобальное усреднение — делает модель нечувствительной к точному размеру входа
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(64, 2)  # logits для классов YES/NO
        )

    def forward(self, x):
        # x: (batch, 1, H, W)
        x = self.features(x)
        x = self.classifier(x)
        return x


# --------------------- ОБУЧЕНИЕ ---------------------
def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.01, device='cpu'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # Валидация
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = correct / total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

    return train_losses, val_losses, train_accs, val_accs


# --------------------- ТЕСТ ---------------------
def test_model(model, test_loader, device='cpu'):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\nФИНАЛЬНАЯ ТОЧНОСТЬ НА ТЕСТЕ: {acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=['NO', 'YES'], digits=4))
    return acc, all_preds, all_labels


# --------------------- MAIN ---------------------
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    TARGET_SIZE = (227, 227)  # Можно менять на 32, 96, 128 и т.д.

    # Загружаем и предобрабатываем данные (функции из AutoEncoder)
    X, y = load_and_preprocess_data_cnn(target_size=TARGET_SIZE)
    train_loader, val_loader, test_loader, _, _, _ = create_data_loaders(
        X, y, batch_size=16
    )

    print(f"Загружено данных: {len(X)} изображений размером {TARGET_SIZE}")

    # Создаём модель
    model = SequentialCNN(num_channels=32)
    print(model)
    print(f"Общее количество параметров: {sum(p.numel() for p in model.parameters()):,}")

    # Обучаем
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, num_epochs=50, device=str(DEVICE)
    )

    # Тестируем
    test_acc, preds, labels = test_model(model, test_loader, device=str(DEVICE))

    # Сохраняем модель
    save_path = Path("models")
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path / f"SequentialCNN_{TARGET_SIZE[0]}x{TARGET_SIZE[1]}.pth")
    print(f"Модель сохранена: {save_path}/SequentialCNN_{TARGET_SIZE[0]}x{TARGET_SIZE[1]}.pth")

    # График обучения
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.grid()
    plt.title('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.legend()
    plt.title('Accuracy')
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"images/learning_curves_{TARGET_SIZE[0]}x{TARGET_SIZE[1]}.png")
    plt.show()
