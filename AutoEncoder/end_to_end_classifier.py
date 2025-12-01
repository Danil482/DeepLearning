import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from greedy_layer_wise_classifier import load_and_preprocess_data, create_data_loaders
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ==================== ЕДИНАЯ СЕТЬ: Encoder1 → Encoder2 → Classifier ====================

class EndToEndClassifier(nn.Module):
    def __init__(self, input_size=64 * 64, latent1=100, latent2=50, clf_hidden=32):
        super(EndToEndClassifier, self).__init__()

        # Encoder 1: 4096 → 100
        self.encoder1 = nn.Sequential(
            nn.Linear(input_size, latent1),
            nn.ReLU(),  # ReLU лучше чем Sigmoid для глубоких сетей
            # nn.BatchNorm1d(latent1),  # можно раскомментить при batch_size > 1
        )

        # Encoder 2: 100 → 50
        self.encoder2 = nn.Sequential(
            nn.Linear(latent1, latent2),
            nn.ReLU(),
        )

        # Classifier: 50 → 2
        self.classifier = nn.Sequential(
            nn.Linear(latent2, clf_hidden),
            nn.ReLU(),
            nn.Linear(clf_hidden, 2)  # CrossEntropyLoss сам применит softmax
        )

    def forward(self, x):
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.classifier(x)
        return x

    # Удобные методы для получения промежуточных представлений (если потом захочешь)
    def encode_to_100d(self, x):
        return self.encoder1(x)

    def encode_to_50d(self, x):
        return self.encoder2(self.encoder1(x))


# ==================== ОБУЧЕНИЕ ====================

def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    print("\nЗапуск end-to-end обучения...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Валидация
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_train_loss = running_loss / len(train_loader)
        train_acc = train_correct / train_total
        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)

        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1:3d}/{num_epochs}]  "
                  f"Train Loss: {avg_train_loss:.4f}  "
                  f"Train Acc: {train_acc: .4f}"
                  f"Val Loss: {avg_val_loss:.4f}  "
                  f"Val Acc: {val_acc:.4f}")

    return train_losses, train_accuracies, val_losses, val_accuracies


# ==================== ОСНОВНОЙ БЛОК ====================

def main():
    # Загружаем данные (те же функции, что и раньше)
    X, y = load_and_preprocess_data()

    train_loader, val_loader, test_loader, _, _, _ = create_data_loaders(
        X, y, batch_size=16, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    )

    input_size = 64 * 64
    model = EndToEndClassifier(input_size=input_size, latent1=100, latent2=50, clf_hidden=32)

    print(model)
    print(f"Общее количество параметров: {sum(p.numel() for p in model.parameters()):,}")

    # Обучение
    train_losses, train_accuracies,val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, num_epochs=50, lr=0.001
    )

    # Тестирование
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    print("\n" + "=" * 50)
    print(f"ФИНАЛЬНАЯ ТОЧНОСТЬ НА ТЕСТЕ: {test_acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=['YES', 'NO']))

    # График обучения
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('End-to-End: Loss')
    plt.xlabel('Epoch')
    plt.ylabel('CrossEntropy Loss')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title('End-to-End: Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig('images/end_to_end_learning_curve.png')
    plt.show()

    # Сохранение модели
    torch.save(model.state_dict(), 'models/end_to_end_classifier.pth')
    print("\nМодель сохранена как 'end_to_end_classifier.pth'")


if __name__ == "__main__":
    main()
