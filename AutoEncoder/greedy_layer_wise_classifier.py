import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from PIL import Image
import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


# ==================== 1. ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ ====================

def load_images_from_folder(folder_path, label, target_size=(64, 64)):
    """Загрузка изображений из папки и их нормализация"""
    images = []
    labels = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.bmp'):
            img_path = os.path.join(folder_path, filename)

            # Загрузка и преобразование изображения
            img = Image.open(img_path).convert('L')  # Конвертируем в grayscale
            img = img.resize(target_size)
            img_array = np.array(img, dtype=np.float32)

            # Нормализация делением на 255
            img_array = img_array / 255.0

            images.append(img_array)
            labels.append(label)

    return images, labels


def load_and_preprocess_data(target_size=(64, 64)):
    """Загрузка и предобработка данных"""
    print("Загрузка данных...")
    yes_images, yes_labels = load_images_from_folder('../DATA/YES/', label=0, target_size=target_size)
    no_images, no_labels = load_images_from_folder('../DATA/NO/', label=1, target_size=target_size)

    all_images = yes_images + no_images
    all_labels = yes_labels + no_labels

    print(f"Загружено {len(yes_images)} YES, {len(no_images)} NO, всего: {len(all_images)}")

    X = torch.tensor(np.array(all_images)).float().view(len(all_images), -1)
    y = torch.tensor(np.array(all_labels)).long()

    return X, y


# ==================== 2. РАЗДЕЛЕНИЕ НА ТРЕНИРОВОЧНУЮ, ВАЛИДАЦИОННУЮ И ТЕСТОВУЮ ВЫБОРКИ ====================
def create_data_loaders(X, y, batch_size=16, train_ratio=0.8, val_ratio=0.1):
    """
    Создает DataLoader'ы для тренировочной, валидационной и тестовой выборок

    Args:
        X: тензор признаков
        y: тензор меток
        batch_size: размер батча
        train_ratio: доля тренировочных данных
        val_ratio: доля валидационных данных

    Returns:
        train_loader, val_loader, test_loader: DataLoader'ы для каждой выборки
        train_dataset, val_dataset, test_dataset: соответствующие Dataset'ы
    """

    # Создаем dataset
    dataset = TensorDataset(X, y)
    total_size = len(dataset)

    # Вычисляем размеры выборок
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # Разделяем данные
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Выводим информацию о разделении
    print(f"\nРазделение данных:")
    print(f"Тренировочная выборка: {len(train_dataset)} образцов")
    print(f"Валидационная выборка: {len(val_dataset)} образцов")
    print(f"Тестовая выборка: {len(test_dataset)} образцов")
    print(f"Всего: {total_size} образцов")

    # Создаем DataLoader'ы
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # перемешиваем только тренировочные данные
        num_workers=0  # можно увеличить для параллельной загрузки
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # не перемешиваем валидацию и тест
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


# ==================== 3. ОПРЕДЕЛЕНИЕ АРХИТЕКТУР АВТОЭНКОДЕРОВ ====================

class Autoencoder1(nn.Module):
    def __init__(self, target_size):
        super(Autoencoder1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(target_size, 100),  # ПРЯМОЙ ПЕРЕХОД 784 -> 100
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(100, target_size),  # ПРЯМОЙ ПЕРЕХОД 100 -> 784
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class Autoencoder2(nn.Module):
    def __init__(self, target_size):
        super(Autoencoder2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(target_size, 50),  # ПРЯМОЙ ПЕРЕХОД 100 -> 50
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(50, target_size),  # ПРЯМОЙ ПЕРЕХОД 50 -> 100
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class Classifier(nn.Module):
    """Классификатор: 50 -> hidden -> 2 (YES/NO)"""

    def __init__(self, hidden_size=32):
        super(Classifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(50, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, 2),  # 2 класса: YES и NO
            # Softmax будет применен в функции потерь CrossEntropyLoss
        )

    def forward(self, x):
        return self.network(x)


# ==================== 4. ФУНКЦИИ ДЛЯ ОБУЧЕНИЯ ====================

def train_autoencoder(model, train_loader, val_loader, num_epochs=100, model_name="Autoencoder"):
    """Обучение автоэнкодера"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []

    print(f"\nОбучение {model_name}...")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)

            optimizer.zero_grad()
            _, reconstructed = model(data)
            loss = criterion(reconstructed, data)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Валидация
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                _, reconstructed = model(data)
                loss = criterion(reconstructed, data)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    return train_losses, val_losses


def train_classifier(model, train_loader, val_loader, num_epochs=100):
    """Обучение классификатора"""
    model.to(device)
    criterion = nn.CrossEntropyLoss()  # Включает Softmax
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []
    val_accuracies = []

    print(f"\nОбучение классификатора...")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

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

                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(accuracy)

        if (epoch + 1) % 20 == 0:
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {accuracy:.4f}')

    return train_losses, val_losses, val_accuracies


# ==================== 5. ОСНОВНОЙ ПРОЦЕСС ОБУЧЕНИЯ ====================

def main():
    # Загрузка данных
    X, y = load_and_preprocess_data()

    # Создание DataLoader'ов
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = create_data_loaders(
        X, y,
        batch_size=16,
        train_ratio=0.8,
        val_ratio=0.1
    )
    target_size1 = 64 * 64
    target_size2 = 100
    # Создание моделей
    autoencoder1 = Autoencoder1(target_size1)
    autoencoder2 = Autoencoder2(target_size2)
    classifier = Classifier(hidden_size=16)  # Маленький скрытый слой из-за малого количества данных

    print("Архитектура моделей создана")

    # Обучение первого автоэнкодера
    train_losses_ae1, val_losses_ae1 = train_autoencoder(
        autoencoder1, train_loader, val_loader, num_epochs=80, model_name="Autoencoder1"
    )

    # Извлечение признаков 100D с помощью первого автоэнкодера
    print("\nИзвлечение признаков 100D...")
    autoencoder1.eval()

    def extract_features_100d(dataset):
        features = []
        labels_list = []
        with torch.no_grad():
            for data, label in dataset:  # data — [784], label — scalar tensor
                data = data.unsqueeze(0).to(device)  # делаем [1, 784]
                encoded, _ = autoencoder1(data)
                features.append(encoded)  # [1, 100]
                labels_list.append(label)

        features_tensor = torch.cat(features, dim=0).squeeze(0)  # [N, 100]
        labels_tensor = torch.stack(labels_list)  # [N]

        print(f"Features shape: {features_tensor.shape}, Labels shape: {labels_tensor.shape}")
        return features_tensor, labels_tensor

    # Создание новых датасетов с признаками 100D
    X_train_100d, y_train_100d = extract_features_100d(train_dataset)
    X_val_100d, y_val_100d = extract_features_100d(val_dataset)
    X_test_100d, y_test_100d = extract_features_100d(test_dataset)

    train_dataset_100d = TensorDataset(X_train_100d, y_train_100d)
    val_dataset_100d = TensorDataset(X_val_100d, y_val_100d)
    test_dataset_100d = TensorDataset(X_test_100d, y_test_100d)

    batch_size = 16

    train_loader_100d = DataLoader(train_dataset_100d, batch_size=batch_size, shuffle=True)
    val_loader_100d = DataLoader(val_dataset_100d, batch_size=batch_size)
    test_loader_100d = DataLoader(test_dataset_100d, batch_size=batch_size)

    # Обучение второго автоэнкодера на признаках 100D
    train_losses_ae2, val_losses_ae2 = train_autoencoder(
        autoencoder2, train_loader_100d, val_loader_100d, num_epochs=60, model_name="Autoencoder2"
    )

    # Извлечение финальных признаков 50D
    print("\nИзвлечение финальных признаков 50D...")
    autoencoder2.eval()

    def extract_features_50d(dataset):
        features = []
        labels_list = []
        with torch.no_grad():
            for data, label in dataset:
                data = data.unsqueeze(0).to(device)  # добавляем batch dimension
                encoded, _ = autoencoder2(data)
                features.append(encoded)
                labels_list.append(label)

        features_tensor = torch.cat(features, dim=0)
        labels_tensor = torch.stack(labels_list)
        return features_tensor, labels_tensor

    # Создание финальных датасетов с признаками 50D
    X_train_50d, y_train_50d = extract_features_50d(train_dataset_100d)
    X_val_50d, y_val_50d = extract_features_50d(val_dataset_100d)
    X_test_50d, y_test_50d = extract_features_50d(test_dataset_100d)

    train_dataset_50d = TensorDataset(X_train_50d, y_train_50d)
    val_dataset_50d = TensorDataset(X_val_50d, y_val_50d)
    test_dataset_50d = TensorDataset(X_test_50d, y_test_50d)

    train_loader_50d = DataLoader(train_dataset_50d, batch_size=batch_size, shuffle=True)
    val_loader_50d = DataLoader(val_dataset_50d, batch_size=batch_size)
    test_loader_50d = DataLoader(test_dataset_50d, batch_size=batch_size)

    print(f"Размерность финальных признаков: {X_train_50d.shape[1]}")

    # Обучение классификатора на признаках 50D
    train_losses_clf, val_losses_clf, val_accuracies = train_classifier(
        classifier, train_loader_50d, val_loader_50d, num_epochs=100
    )

    # ==================== 6. ТЕСТИРОВАНИЕ ====================

    print("\n" + "=" * 50)
    print("ФИНАЛЬНОЕ ТЕСТИРОВАНИЕ")
    print("=" * 50)

    classifier.eval()
    all_preds = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for data, labels in test_loader_50d:
            data, labels = data.to(device), labels.to(device)
            outputs = classifier(data)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # Метрики
    test_accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nТочность на тестовой выборке: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['YES', 'NO']))

    # ==================== 7. СОХРАНЕНИЕ МОДЕЛЕИ ====================

    torch.save(autoencoder1.state_dict(), 'models/autoencoder1.pth')
    torch.save(autoencoder2.state_dict(), 'models/autoencoder2.pth')
    torch.save(classifier.state_dict(), 'models/greedy_layer_wise_classifier.pth')
    print("\nМодели сохранены: autoencoder1.pth, autoencoder2.pth, greedy_layer_wise_classifier.pth")

    # ==================== 8. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ====================

    # Визуализация кривых обучения
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses_ae1, label='Train Loss')
    plt.plot(val_losses_ae1, label='Val Loss')
    plt.title('Autoencoder 1 Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(train_losses_ae2, label='Train Loss')
    plt.plot(val_losses_ae2, label='Val Loss')
    plt.title('Autoencoder 2 Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(train_losses_clf, label='Train Loss')
    plt.plot(val_losses_clf, label='Val Loss')
    plt.title('Classifier Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig('images/learning_curves.png')
    plt.show()

    # Визуализация нескольких примеров реконструкции
    autoencoder1.eval()
    autoencoder2.eval()

    with torch.no_grad():
        # Берем несколько тестовых примеров
        test_samples, _ = next(iter(test_loader))
        test_samples = test_samples.to(device)

        # Реконструкция через оба автоэнкодера
        encoded_100d, reconstructed_100d = autoencoder1(test_samples)
        encoded_50d, reconstructed_100d_from_50d = autoencoder2(encoded_100d)

        restored_100d = autoencoder2.decoder(encoded_50d)  # [B, 100]
        final_reconstructed = autoencoder1.decoder(restored_100d)  # [B, 784]

        image_size = (64, 64)
        # Визуализация
        fig, axes = plt.subplots(3, 5, figsize=(12, 8))
        for i in range(5):
            # Оригинал
            axes[0, i].imshow(test_samples[i].cpu().view(image_size), cmap='gray')
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')

            # Реконструкция после AE1
            axes[1, i].imshow(reconstructed_100d[i].cpu().view(image_size), cmap='gray')
            axes[1, i].set_title('After AE1')
            axes[1, i].axis('off')

            # Реконструкция после AE2
            axes[2, i].imshow(final_reconstructed[i].cpu().view(image_size), cmap='gray')
            axes[2, i].set_title('After AE1→AE2→AE1')
            axes[2, i].axis('off')

        plt.tight_layout()
        plt.savefig('images/reconstruction_examples.png')
        plt.show()


if __name__ == "__main__":
    main()
