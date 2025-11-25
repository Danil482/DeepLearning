import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from datetime import datetime
from end_to_end_classifier import EndToEndClassifier, train_model
from greedy_layer_wise_classifier import load_and_preprocess_data, create_data_loaders

# --------------------- КОНФИГУРАЦИЯ ---------------------
IMAGE_SIZES = [16, 32, 48, 64, 96, 128]
LATENT_DIMS = [50, 100, 150, 200, 250]   # это latent2 (bottleneck), latent1 будет в 2 раза больше
EPOCHS = 100
N_RUNS = 5
BATCH_SIZE = 16

# Папка для результатов
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
EXP_DIR = f"experiments/end_to_end_{TIMESTAMP}"
os.makedirs(EXP_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Запуск бенчмарка на {device}")
print(f"Результаты будут в папке: {EXP_DIR}\n")

# --------------------- СПИСОК ДЛЯ РЕЗУЛЬТАТОВ ---------------------
results = []

best_acc = 0.0
best_config = None
best_model_state = None

# --------------------- ОСНОВНОЙ ЦИКЛ ---------------------
total_configs = len(IMAGE_SIZES) * len(LATENT_DIMS) * N_RUNS
current = 0

for img_size in IMAGE_SIZES:
    print(f"\n{'='*60}")
    print(f"ИЗОБРАЖЕНИЕ: {img_size}x{img_size}")
    print(f"{'='*60}")

    # Загружаем данные под текущий размер
    X, y = load_and_preprocess_data(target_size=(img_size, img_size))
    train_loader, val_loader, test_loader, _, _, _ = create_data_loaders(
        X, y, batch_size=BATCH_SIZE, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    )

    input_size = img_size * img_size

    for latent2 in LATENT_DIMS:
        latent1 = latent2 * 2  # соотношение 2:1 — классика для автоэнкодеров
        print(f"\n→ Тестируем latent1={latent1} → latent2={latent2}")

        run_accs = []

        for run in range(1, N_RUNS + 1):
            current += 1
            print(f"   Run {run}/{N_RUNS} (всего {current}/{total_configs}) ... ", end="")

            model = EndToEndClassifier(
                input_size=input_size,
                latent1=latent1,
                latent2=latent2,
                clf_hidden=32
            ).to(device)

            start_time = time.time()
            train_losses, val_losses, val_accuracies, train_accuracies = train_model(
                model, train_loader, val_loader,
                num_epochs=EPOCHS, lr=0.001
            )
            train_time = time.time() - start_time

            # Тест
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for data, labels in test_loader:
                    data = data.to(device)
                    outputs = model(data)
                    _, pred = torch.max(outputs, 1)
                    all_preds.extend(pred.cpu().numpy())
                    all_labels.extend(labels.numpy())

            test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
            run_accs.append(test_acc)

            # Сохраняем в общий список
            results.append({
                "image_size": img_size,
                "latent1": latent1,
                "latent2": latent2,
                "run": run,
                "test_accuracy": test_acc,
                "val_peak_accuracy": max(val_accuracies),
                "train_time_sec": train_time
            })

            # Обновляем лучшую модель
            if test_acc > best_acc:
                best_acc = test_acc
                best_config = (img_size, latent1, latent2, run)
                best_model_state = model.state_dict()
                print(f"НОВЫЙ РЕКОРД: {test_acc:.4f}")
            else:
                print(f"{test_acc:.4f}")

        # Среднее по запускам
        mean_acc = np.mean(run_accs)
        std_acc = np.std(run_accs)
        print(f"   Среднее по {N_RUNS} запускам: {mean_acc:.4f} ± {std_acc:.4f}")

# --------------------- СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ---------------------
df = pd.DataFrame(results)
csv_path = os.path.join(EXP_DIR, "results.csv")
df.to_csv(csv_path, index=False)
print(f"\nРезультаты сохранены в {csv_path}")

# Сохраняем лучшую модель
if best_model_state is not None:
    best_path = os.path.join(EXP_DIR, "best_model.pth")
    torch.save({
        'model_state_dict': best_model_state,
        'config': best_config,
        'test_accuracy': best_acc
    }, best_path)
    print(f"Лучшая модель ({best_acc:.4f}) сохранена: {best_path}")
    print(f"Конфигурация: image_size={best_config[0]}, latent1={best_config[1]}, latent2={best_config[2]}")

# --------------------- ВИЗУАЛИЗАЦИЯ — ТЕПЛОВАЯ КАРТА ---------------------
pivot = df.groupby(['image_size', 'latent2'])['test_accuracy'].mean().unstack()
plt.figure(figsize=(10, 7))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'Test Accuracy'})
plt.title("End-to-End: Test Accuracy (mean over 5 runs)\nimage_size × bottleneck (latent2)")
plt.xlabel("Bottleneck size (latent2)")
plt.ylabel("Image size")
plt.tight_layout()
heatmap_path = os.path.join(EXP_DIR, "accuracy_heatmap.png")
plt.savefig(heatmap_path, dpi=200)
plt.show()

print(f"\nТепловая карта сохранена: {heatmap_path}")
print(f"Всё завершено! Результаты в папке: {EXP_DIR}")