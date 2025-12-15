import time
import numpy as np
import torch
import mlflow.pytorch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Импортируем модель и данные
from CNN.parallel_CNN import ParallelCNN, load_and_preprocess_data_cnn, train_model
from AutoEncoder.greedy_layer_wise_classifier import create_data_loaders

# --------------------- КОНФИГУРАЦИЯ БЕНЧМАРКА ---------------------
IMAGE_SIZES = [16, 32, 48, 64, 96, 128]
KERNEL_SIZES = [(3, 5), (3, 7), (5, 5), (5, 7)]  # (branch1, branch2)
CHANNELS1 = [16, 32]  # для branch1
CHANNELS2 = [32, 64, 96]  # для branch2
STRIDES = [1, 2]  # stride в MaxPool2d
EPOCHS = 30
N_RUNS = 3
BATCH_SIZE = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("parallel_cnn_benchmark")

results = []
best_acc = 0.0
best_run_id = None

total_configs = len(IMAGE_SIZES) * len(KERNEL_SIZES) * len(CHANNELS1) * len(CHANNELS2) * len(STRIDES) * N_RUNS
current = 0

print(f"Запуск бенчмарка ParallelCNN на {DEVICE}")
print(f"Всего конфигураций: {total_configs}\n")

with mlflow.start_run(run_name="ParallelCNN_Benchmark"):
    for img_size in IMAGE_SIZES:
        print(f"\n{'=' * 70}")
        print(f"IMAGE SIZE: {img_size}×{img_size}")
        print(f"{'=' * 70}")

        X, y = load_and_preprocess_data_cnn(target_size=(img_size, img_size))
        train_loader, val_loader, test_loader, _, _, _ = create_data_loaders(
            X, y, batch_size=BATCH_SIZE
        )

        for kernel1, kernel2 in KERNEL_SIZES:
            for ch1 in CHANNELS1:
                for ch2 in CHANNELS2:
                    for stride in STRIDES:
                        total_channels = ch1 + ch2

                        for run in range(1, N_RUNS + 1):
                            current += 1
                            run_name = f"size_{img_size}_k_{kernel1}_{kernel2}_c{ch1}_{ch2}_s_{stride}_run_{run}"

                            with mlflow.start_run(run_name=run_name, nested=True):
                                print(f"{current:3d}/{total_configs} | {run_name} | ", end="")

                                mlflow.log_params({
                                    "image_size": img_size,
                                    "kernel_branch1": kernel1,
                                    "kernel_branch2": kernel2,
                                    "channels_branch1": ch1,
                                    "channels_branch2": ch2,
                                    "maxpool_stride": stride,
                                    "total_channels": total_channels,
                                    "epochs": EPOCHS,
                                    "run": run
                                })

                                # Модель с кастомными параметрами
                                model = ParallelCNN(
                                    channels1=ch1,
                                    channels2=ch2,
                                    kernel1=kernel1,
                                    kernel2=kernel2,
                                    stride=stride
                                ).to(DEVICE)

                                # Обучаем (переиспользуем функцию из Sequential_CNN или пишем свою)
                                start_time = time.time()
                                history = train_model(model, train_loader, val_loader,
                                                      num_epochs=EPOCHS, device=DEVICE)
                                train_time = time.time() - start_time

                                # Тест
                                model.eval()
                                preds, labels = [], []
                                with torch.no_grad():
                                    for data, label in test_loader:
                                        out = model(data.to(DEVICE))
                                        pred = torch.argmax(out, dim=1)
                                        preds.extend(pred.cpu().numpy())
                                        labels.extend(label.numpy())

                                test_acc = np.mean(np.array(preds) == np.array(labels))

                                # Логируем
                                mlflow.log_metric("test_accuracy", test_acc)
                                mlflow.log_metric("val_peak_accuracy", max(history['val_acc']))
                                mlflow.log_metric("train_time_sec", train_time)

                                # Сохраняем результат для тепловой карты
                                results.append({
                                    "image_size": img_size,
                                    "kernel1": kernel1,
                                    "kernel2": kernel2,
                                    "channels1": ch1,
                                    "channels2": ch2,
                                    "stride": stride,
                                    "total_channels": total_channels,
                                    "test_accuracy": test_acc,
                                    "run": run
                                })

                                if test_acc > best_acc:
                                    best_acc = test_acc
                                    best_run_id = mlflow.active_run().info.run_id
                                    mlflow.set_tag("best_model", "true")
                                    print(f"NEW BEST → {test_acc:.4f}")
                                else:
                                    print(f"{test_acc:.4f}")

    # Финальные метрики
    mlflow.log_metric("best_test_accuracy", best_acc)
    print(f"\nБенчмарк завершён! Лучшая точность: {best_acc:.4f}")

# --------------------- ТЕПЛОВАЯ КАРТА ---------------------
df = pd.DataFrame(results)
mean_df = df.groupby(["image_size", "total_channels"])["test_accuracy"].mean().reset_index()

pivot = mean_df.pivot(index="image_size", columns="total_channels", values="test_accuracy")

plt.figure(figsize=(10, 7))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", cbar_kws={'label': 'Test Accuracy'})
plt.title("ParallelCNN: Test Accuracy (mean over 5 runs)\nimage_size × total_channels")
plt.xlabel("Total channels after concat (branch1 + branch2)")
plt.ylabel("Image size")
plt.tight_layout()

save_path = "images/parallel_cnn_heatmap.png"
plt.savefig(save_path, dpi=200)
plt.show()

print(f"Тепловая карта сохранена: {save_path}")
# из папки с бд
# mlflow ui --backend-store-uri sqlite:///mlflow.db
# http://127.0.0.1:5000