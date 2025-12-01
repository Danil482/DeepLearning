import time
import numpy as np
import torch
import mlflow.pytorch
from end_to_end_classifier import EndToEndClassifier, train_model
from greedy_layer_wise_classifier import load_and_preprocess_data, create_data_loaders

# --------------------- КОНФИГУРАЦИЯ ---------------------
IMAGE_SIZES = [16, 32, 48, 64, 96, 128]
LATENT_DIMS = [50, 100, 150, 200, 250]   # latent2
EPOCHS = 50
N_RUNS = 5
BATCH_SIZE = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------- MLFLOW SETUP ---------------------
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("end_to_end_benchmark")

print(f"Запуск бенчмарка с MLflow на {device}")
print("После завершения: запусти → mlflow ui ← и открой http://127.0.0.1:5000\n")

best_acc = 0.0
best_run_id = None

# --------------------- ОСНОВНОЙ ЦИКЛ ---------------------
total_configs = len(IMAGE_SIZES) * len(LATENT_DIMS) * N_RUNS
current = 0

for img_size in IMAGE_SIZES:
    print(f"\n{'='*60}")
    print(f"IMAGE SIZE: {img_size}x{img_size}")
    print(f"{'='*60}")

    # Загружаем данные под текущий размер
    X, y = load_and_preprocess_data(target_size=(img_size, img_size))
    train_loader, val_loader, test_loader, _, _, _ = create_data_loaders(
        X, y, batch_size=BATCH_SIZE
    )
    input_size = img_size * img_size

    for latent2 in LATENT_DIMS:
        latent1 = latent2 * 2  # 2:1 соотношение

        for run in range(1, N_RUNS + 1):
            current += 1
            run_name = f"img{img_size}_l1_{latent1}_l2_{latent2}_run{run}"

            with mlflow.start_run(run_name=run_name):
                print(f"{current}/{total_configs} | {run_name} | ", end="")

                # Логируем параметры
                mlflow.log_params({
                    "image_size": img_size,
                    "input_pixels": input_size,
                    "latent1": latent1,
                    "latent2": latent2,
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "run_id": run
                })

                # Создаём и обучаем модель
                model = EndToEndClassifier(
                    input_size=input_size,
                    latent1=latent1,
                    latent2=latent2,
                    clf_hidden=32
                ).to(device)

                start_time = time.time()
                train_losses, val_losses, val_accuracies, train_accuracies = train_model(
                    model, train_loader, val_loader, num_epochs=EPOCHS, lr=0.001
                )
                train_time = time.time() - start_time

                # Тест
                model.eval()
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for data, labels in test_loader:
                        outputs = model(data.to(device))
                        _, pred = torch.max(outputs, 1)
                        all_preds.extend(pred.cpu().numpy())
                        all_labels.extend(labels.numpy())

                test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
                val_peak = max(val_accuracies)

                # Логируем метрики
                mlflow.log_metric("test_accuracy", test_acc)
                mlflow.log_metric("val_peak_accuracy", val_peak)
                mlflow.log_metric("train_time_sec", train_time)

                # Сохраняем модель как артефакт
                model_path = f"models/model_img{img_size}_l2{latent2}_run{run}.pth"
                example_batch, _ = next(iter(train_loader))
                example_input = example_batch[:2].numpy()
                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    name=f"img{img_size}_l1{latent1}_l2{latent2}_run{run}",
                    input_example=example_input
                )

                # Если это лучший результат — запоминаем
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_run_id = mlflow.active_run().info.run_id
                    mlflow.set_tag("best_model", "true")
                    print(f"NEW BEST: {test_acc:.4f}")
                else:
                    print(f"{test_acc:.4f}")

# После всех запусков — логируем общее окончание
mlflow.log_metric("best_test_accuracy_overall", best_acc)
mlflow.set_tag("status", "completed")
print(f"\nБенчмарк завершён! Лучшая точность: {best_acc:.4f} (run_id: {best_run_id})")

print(f"\nГотово! Запусти сейчас:")
print(f"mlflow ui --backend-store-uri sqlite:///mlflow.db")
print(f"И открой в браузере: http://127.0.0.1:5000")

# из папки с бд
# mlflow ui --backend-store-uri sqlite:///mlflow.db