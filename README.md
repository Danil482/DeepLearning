# DeepLearning 
**PyTorch + MLflow • Autoencoders, CNNs and ViT**

Пет-проект по бинарной классификации рентгеновских медицинских изображений (два класса).

Цель — сравнить 4 принципиально разных подхода на одном небольшом датасете (~140 изображений):

- Stacked Autoencoder (Greedy Layer-wise pretraining)
- Stacked Autoencoder (End-to-End training)
- Компактные сверточные нейронные сети (SequentialCNN & ParallelCNN)
- Visual Transformer (пока разрабатывается)

Особенности реализации:
- Полная воспроизводимость экспериментов через **MLflow**
- Современные приёмы в CNN: Global Average Pooling, BatchNorm, Adaptive Pooling → модели не зависят от входного разрешения
- Минимальное количество параметров при максимальной точности
- Чистый, документированный код, готовый к расширению и бенчмаркам

## Как запустить

```bash
# 1. Клонируем репозиторий
git clone https://github.com/Danil482/DeepLearning.git
cd DeepLearning

# 2. Создаём окружение и ставим зависимости
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Кладём данные в папку DATA/
#    ├── YES/     ← изображения положительного класса (.bmp)
#    └── NO/      ← изображения отрицательного класса (.bmp)
