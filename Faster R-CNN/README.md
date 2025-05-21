# 🏀 Faster R-CNN (ResNet-50 + FPN) — Basketball Foul Detector

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-red.svg)
![TorchMetrics](https://img.shields.io/badge/TorchMetrics-0.11-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-yellow.svg)
![Kaggle API](https://img.shields.io/badge/Kaggle-API-orange.svg)
![Colab](https://img.shields.io/badge/Google%20Colab-compatible-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

---

## 📖 Overview
Этот репозиторий содержит ноутбук, который обучает **однокадровый** детектор баскетбольных фолов.  
В основе — **Faster R-CNN** с **ResNet-50 + Feature Pyramid Network**, предобученный на COCO и дообученный на размеченных кадрах из Label Studio.

---

## 🎯 TL;DR – текущие результаты  
| Метрика (val) | Значение * |
|---------------|-----------|
| **mAP @ 0.50** | **0.0105** |
| AR @ 10        | 0.0749 |

\* получено после 30 эпох на GPU A100 40 GB.


---

## 📦 Dataset

| Источник                          | Размер | Ссылка                                                                                                                     |
| --------------------------------- | -----: | -------------------------------------------------------------------------------------------------------------------------- |
| Kaggle: **Foul Detection (test)** | 1.3 GB | [https://www.kaggle.com/datasets/sesmlhs/foul-detection-test](https://www.kaggle.com/datasets/sesmlhs/foul-detection-test) |

---

## ⚙️ Ноутбук по блокам

| Блок      | Ключевая логика                                                                                                                    | Why / Особенности                         |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| **BLK-1** | Установка, монтирование Drive, `kaggle datasets download`                                                                          | Чек-пойнты хранятся на Drive              |
| **BLK-2** | Чтение JSON → `{frame : bbox}`                                                                                                     | Обрабатывает пропуски и пустые bbox       |
| **BLK-3** | Stratified split 80/20, incremental-training лог                                                                                   | `seen_videos.txt` — не переобучать старое |
| **BLK-4** | `FrameDataset`: берём **центральный кадр**, resize 512, нормализация, аугментации (ColorJitter, flip, blur, rotate) + safe-wrapper | Центральный кадр ≈ вероятный момент фола  |
| **BLK-5** | Создание Faster R-CNN ResNet-50-FPN, замена head→2 класса                                                                          | «фон» / «foul»                            |
| **BLK-6** | EMA-update, метрики, LR-планировщики                                                                                               | —                                         |
| **BLK-7** | Обучение: **AdamW + OneCycle → CosineWarmRestarts → SWA** (+ EMA, AMP). Чек-пойнт при mAP↑, сохранение SWA                         | Устойчивый DataLoader (`num_workers=0`)   |
| **BLK-8** | Сквозная валидация: mAP @\[0.50‒0.95], @0.50, @0.75, AR @ 1/10/100                                                                 | Табличный вывод                           |

---
## 🧠 Почему такая модель?

| Компонент                 | Причина выбора                                         |
| ------------------------- | ------------------------------------------------------ |
| **Faster R-CNN**          | Надёжный баланс точность/скорость; легко fine-tune     |
| **ResNet-50**             | Достаточная глубина при умеренных FLOPs                |
| **FPN**                   | Держит мульти-скейл: фол может быть крупным или мелким |
| **AdamW + OneCycle**      | Быстрый прогрев и плавный спад LR                      |
| **SWA + EMA**             | Усреднение снижает шум последних шагов – + mAP         |
| **AMP**                   | \~×1.5 скорость, −40 % VRAM                            |
| **WeightedRandomSampler** | Компенсирует класс-дисбаланс («фон» >> «foul»)         |



---

## 🛠️ Hyper-Parameters

| Параметр            |        Значение | Описание                |
| ------------------- | --------------: | ----------------------- |
| `EPOCHS`            |              30 | Полное обучение         |
| `PHASE1` / `PHASE2` |         10 / 20 | OneCycle → Cosine       |
| `swa_start`         |              25 | Старт SWA               |
| `batch_size`        |               4 | Для 12 GB VRAM          |
| `IMG_SIZE`          | 512 → 384 → 512 | Прогрессивное изменение |
| `lr`                |           1 e-3 | Базовый LR              |
| `weight_decay`      |           1 e-4 | Регуляризация           |
| `EMA_DECAY`         |          0.9999 | «Инерция» EMA           |
| `num_workers`       |         0 (→ 2) | Стабильность → скорость |


---
## 🗂️ Артефакты

| Файл                                  | Содержимое                                       |
| ------------------------------------- | ------------------------------------------------ |
| `checkpoints/best_detector.pth`       | Веса с наилучшим mAP\@0.50 (сохр. каждые 5 эпох) |
| `checkpoints/swa_final.pth`           | Итоговая SWA-усреднённая модель                  |
| `train_videos.txt` / `val_videos.txt` | Списки роликов для воспроизводимости             |
| `seen_videos.txt`                     | Видео, уже использованные в прошлых запусках     |
