## RF-DETR Pretrained Weights

The RF-DETR model weights (~122 MB) are available as a GitHub Release asset. You can download them directly here:

[🔗 rf_detr_803.pt (122 MB)](https://github.com/vkalinovski/-Basketball_foul_detection/releases/download/RF-DETR_WEIGHTS/rf_detr_803.pt)

# RF-DETR: Обучение и аннотация видео

Добро пожаловать в репозиторий **RF-DETR** — архитектуру на основе Transformer для детектирования и трекинга объектов на видео c использованием радиочастотных признаков.

---

## 📂 Структура проекта

    ├── rf-detr-train.ipynb        # Ноутбук для обучения модели RF-DETR
    ├── video_annotation.ipynb     # Ноутбук для разметки и обработки видео
    ├── configs/                   # Конфигурационные файлы и гиперпараметры
    │   ├── default.yaml           # Основные параметры обучения
    │   └── radar_features.yaml    # Параметры радиочастотных признаков
    ├── data/                      # Исходные видео и разметка
    │   ├── raw/                   # Сырые видеофайлы (.mp4, .avi …)
    │   └── labels/                # Размеченные кадры в формате COCO (.json)
    ├── outputs/                   # Результаты обучения и вывода моделей
    │   ├── checkpoints/           # Сохранённые контрольные точки (.pth)
    │   └── annotated_videos/      # Видео с результатами детекции (.mp4)
    └── README.md                  # Этот файл

---

## 🧠 Модель: RF-DETR

**RF-DETR (Radar Frequency Detection Transformer)** объединяет свёрточные сети и Transformer для:

- **Высокоточной локализации** объектов в видеокадрах  
- **Стабильного трекинга** при низком качестве изображения  
- **Интеграции радиочастотных признаков** для повышения надёжности  

Эксперименты выполнены на **PyTorch** с **Detectron2**. Доступны backbone-ы *ResNet-50* и *ResNet-101*.

---

## 🚀 Обучение модели (`rf-detr-train.ipynb`)

1. **Подготовка данных**  
   • Импорт видео из `data/raw/`  
   • Экспорт ключевых кадров и генерация аннотаций COCO → `data/labels/`  

2. **Конфигурация гиперпараметров**  
   • Пути к данным, `batch_size`, `learning_rate`, число эпох  
   • Выбор backbone и радиочастотного конфигурационного файла  

3. **Тренировка**  
   • Инициализация RF-DETR  
   • Автосохранение чекпоинтов в `outputs/checkpoints/`  
   • Логирование *loss* и *mAP*  

4. **Валидация**  
   • Запуск на валидационном наборе  
   • Метрики `mAP`, `AR`, время инференса  

---

## 🎥 Аннотация видео (`video_annotation.ipynb`)

1. **Загрузка модели**  
   • Чекпоинт: `outputs/checkpoints/latest.pth`  

2. **Процесс аннотации**  
   • Кадровая обработка видео из `data/raw/`  
   • Детекция и формирование разметки (COCO или YOLO)  

3. **Визуализация**  
   • Bounding-boxes, метки, confidence-score  
   • (Опционально) ID трекинга  
   • Сохранение видео в `outputs/annotated_videos/`  

4. **Пример использования**  

        from video_annotation import annotate_video

        annotate_video(
            model_path='outputs/checkpoints/latest.pth',
            input_video='data/raw/sample.mp4',
            output_video='outputs/annotated_videos/sample_annotated.mp4',
            annotation_format='coco',
            show_tracking_ids=True
        )
