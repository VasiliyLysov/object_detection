# Anchor-Free Object Detection Utils

Набор функций для обучения и визуализации моделей **anchor-free object detection** в TensorFlow.
Подходит для учебных проектов и проектов по компьютерному зрению.

## 📦 Возможности

* Загрузка и распаковка датасетов
* Чтение аннотаций и изображений
* Применение аугментаций (Albumentations)
* Формирование `tf.data.Dataset`
* Подготовка батчей для anchor-free моделей
* Декодирование предсказаний модели

## 🚀 Установка

Склонируйте репозиторий и импортируйте модуль:

```python
!git clone https://github.com/username/object_detection.git
import sys
sys.path.append("/content/object_detection")
import detection_utils as du
```

## 🧠 Пример использования

```python
images, annots = du.img_paths_and_annot("data/images", "data/labels")
dataset_train = du.make_dataset(images, annots, transform, batch_size=16, num_classes=2, aug=True)
```

## ⚙️ Зависимости

* TensorFlow
* Albumentations
* Pillow
* NumPy
* zipfile
* os
