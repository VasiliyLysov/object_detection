import zipfile
import os

import numpy as np
from sklearn.utils import shuffle
from PIL import Image
import tensorflow as tf
import albumentations as A
from tensorflow.image import combined_non_max_suppression

# Распаковка zip
def zip_extrcat(file):
  with zipfile.ZipFile(f'{file}.zip', 'r') as zip_ref:
      zip_ref.extractall(file)

# Аннотации и пути к изображениям
def img_paths_and_annot(root_folder, labels_folder=None):
    images, annots = [], []

    for folder in os.listdir(root_folder):
        if folder.startswith(('.', '_')):
            continue
        folder_path = os.path.join(root_folder, folder)
        print(folder_path)

        for file in os.listdir(folder_path):
            if file.startswith(('.', '_')):
                continue
            images.append(os.path.join(folder_path, file))

            if labels_folder:
                label_file = os.path.join(labels_folder, f'{file[:-4]}.txt')
                with open(label_file) as f:
                    lines = f.read().strip().splitlines()
                    annots.append(np.array([list(map(float, l.split())) for l in lines], dtype=np.float32))
            else:
                annots.append(np.zeros((0,5), dtype=np.float32))

    return shuffle(np.array(images), np.array(annots), random_state=8)

# Применение агументации
def load_data(path, annotations, transform, aug=True,):
    image = np.array(Image.open(path).convert("RGB").resize((224, 224)))

    if annotations.shape[0] == 0:
        return image.astype(np.float32) / 255.0, np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    labels = annotations[:, 0].astype(np.int32)
    bboxes = annotations[:, 1:].astype(np.float32)

    if aug and len(bboxes) > 0:
        aug_data = transform(image=image, bboxes=bboxes.tolist(), class_labels=labels.tolist())
        image = aug_data["image"]
        bboxes = np.array(aug_data["bboxes"], dtype=np.float32)
        labels = np.array(aug_data["class_labels"], dtype=np.int32)

    return image.astype(np.float32) / 255.0, bboxes, labels

# функция для подачи изображений и аннотация по одному объекту за раз
def data_generator(img_paths, annotations, transform, aug=True, num_classes=2):
    for path, annots in zip(img_paths, annotations):
        img, bboxes, labels = load_data(path, annots, transform, aug)

        # если нет объектов, создаем массив правильной формы
        if len(bboxes) == 0:
            bboxes = np.zeros((0,4), dtype=np.float32)
            labels_one_hot = np.zeros((0, num_classes), dtype=np.float32)
            has_objects = tf.constant(0.0, tf.float32)
        else:
            labels_one_hot = tf.one_hot(labels, num_classes, dtype=tf.float32)
            has_objects = tf.constant(1.0, tf.float32)

        yield img, {"bboxes": bboxes, "labels": labels_one_hot, "has_objects": has_objects}

def make_dataset(img_paths, annotations, transform, batch_size=8, num_classes=2, aug=True):
    ds = tf.data.Dataset.from_generator(
        lambda: data_generator(img_paths, annotations, transform, aug, num_classes),
        output_signature=(
            tf.TensorSpec((224,224,3), tf.float32),
            {
                "bboxes": tf.TensorSpec((None,4), tf.float32),
                "labels": tf.TensorSpec((None,num_classes), tf.float32),
                "has_objects": tf.TensorSpec((), tf.float32)
            }
        )
    )
    if aug: ds = ds.shuffle(500)
    ds = ds.padded_batch(batch_size,
        padded_shapes=((224,224,3), {"bboxes": (None,4), "labels": (None,num_classes), "has_objects": ()}),
        padding_values=(0.0, {"bboxes":0.0, "labels":0.0, "has_objects":0.0})
    )
    if aug: ds = ds.repeat()
    return ds.prefetch(tf.data.AUTOTUNE)


def preprocess_batch(images, targets, fm_shape = (7, 7), num_classes = 2):
    batch_size = tf.shape(images)[0]
    fm_height, fm_width = fm_shape

    bbox_targets = tf.zeros((batch_size, fm_height, fm_width, 4), dtype=tf.float32)
    class_targets = tf.zeros((batch_size, fm_height, fm_width, num_classes), dtype=tf.float32)
    objectness_targets = tf.zeros((batch_size, fm_height, fm_width, 1), dtype=tf.float32)

    for i in tf.range(batch_size):
        if targets['has_objects'][i] > 0:
            bboxes = targets['bboxes'][i]
            labels = targets['labels'][i]

            # Фильтруем реальные объекты (не padding)
            mask = tf.reduce_sum(tf.abs(bboxes), axis=1) > 0
            bboxes = tf.boolean_mask(bboxes, mask)
            labels = tf.boolean_mask(labels, mask)

            for obj_idx in tf.range(tf.shape(bboxes)[0]):
                bbox = bboxes[obj_idx]
                label = labels[obj_idx]

                x_center, y_center, w, h = bbox[0], bbox[1], bbox[2], bbox[3]

                # Определяем ячейку на feature map
                fm_x = tf.cast(x_center * fm_width, tf.int32)
                fm_y = tf.cast(y_center * fm_height, tf.int32)
                fm_x = tf.clip_by_value(fm_x, 0, fm_width - 1)
                fm_y = tf.clip_by_value(fm_y, 0, fm_height - 1)

                # Относительные координаты центра внутри ячейки
                cell_center_x = (tf.cast(fm_x, tf.float32) + 0.5) / fm_width
                cell_center_y = (tf.cast(fm_y, tf.float32) + 0.5) / fm_height

                # Смещения относительно ячейки
                dx = (x_center - cell_center_x) * fm_width
                dy = (y_center - cell_center_y) * fm_height

                # Логарифмируем размеры для стабильности
                dw = tf.math.log(w * fm_width + 1e-8)
                dh = tf.math.log(h * fm_height + 1e-8)

                bbox_target = tf.stack([dx, dy, dw, dh])

                indices = tf.stack([i, fm_y, fm_x])

                bbox_targets = tf.tensor_scatter_nd_update(bbox_targets, [indices], [bbox_target])
                class_targets = tf.tensor_scatter_nd_update(class_targets, [indices], [label])
                objectness_targets = tf.tensor_scatter_nd_update(objectness_targets, [indices], [[1.0]])

    combined_targets = tf.concat([bbox_targets, class_targets, objectness_targets], axis=-1)

    return images, combined_targets

def decode_preds(preds, threshold=0.5, fm_size=7, max_objects=2):
    batch_size = tf.shape(preds)[0]
    bboxes_list, labels_list, obj_list = [], [], []

    # Координаты ячеек
    grid_y, grid_x = tf.meshgrid(tf.range(fm_size), tf.range(fm_size), indexing='ij')
    grid_x = tf.cast(grid_x, tf.float32)
    grid_y = tf.cast(grid_y, tf.float32)

    for i in tf.range(batch_size):
        pred = preds[i]
        obj = pred[..., 6]
        idx = tf.where(obj > threshold)

        # Если нет объектов
        if tf.shape(idx)[0] == 0:
            bboxes_list.append(tf.zeros((max_objects, 4)))
            labels_list.append(tf.zeros((max_objects, 2)))
            obj_list.append(0.0)
            continue

        selected = tf.gather_nd(pred, idx)
        dx, dy, dw, dh = tf.unstack(selected[:, 0:4], axis=-1)
        label = selected[:, 4:6]
        obj_score = selected[:, 6]
        cell_x = tf.cast(idx[:, 1], tf.float32)
        cell_y = tf.cast(idx[:, 0], tf.float32)

        # Восстановление координат
        x_center = (cell_x + 0.5 + dx) / fm_size
        y_center = (cell_y + 0.5 + dy) / fm_size
        w = tf.exp(dw) / fm_size
        h = tf.exp(dh) / fm_size

        bboxes = tf.stack([x_center, y_center, w, h], axis=-1)

        # --- выравнивание по размеру ---
        n = tf.shape(bboxes)[0]
        pad = tf.maximum(max_objects - n, 0)
        bboxes = tf.concat([bboxes[:max_objects], tf.zeros((pad, 4))], axis=0)[:max_objects]
        label = tf.concat([label[:max_objects], tf.zeros((pad, 2))], axis=0)[:max_objects]

        bboxes_list.append(bboxes)
        labels_list.append(label)
        obj_list.append(tf.reduce_mean(obj_score))

    return tf.stack(bboxes_list), tf.stack(labels_list), tf.stack(obj_list)

# NMS
def nms_predictions(preds, max_output_size_per_class=5, max_total_size=10, iou_threshold=0.3, score_threshold=0.3, max_objects = 49):
    pred_bb, pred_label, _ = decode_preds(preds, threshold=0, max_objects=max_objects)
    preds_cp = preds[..., -1]

    B, N, _ = pred_bb.shape
    num_classes = pred_label.shape[-1]

    conf = tf.reshape(preds_cp, [B, N, 1])
    scores = pred_label * conf

    # cx, cy, w, h → ymin, xmin, ymax, xmax
    cx, cy, w, h = tf.split(pred_bb, 4, axis=-1)
    boxes = tf.concat([cy - h/2, cx - w/2, cy + h/2, cx + w/2], axis=-1)

    all_boxes, all_classes, valid = [], [], []

    for b in range(B):
        bboxes, bclasses = [], []

        for c in range(num_classes):
            s = scores[b, :, c]
            mask = s > score_threshold
            if not tf.reduce_any(mask): 
                continue

            sel = tf.image.non_max_suppression(
                tf.boolean_mask(boxes[b], mask),
                tf.boolean_mask(s, mask),
                max_output_size=max_output_size_per_class,
                iou_threshold=iou_threshold
            )

            bboxes.append(tf.gather(tf.boolean_mask(boxes[b], mask), sel))
            bclasses.append(tf.ones_like(sel, dtype=tf.int32) * c)

        if bboxes:
            bboxes = tf.concat(bboxes, axis=0)[:max_total_size]
            bclasses = tf.concat(bclasses, axis=0)[:max_total_size]
        else:
            bboxes = tf.zeros((0, 4), tf.float32)
            bclasses = tf.zeros((0,), tf.int32)

        all_boxes.append(bboxes)
        all_classes.append(bclasses)
        valid.append(tf.shape(bboxes)[0])

    return all_boxes, all_classes, valid
