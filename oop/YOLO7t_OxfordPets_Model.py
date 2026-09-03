import os
import sys
import random
import numpy as np
import csv
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm

import tensorflow as tf
import keras_cv

# ---------------------------------------------------------
# Server Optimizations
# Uncomment the following lines if running on AUTH HPC.
# ---------------------------------------------------------
# os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# tf.config.optimizer.set_jit(False)

# Add the parent directory to the path to locate the 'my_models' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from my_models.yolo_model7 import build_yolo7_model


# ---------------------------------------------------------
# YOLOv7 Loss Functions (CIoU Component)
# ---------------------------------------------------------
def bbox_ciou(b1_xy, b1_wh, b2_xy, b2_wh):
    b1_xmin, b1_ymin = b1_xy[..., 0] - b1_wh[..., 0] / 2.0, b1_xy[..., 1] - b1_wh[..., 1] / 2.0
    b1_xmax, b1_ymax = b1_xy[..., 0] + b1_wh[..., 0] / 2.0, b1_xy[..., 1] + b1_wh[..., 1] / 2.0
    b2_xmin, b2_ymin = b2_xy[..., 0] - b2_wh[..., 0] / 2.0, b2_xy[..., 1] - b2_wh[..., 1] / 2.0
    b2_xmax, b2_ymax = b2_xy[..., 0] + b2_wh[..., 0] / 2.0, b2_xy[..., 1] + b2_wh[..., 1] / 2.0

    inter_area = tf.maximum(tf.minimum(b1_xmax, b2_xmax) - tf.maximum(b1_xmin, b2_xmin), 0.0) * \
                 tf.maximum(tf.minimum(b1_ymax, b2_ymax) - tf.maximum(b1_ymin, b2_ymin), 0.0)

    iou = inter_area / tf.maximum((b1_wh[..., 0] * b1_wh[..., 1]) + (b2_wh[..., 0] * b2_wh[..., 1]) - inter_area, 1e-7)

    c_squared = tf.square(tf.maximum(b1_xmax, b2_xmax) - tf.minimum(b1_xmin, b2_xmin)) + \
                tf.square(tf.maximum(b1_ymax, b2_ymax) - tf.minimum(b1_ymin, b2_ymin))

    center_dist_squared = tf.square(b1_xy[..., 0] - b2_xy[..., 0]) + tf.square(b1_xy[..., 1] - b2_xy[..., 1])
    factor = (4.0 / (np.pi ** 2))

    v = factor * tf.square(tf.math.atan(b1_wh[..., 0] / tf.maximum(b1_wh[..., 1], 1e-7)) - \
                           tf.math.atan(b2_wh[..., 0] / tf.maximum(b2_wh[..., 1], 1e-7)))

    alpha = v / tf.maximum((1.0 - iou) + v, 1e-7)

    return iou - (center_dist_squared / tf.maximum(c_squared, 1e-7)) - alpha * v


# ---------------------------------------------------------
# Main API Class: YOLO7t_OxfordPets
# ---------------------------------------------------------
class YOLO7t_OxfordPets:

    def __init__(self):
        self.name = 'YOLOv7_Tiny_OxfordPets'

        # Fine-tuning parameters designed for pruning recovery
        self.fine_tune_epochs = 5
        self.batch_size = 16

        # Conservative learning rate to prevent catastrophic forgetting during pruning recovery
        # Using Adam as per the original training script for YOLOv7, but with small fixed LR
        self.initial_learning_rate = 1e-4
        self.loss_function = 'YOLOv7_Loss (CIoU + Objectness + BCE)'

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.initial_learning_rate
        )

        self.metrics = ['mAP_50', 'mAP_50_95']
        self.best_model_metrics = {'monitor': 'mAP_50', 'mode': 'max'}
        self.header = ['train_loss', 'val_loss', 'mAP_50', 'mAP_50_95']

        # Network architecture parameters (YOLOv7-Tiny specific)
        self.target_size = (416, 416)
        self.grid_sizes = [52, 26, 13]
        self.classes = ["dog", "cat"]
        self.num_classes = len(self.classes)
        self.class_to_id = {name.lower(): i for i, name in enumerate(self.classes)}

        # YOLOv7 Anchors
        self.anchors = [
            [(10, 13), (16, 30), (33, 23)],  # P3 (for 52x52)
            [(30, 61), (62, 45), (59, 119)],  # P4 (for 26x26)
            [(116, 90), (156, 198), (373, 326)]  # P5 (for 13x13)
        ]

        # Paths (Relative to the 'oop' directory)
        self.image_dir = "../datasets/oxford_pets/images"
        self.annot_path = "../datasets/oxford_pets/annotations/xmls"

        # Pretrained Weights & Logs
        self.coco_pretrained_weights = "../weights/yolov7_tiny_coco.h5"
        self.baseline_model_weights = "../trained_models/yolo7t/oxford/yolov7_tiny_Oxford_Pets_Pretrained.weights.h5"

        self.log_dir = "../logs/oop_experiments/oxford_v7"
        os.makedirs(self.log_dir, exist_ok=True)

        self.model_architecture = None
        self.model_history = None
        self.train_dataset = None
        self.val_dataset = None
        self.all_image_files = []

        print(f"\n[INFO] Initialized {self.name} API for Pruning Recovery.")
        print(f"       - Recovery Epochs: {self.fine_tune_epochs}")
        print(f"       - Batch Size: {self.batch_size}")
        print(f"       - Target Size: {self.target_size}")
        print(f"       - Recovery Learning Rate: {self.initial_learning_rate}")

    def build(self, print_summary=True):
        print(f"[INFO] Constructing architecture for {self.name}...")

        model = build_yolo7_model(variant="tiny", input_shape=(416, 416, 3), num_classes=self.num_classes)
        model._name = self.name

        if print_summary:
            model.summary()
            print()

        self.model_architecture = model
        return model

    def data_preprocessing(self):
        print("[INFO] Initializing dataset generator (Mosaic & Letterbox)...")

        if os.path.exists(self.image_dir):
            self.all_image_files = [f for f in os.listdir(self.image_dir) if f.lower().endswith('.jpg')]

        if not self.all_image_files:
            print(f"[ERROR] No images found at {self.image_dir}. Please verify dataset path.")
            sys.exit(1)

        split_index = int(len(self.all_image_files) * 0.8)

        def load_raw_image_and_boxes(img_file):
            img_path = os.path.join(self.image_dir, img_file)
            xml_name = img_file.rsplit('.', 1)[0] + '.xml'
            xml_path = os.path.join(self.annot_path, xml_name)

            if not os.path.exists(xml_path):
                return None, None, None

            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                boxes, classes = [], []
                for obj in root.findall('object'):
                    name = obj.find('name').text.lower().strip()
                    if name not in self.class_to_id: continue
                    current_class = self.class_to_id[name]
                    bndbox = obj.find('bndbox')
                    boxes.append(
                        [float(bndbox.find('xmin').text), float(bndbox.find('ymin').text),
                         float(bndbox.find('xmax').text),
                         float(bndbox.find('ymax').text)])
                    classes.append(current_class)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img, np.array(boxes, dtype=np.float32), np.array(classes, dtype=np.int32)
            except Exception:
                return None, None, None

        def load_mosaic_dataset(img_file):
            w_target, h_target = self.target_size
            xc = int(random.uniform(w_target // 4, 3 * w_target // 4))
            yc = int(random.uniform(h_target // 4, 3 * h_target // 4))
            indices = [img_file] + random.sample(self.all_image_files, 3)
            mosaic_img = np.full((h_target, w_target, 3), 128, dtype=np.uint8)
            mosaic_boxes, mosaic_classes = [], []

            for i, file in enumerate(indices):
                img, boxes, classes = load_raw_image_and_boxes(file)
                if img is None or len(boxes) == 0: continue
                h, w, _ = img.shape

                if i == 0:
                    x1_a, y1_a, x2_a, y2_a = 0, 0, xc, yc;
                    x1_b, y1_b, x2_b, y2_b = w - xc, h - yc, w, h
                elif i == 1:
                    x1_a, y1_a, x2_a, y2_a = xc, 0, w_target, yc;
                    x1_b, y1_b, x2_b, y2_b = 0, h - yc, w_target - xc, h
                elif i == 2:
                    x1_a, y1_a, x2_a, y2_a = 0, yc, xc, h_target;
                    x1_b, y1_b, x2_b, y2_b = w - xc, 0, w, h_target - yc
                elif i == 3:
                    x1_a, y1_a, x2_a, y2_a = xc, yc, w_target, h_target;
                    x1_b, y1_b, x2_b, y2_b = 0, 0, w_target - xc, h_target - yc

                x1_b, y1_b = max(0, x1_b), max(0, y1_b)
                x2_b, y2_b = min(w, x2_b), min(h, y2_b)
                pad_w, pad_h = (x2_a - x1_a), (y2_a - y1_a)

                if x2_b - x1_b > pad_w: x2_b = x1_b + pad_w
                if y2_b - y1_b > pad_h: y2_b = y1_b + pad_h

                mosaic_img[y1_a:y1_a + (y2_b - y1_b), x1_a:x1_a + (x2_b - x1_b)] = img[y1_b:y2_b, x1_b:x2_b]
                pad_x, pad_y = x1_a - x1_b, y1_a - y1_b

                for b_idx, box in enumerate(boxes):
                    xmin, ymin, xmax, ymax = box[0] + pad_x, box[1] + pad_y, box[2] + pad_x, box[3] + pad_y
                    xmin, ymin = max(x1_a, min(xmin, x2_a)), max(y1_a, min(ymin, y2_a))
                    xmax, ymax = max(x1_a, min(xmax, x2_a)), max(y1_a, min(ymax, y2_a))
                    if (xmax - xmin) > 5 and (ymax - ymin) > 5:
                        mosaic_boxes.append(
                            [(xmin + xmax) / 2 / w_target, (ymin + ymax) / 2 / h_target, (xmax - xmin) / w_target,
                             (ymax - ymin) / h_target])
                        mosaic_classes.append(classes[b_idx])

            return mosaic_img.astype(np.float32) / 255.0, mosaic_boxes, mosaic_classes

        def letterbox_image(img_file):
            img, boxes, classes = load_raw_image_and_boxes(img_file)
            if img is None: return None, None, None
            shape = img.shape[:2]
            r = min(self.target_size[0] / shape[0], self.target_size[1] / shape[1])
            new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
            dw, dh = (self.target_size[1] - new_unpad[0]) / 2, (self.target_size[0] - new_unpad[1]) / 2

            if shape[::-1] != new_unpad:
                img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128))

            transformed_boxes = []
            for b in boxes:
                transformed_boxes.append(
                    [(((b[0] + b[2]) / 2) * r + left) / self.target_size[1],
                     (((b[1] + b[3]) / 2) * r + top) / self.target_size[0],
                     ((b[2] - b[0]) * r) / self.target_size[1], ((b[3] - b[1]) * r) / self.target_size[0]])
            return img.astype(np.float32) / 255.0, transformed_boxes, classes

        def build_targets(boxes, classes):
            target_grids = [
                np.zeros((self.grid_sizes[i], self.grid_sizes[i], 3, 5 + self.num_classes), dtype=np.float32) for i in
                range(3)]
            anchor_ratio_thresh = 4.0

            for i in range(len(boxes)):
                cx, cy, w, h = boxes[i]
                c, w_pixels, h_pixels = int(classes[i]), w * self.target_size[0], h * self.target_size[1]
                for scale_idx in range(3):
                    grid_size = self.grid_sizes[scale_idx]
                    grid_x, grid_y = cx * grid_size, cy * grid_size
                    for anchor_idx in range(3):
                        aw, ah = self.anchors[scale_idx][anchor_idx]
                        rw, rh = w_pixels / aw, h_pixels / ah
                        if max(max(rw, 1 / rw), max(rh, 1 / rh)) < anchor_ratio_thresh:
                            gi_x, gi_y = int(grid_x), int(grid_y)
                            if 0 <= gi_x < grid_size and 0 <= gi_y < grid_size:
                                target_grids[scale_idx][gi_y, gi_x, anchor_idx, 0:5] = [grid_x - gi_x, grid_y - gi_y,
                                                                                        np.log(w_pixels / aw),
                                                                                        np.log(h_pixels / ah), 1.0]
                                target_grids[scale_idx][gi_y, gi_x, anchor_idx, 5 + c] = 1.0

                            # YOLOv7 Adjacent Grid Cell Assigment
                            for nx, ny in [(gi_x + (1 if (grid_x - gi_x) > 0.5 else -1), gi_y),
                                           (gi_x, gi_y + (1 if (grid_y - gi_y) > 0.5 else -1))]:
                                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                                    target_grids[scale_idx][ny, nx, anchor_idx, 0:5] = [grid_x - nx, grid_y - ny,
                                                                                        np.log(w_pixels / aw),
                                                                                        np.log(h_pixels / ah), 1.0]
                                    target_grids[scale_idx][ny, nx, anchor_idx, 5 + c] = 1.0

            return (target_grids[0].reshape((self.grid_sizes[0], self.grid_sizes[0], 3 * (5 + self.num_classes))),
                    target_grids[1].reshape((self.grid_sizes[1], self.grid_sizes[1], 3 * (5 + self.num_classes))),
                    target_grids[2].reshape((self.grid_sizes[2], self.grid_sizes[2], 3 * (5 + self.num_classes))))

        def dataset_generator():
            for i, img_file in enumerate(self.all_image_files):
                is_val = 0 if i < split_index else 1
                if is_val == 0:
                    if random.random() > 0.3:
                        image, boxes, classes = load_mosaic_dataset(img_file)
                    else:
                        image, boxes, classes = letterbox_image(img_file)
                        if image is not None and random.random() > 0.5:
                            image = np.fliplr(image)
                            for b in boxes: b[0] = 1.0 - b[0]
                else:
                    image, boxes, classes = letterbox_image(img_file)

                if image is None or len(boxes) == 0: continue

                boxes_padded = np.zeros((100, 4), dtype=np.float32)
                classes_padded = np.zeros((100,), dtype=np.float32) - 1
                num_boxes = min(len(boxes), 100)
                boxes_padded[:num_boxes] = np.array(boxes[:num_boxes], dtype=np.float32)
                classes_padded[:num_boxes] = np.array(classes[:num_boxes], dtype=np.float32)
                t1, t2, t3 = build_targets(boxes, classes)

                yield image, boxes_padded, classes_padded, t1, t2, t3, is_val

        @tf.function
        def augment_colors(image, t1, t2, t3):
            image = tf.image.random_brightness(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
            image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
            return tf.clip_by_value(image, 0.0, 1.0), t1, t2, t3

        dataset = tf.data.Dataset.from_generator(
            dataset_generator,
            output_signature=(
                tf.TensorSpec(shape=(416, 416, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(100, 4), dtype=tf.float32),
                tf.TensorSpec(shape=(100,), dtype=tf.float32),
                tf.TensorSpec(shape=(self.grid_sizes[0], self.grid_sizes[0], 3 * (5 + self.num_classes)),
                              dtype=tf.float32),
                tf.TensorSpec(shape=(self.grid_sizes[1], self.grid_sizes[1], 3 * (5 + self.num_classes)),
                              dtype=tf.float32),
                tf.TensorSpec(shape=(self.grid_sizes[2], self.grid_sizes[2], 3 * (5 + self.num_classes)),
                              dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        )

        self.train_dataset = dataset.filter(lambda *args: args[-1] == 0).map(
            lambda img, box, cls, t1, t2, t3, is_val: (img, t1, t2, t3)).map(
            augment_colors, num_parallel_calls=tf.data.AUTOTUNE).batch(self.batch_size, drop_remainder=True).prefetch(
            tf.data.AUTOTUNE)

        self.val_dataset = dataset.filter(lambda *args: args[-1] == 1).map(
            lambda img, box, cls, t1, t2, t3, is_val: (img, box, cls, t1, t2, t3)).batch(self.batch_size,
                                                                                         drop_remainder=True).prefetch(
            tf.data.AUTOTUNE)

        print(f"[INFO] Dataset successfully loaded. Total samples: {len(self.all_image_files)}")
        return self.train_dataset, self.val_dataset

    # ---------------------------------------------------------
    # Training Loop (Bypassed since baseline model is provided)
    # ---------------------------------------------------------
    def train(self, fit_flag=True, print_summary=False):
        print("[INFO] Primary training method bypassed.")
        print(f"[INFO] The baseline model should be loaded from: {self.baseline_model_weights}")
        return None

    # ---------------------------------------------------------
    # Fine-Tuning Loop (Pruning Recovery Mechanism)
    # ---------------------------------------------------------
    def fine_tune(self, path_to_save, new_model=None, print_summary=False, new_fine_tune_epochs=None):
        print("\n" + "=" * 50)
        print("[INFO] Initiating Pruning Recovery (Fine-Tuning) phase...")

        if new_model is None:
            print("[ERROR] The fine_tune method requires an external 'new_model' (e.g., pruned model) to operate.")
            return None, None

        print("[INFO] External pruned model loaded successfully.")
        model = new_model

        fine_tune_epochs = new_fine_tune_epochs if new_fine_tune_epochs else self.fine_tune_epochs

        if not self.train_dataset:
            self.data_preprocessing()

        # Build optimizer with model variables
        dummy_img = tf.zeros((1, 416, 416, 3))
        _ = model(dummy_img, training=True)
        self.optimizer.build(model.trainable_variables)

        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        coco_metric = keras_cv.metrics.BoxCOCOMetrics(bounding_box_format="center_xyWH", evaluate_freq=1)

        @tf.function(jit_compile=False)
        def compute_loss(targets, model_outputs):
            loss = 0.0
            for i, grid_pred in enumerate(model_outputs):
                target_grid, batch_size, grid_size = targets[i], tf.shape(grid_pred)[0], self.grid_sizes[i]

                pred_reshaped = tf.reshape(grid_pred, [batch_size, grid_size, grid_size, 3, 5 + self.num_classes])
                true_reshaped = tf.reshape(target_grid, [batch_size, grid_size, grid_size, 3, 5 + self.num_classes])

                pred_xy, pred_wh, pred_obj, pred_cls = pred_reshaped[..., 0:2], pred_reshaped[..., 2:4], pred_reshaped[
                    ..., 4:5], pred_reshaped[..., 5:]
                t_xy, t_wh, t_obj, t_cls = true_reshaped[..., 0:2], true_reshaped[..., 2:4], true_reshaped[..., 4:5], \
                true_reshaped[..., 5:]

                obj_mask_sq, noobj_mask_sq = tf.squeeze(t_obj, axis=-1), tf.squeeze(1.0 - t_obj, axis=-1)

                obj_loss = tf.reduce_mean(bce(t_obj, pred_obj) * obj_mask_sq) + 0.5 * tf.reduce_mean(
                    bce(t_obj, pred_obj) * noobj_mask_sq)
                class_loss = tf.reduce_mean(bce(t_cls, pred_cls) * obj_mask_sq)

                grid_x, grid_y = tf.reshape(
                    tf.meshgrid(tf.range(grid_size, dtype=tf.float32), tf.range(grid_size, dtype=tf.float32))[0],
                    [1, grid_size, grid_size, 1, 1]), tf.reshape(
                    tf.meshgrid(tf.range(grid_size, dtype=tf.float32), tf.range(grid_size, dtype=tf.float32))[1],
                    [1, grid_size, grid_size, 1, 1])

                anchor_tensor = tf.reshape(tf.constant(self.anchors[i], dtype=tf.float32), [1, 1, 1, 3, 2])

                pred_xy_decoded = (tf.math.sigmoid(pred_xy) + tf.concat([grid_x, grid_y], axis=-1)) / float(grid_size)
                pred_wh_decoded = (tf.math.exp(tf.clip_by_value(pred_wh, -10.0, 10.0)) * anchor_tensor) / float(
                    self.target_size[0])
                t_xy_decoded = (t_xy + tf.concat([grid_x, grid_y], axis=-1)) / float(grid_size)
                t_wh_decoded = (tf.math.exp(t_wh) * anchor_tensor) / float(self.target_size[0])

                ciou = bbox_ciou(pred_xy_decoded, pred_wh_decoded, t_xy_decoded, t_wh_decoded)
                ciou_loss = tf.reduce_mean((1.0 - ciou) * obj_mask_sq)

                loss += obj_loss + 5.0 * ciou_loss + class_loss
            return loss

        @tf.function
        def decode_preds(model_outputs):
            all_boxes, all_scores = [], []
            for i, grid in enumerate(model_outputs):
                batch_size, grid_size = tf.shape(grid)[0], self.grid_sizes[i]
                grid_reshaped = tf.reshape(grid, [batch_size, grid_size, grid_size, 3, 5 + self.num_classes])

                grid_x, grid_y = tf.reshape(
                    tf.meshgrid(tf.range(grid_size, dtype=tf.float32), tf.range(grid_size, dtype=tf.float32))[0],
                    [1, grid_size, grid_size, 1]), tf.reshape(
                    tf.meshgrid(tf.range(grid_size, dtype=tf.float32), tf.range(grid_size, dtype=tf.float32))[1],
                    [1, grid_size, grid_size, 1])

                anchor_tensor = tf.reshape(tf.constant(self.anchors[i], dtype=tf.float32), [1, 1, 1, 3, 2])
                anchor_w, anchor_h = anchor_tensor[..., 0], anchor_tensor[..., 1]

                pred_x = ((tf.math.sigmoid(grid_reshaped[..., 0]) + grid_x) / float(grid_size)) * self.target_size[0]
                pred_y = ((tf.math.sigmoid(grid_reshaped[..., 1]) + grid_y) / float(grid_size)) * self.target_size[1]
                pred_w = tf.math.exp(tf.clip_by_value(grid_reshaped[..., 2], -10.0, 10.0)) * anchor_w
                pred_h = tf.math.exp(tf.clip_by_value(grid_reshaped[..., 3], -10.0, 10.0)) * anchor_h

                pred_y1 = pred_y - pred_h / 2.0
                pred_x1 = pred_x - pred_w / 2.0
                pred_y2 = pred_y + pred_h / 2.0
                pred_x2 = pred_x + pred_w / 2.0

                boxes = tf.stack([pred_y1, pred_x1, pred_y2, pred_x2], axis=-1)
                all_boxes.append(tf.reshape(boxes, [batch_size, -1, 4]))

                obj_conf = tf.math.sigmoid(grid_reshaped[..., 4:5])
                class_probs = tf.math.sigmoid(grid_reshaped[..., 5:])
                scores = obj_conf * class_probs
                all_scores.append(tf.reshape(scores, [batch_size, -1, self.num_classes]))

            final_boxes_yxyx = tf.concat(all_boxes, axis=1)
            final_scores = tf.concat(all_scores, axis=1)

            boxes_exp = tf.expand_dims(final_boxes_yxyx, axis=2)
            nms_out = tf.image.combined_non_max_suppression(
                boxes=boxes_exp,
                scores=final_scores,
                max_output_size_per_class=100,
                max_total_size=100,
                iou_threshold=0.60,
                score_threshold=0.001,
                clip_boxes=False
            )

            y1, x1, y2, x2 = tf.split(nms_out.nmsed_boxes, 4, axis=-1)
            nms_boxes_cyxhwh = tf.concat([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=-1)

            mask = tf.sequence_mask(nms_out.valid_detections, maxlen=100)
            final_classes_masked = tf.where(mask, nms_out.nmsed_classes, tf.fill(tf.shape(nms_out.nmsed_classes), -1.0))

            return {"boxes": nms_boxes_cyxhwh, "classes": final_classes_masked, "confidence": nms_out.nmsed_scores}

        @tf.function(jit_compile=False)
        def train_step(images, targets):
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = compute_loss(targets, predictions)
            self.optimizer.apply_gradients(
                zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
            return loss

        os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
        history_dict = {key: [] for key in self.header}

        for epoch in range(fine_tune_epochs):
            train_loss_total, num_train_batches = 0.0, 0

            pbar_train = tqdm(self.train_dataset, desc=f"Epoch {epoch + 1}/{fine_tune_epochs} [Train]", leave=True)
            for batch_images, t1, t2, t3 in pbar_train:
                loss = train_step(batch_images, [t1, t2, t3])
                train_loss_total += float(loss)
                num_train_batches += 1
                pbar_train.set_postfix(Loss=f"{train_loss_total / num_train_batches:.4f}")

            val_loss_total, num_val_batches = 0.0, 0
            coco_metric.reset_state()
            pbar_val = tqdm(self.val_dataset, desc=f"Epoch {epoch + 1}/{fine_tune_epochs} [Eval]", leave=False)

            for val_images, val_boxes, val_classes, t1, t2, t3 in pbar_val:
                val_predictions = model(val_images, training=False)
                val_loss_total += float(compute_loss([t1, t2, t3], val_predictions))
                num_val_batches += 1

                y_pred_dict = decode_preds(val_predictions)
                coco_metric.update_state({"boxes": val_boxes * [416, 416, 416, 416], "classes": val_classes},
                                         y_pred_dict)

            metric_res = coco_metric.result()
            map_50 = float(metric_res['MaP@[IoU=50]'])
            map_50_95 = float(metric_res['MaP'])

            t_loss = train_loss_total / max(1, num_train_batches)
            v_loss = val_loss_total / max(1, num_val_batches)

            history_dict['train_loss'].append(t_loss)
            history_dict['val_loss'].append(v_loss)
            history_dict['mAP_50'].append(map_50)
            history_dict['mAP_50_95'].append(map_50_95)

            print(
                f"[METRICS] Epoch {epoch + 1} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | mAP@0.50: {map_50:.4f}")

            if epoch == 0 or map_50 >= max(history_dict['mAP_50'][:-1]):
                model.save_weights(path_to_save)

        best_info = self.get_best_epoch_info(history_dict)
        print(f"\n[INFO] Pruning Recovery Completed. Best Metrics: {best_info}")

        new_model = build_yolo7_model(variant="tiny", input_shape=(416, 416, 3), num_classes=self.num_classes)
        new_model.load_weights(path_to_save)

        self.model_history = history_dict
        return new_model, best_info

    def get_best_epoch_info(self, history, metrics=None):
        hist_dict = history.history if hasattr(history, 'history') else history
        monitor, mode = self.best_model_metrics['monitor'], self.best_model_metrics['mode']
        metric_per_epoch = hist_dict[monitor]

        if mode == 'max':
            best_epoch_ix = np.argmax(metric_per_epoch)
        elif mode == 'min':
            best_epoch_ix = np.argmin(metric_per_epoch)
        else:
            raise ValueError('Mode should be either min or max')

        new_history = {key: hist_dict[key][best_epoch_ix] for key in self.header if key in hist_dict}
        if metrics: new_history = dict(new_history, **dict(metrics))
        new_history['best_epoch'] = best_epoch_ix + 1
        return new_history

    def extract_monitoring_metric_value(self, model=None):
        if not self.model_history:
            print("[WARNING] No training history found. Ensure fine_tune was executed.")
            return 0.0

        best_info = self.get_best_epoch_info(self.model_history)
        monitor_metric = self.best_model_metrics['monitor']

        if monitor_metric in best_info:
            metric_val = best_info[monitor_metric]
            print(f"[EVALUATION] Retained metric '{monitor_metric}': {metric_val:.4f}")
            return metric_val
        else:
            print(f"[ERROR] Metric '{monitor_metric}' not found in history.")
            return 0.0