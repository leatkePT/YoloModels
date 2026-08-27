import os
import sys
import random
import numpy as np
import csv
import ast
import cv2
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
from my_models.yolo_model11 import build_yolo11_model


# ---------------------------------------------------------
# SOTA YOLOv11 Loss Functions (CIoU Component)
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
# Main API Class: YOLO11_GlobalWheat
# ---------------------------------------------------------
class YOLO11_GlobalWheat:

    def __init__(self):
        self.name = 'YOLO11_Nano_GlobalWheat'

        # Fine-tuning parameters designed for pruning recovery
        self.fine_tune_epochs = 5
        self.batch_size = 16

        # Conservative learning rate to prevent catastrophic forgetting during pruning recovery
        self.initial_learning_rate = 1e-4
        self.loss_function = 'YOLO_SOTA_Loss (CIoU + DFL + BCE)'

        self.optimizer = tf.keras.optimizers.SGD(
            learning_rate=self.initial_learning_rate,
            momentum=0.9,
            weight_decay=0.0005
        )

        self.metrics = ['mAP_50', 'mAP_50_95']
        self.best_model_metrics = {'monitor': 'mAP_50', 'mode': 'max'}
        self.header = ['train_loss', 'val_loss', 'mAP_50', 'mAP_50_95']

        # Network architecture parameters
        self.target_size = (640, 640)
        self.grid_sizes = [80, 40, 20]
        self.reg_max = 16
        self.classes = ["wheat_head"]
        self.num_classes = len(self.classes)
        self.class_to_id = {name.lower(): i for i, name in enumerate(self.classes)}

        # Paths (Relative to the 'oop' directory)
        self.image_dir = "../datasets/global_wheat/train"
        self.annot_path = "../datasets/global_wheat/train.csv"

        # Pretrained Weights & Logs
        self.coco_pretrained_weights = "../weights/yolo11n_coco_pretrained.weights.h5"
        self.baseline_model_weights = "../trained_models/yolo11n/wheathead/yolo11_nano_Global_Wheat_Preatrained.weights.h5"

        self.log_dir = "../logs/oop_experiments/wheat"
        os.makedirs(self.log_dir, exist_ok=True)

        self.model_architecture = None
        self.model_history = None
        self.train_dataset = None
        self.val_dataset = None

        # Specific to Global Wheat CSV parsing
        self.grouped_boxes = {}
        self.all_image_ids = []

        print(f"\n[INFO] Initialized {self.name} API for Pruning Recovery.")
        print(f"       - Recovery Epochs: {self.fine_tune_epochs}")
        print(f"       - Batch Size: {self.batch_size}")
        print(f"       - Recovery Learning Rate: {self.initial_learning_rate}")

    def build(self, print_summary=True):
        print(f"[INFO] Constructing architecture for {self.name}...")
        import my_models.yolo_model11 as yolo11_module
        yolo11_module.MODEL_CONFIGS["nano"]["nc"] = self.num_classes

        model = build_yolo11_model(variant="nano", input_shape=(640, 640, 3))
        model._name = self.name

        if print_summary:
            model.summary()
            print()

        self.model_architecture = model
        return model

    def data_preprocessing(self):
        print("[INFO] Initializing dataset generator (CSV Parsing, Mosaic & Letterbox)...")

        if os.path.exists(self.annot_path):
            with open(self.annot_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img_id = row['image_id']
                    if img_id not in self.grouped_boxes:
                        self.grouped_boxes[img_id] = []
                    bbox = ast.literal_eval(row['bbox'])
                    xmin, ymin, w, h = bbox
                    self.grouped_boxes[img_id].append([xmin, ymin, xmin + w, ymin + h])

            self.all_image_ids = list(self.grouped_boxes.keys())

        if len(self.all_image_ids) == 0:
            print(f"[ERROR] No annotations found at {self.annot_path}. Please verify dataset path.")
            sys.exit(1)

        split_index = int(len(self.all_image_ids) * 0.8)

        def load_raw_image_and_boxes(img_id):
            img_path = os.path.join(self.image_dir, img_id + ".jpg")
            if not os.path.exists(img_path) or img_id not in self.grouped_boxes:
                return None, None, None

            boxes = self.grouped_boxes[img_id]
            classes = [0] * len(boxes)

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img, np.array(boxes, dtype=np.float32), np.array(classes, dtype=np.int32)

        def load_mosaic_dataset(img_id):
            w_target, h_target = self.target_size
            xc = int(random.uniform(w_target // 4, 3 * w_target // 4))
            yc = int(random.uniform(h_target // 4, 3 * h_target // 4))
            indices = [img_id] + random.sample(self.all_image_ids, 3)
            mosaic_img = np.full((h_target, w_target, 3), 128, dtype=np.uint8)
            mosaic_boxes, mosaic_classes = [], []

            for i, current_id in enumerate(indices):
                img, boxes, classes = load_raw_image_and_boxes(current_id)
                if img is None or len(boxes) == 0: continue
                h, w, _ = img.shape

                if i == 0:
                    x1_a, y1_a, x2_a, y2_a = 0, 0, xc, yc
                    x1_b, y1_b, x2_b, y2_b = w - xc, h - yc, w, h
                elif i == 1:
                    x1_a, y1_a, x2_a, y2_a = xc, 0, w_target, yc
                    x1_b, y1_b, x2_b, y2_b = 0, h - yc, w_target - xc, h
                elif i == 2:
                    x1_a, y1_a, x2_a, y2_a = 0, yc, xc, h_target
                    x1_b, y1_b, x2_b, y2_b = w - xc, 0, w, h_target - yc
                elif i == 3:
                    x1_a, y1_a, x2_a, y2_a = xc, yc, w_target, h_target
                    x1_b, y1_b, x2_b, y2_b = 0, 0, w_target - xc, h_target - yc

                x1_b, y1_b, x2_b, y2_b = max(0, x1_b), max(0, y1_b), min(w, x2_b), min(h, y2_b)
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

        def load_letterbox(img_id):
            img, boxes, classes = load_raw_image_and_boxes(img_id)
            if img is None: return None, None, None
            shape = img.shape[:2]
            r = min(self.target_size[0] / shape[0], self.target_size[1] / shape[1])
            new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
            dw, dh = (self.target_size[1] - new_unpad[0]) / 2, (self.target_size[0] - new_unpad[1]) / 2

            if shape[::-1] != new_unpad:
                img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

            top, bottom, left, right = int(round(dh - 0.1)), int(round(dh + 0.1)), int(round(dw - 0.1)), int(
                round(dw + 0.1))
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128))

            transformed_boxes = [
                [(((b[0] + b[2]) / 2) * r + left) / self.target_size[1],
                 (((b[1] + b[3]) / 2) * r + top) / self.target_size[0],
                 ((b[2] - b[0]) * r) / self.target_size[1], ((b[3] - b[1]) * r) / self.target_size[0]] for b in boxes]
            return img.astype(np.float32) / 255.0, transformed_boxes, classes

        def build_targets_11(boxes, classes):
            target_grids = [np.zeros((g, g, 5 + self.num_classes), dtype=np.float32) for g in self.grid_sizes]
            for i in range(len(boxes)):
                if boxes[i][2] == 0 and boxes[i][3] == 0: continue
                cx, cy, w, h = boxes[i]
                c = int(classes[i])
                max_dim = max(w * self.target_size[0], h * self.target_size[1])
                scale_idx = 0 if max_dim < 64 else 1 if max_dim < 128 else 2
                grid_size = self.grid_sizes[scale_idx]
                grid_x, grid_y = int(cx * grid_size), int(cy * grid_size)

                if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                    target_grids[scale_idx][grid_y, grid_x, 0] = 1.0
                    target_grids[scale_idx][grid_y, grid_x, 1:5] = [cx, cy, w, h]
                    target_grids[scale_idx][grid_y, grid_x, 5 + c] = 1.0
            return tuple(target_grids)

        def dataset_generator():
            for i, img_id in enumerate(self.all_image_ids):
                is_val = 0 if i < split_index else 1
                if is_val == 0:
                    if random.random() > 0.3:
                        image, boxes, classes = load_mosaic_dataset(img_id)
                    else:
                        image, boxes, classes = load_letterbox(img_id)
                        if image is not None and random.random() > 0.5:
                            image = np.fliplr(image)
                            for b in boxes: b[0] = 1.0 - b[0]
                else:
                    image, boxes, classes = load_letterbox(img_id)

                if image is None or len(boxes) == 0: continue

                MAX_BOXES = 300
                boxes_padded, classes_padded = np.zeros((MAX_BOXES, 4), dtype=np.float32), np.zeros((MAX_BOXES,),
                                                                                                    dtype=np.float32) - 1.0
                num_boxes = min(len(boxes), MAX_BOXES)
                boxes_padded[:num_boxes], classes_padded[:num_boxes] = np.array(boxes[:num_boxes],
                                                                                dtype=np.float32), np.array(
                    classes[:num_boxes], dtype=np.float32)
                t1, t2, t3 = build_targets_11(boxes, classes)

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
                tf.TensorSpec(shape=(640, 640, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(300, 4), dtype=tf.float32),
                tf.TensorSpec(shape=(300,), dtype=tf.float32),
                tf.TensorSpec(shape=(80, 80, 5 + self.num_classes), dtype=tf.float32),
                tf.TensorSpec(shape=(40, 40, 5 + self.num_classes), dtype=tf.float32),
                tf.TensorSpec(shape=(20, 20, 5 + self.num_classes), dtype=tf.float32),
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

        print(f"[INFO] Dataset successfully loaded. Total samples: {len(self.all_image_ids)}")
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

        dummy_img = tf.zeros((1, 640, 640, 3))
        _ = model(dummy_img, training=True)
        self.optimizer.build(model.trainable_variables)

        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        coco_metric = keras_cv.metrics.BoxCOCOMetrics(bounding_box_format="center_xyWH", evaluate_freq=1)

        @tf.function(jit_compile=False)
        def compute_loss(targets, model_outputs):
            loss = 0.0
            for i, grid_pred in enumerate(model_outputs):
                target_grid, grid_size = targets[i], self.grid_sizes[i]
                box_preds, cls_preds = grid_pred[..., :4 * self.reg_max], grid_pred[..., 4 * self.reg_max:]
                mask, t_boxes, t_cls = target_grid[..., 0], target_grid[..., 1:5], target_grid[..., 5:]

                normalizer = tf.maximum(tf.reduce_sum(mask), 1.0)
                raw_cls_loss = bce(t_cls, cls_preds)
                cls_loss = tf.reduce_sum(raw_cls_loss) / normalizer

                dfl_tensor = tf.reshape(box_preds, [-1, grid_size, grid_size, 4, self.reg_max])
                distances = tf.reduce_sum(tf.nn.softmax(dfl_tensor, axis=-1) * tf.range(self.reg_max, dtype=tf.float32),
                                          axis=-1)
                l, t, r, b = [tf.squeeze(x, -1) for x in tf.split(distances, 4, axis=-1)]

                col, row = tf.range(grid_size, dtype=tf.float32), tf.range(grid_size, dtype=tf.float32)
                grid_x, grid_y = tf.reshape(tf.meshgrid(col, row)[0], [1, grid_size, grid_size]), tf.reshape(
                    tf.meshgrid(col, row)[1], [1, grid_size, grid_size])
                pred_cx, pred_cy = (grid_x + 0.5 + (r - l) / 2.0) / float(grid_size), (
                            grid_y + 0.5 + (b - t) / 2.0) / float(grid_size)
                pred_w, pred_h = (l + r) / float(grid_size), (t + b) / float(grid_size)

                ciou = bbox_ciou(tf.stack([pred_cx, pred_cy], axis=-1), tf.stack([pred_w, pred_h], axis=-1),
                                 t_boxes[..., 0:2], t_boxes[..., 2:4])
                box_loss = tf.reduce_sum((1.0 - ciou) * mask) / normalizer

                loss += 0.5 * cls_loss + 7.5 * box_loss
            return loss

        @tf.function
        def decode_preds(model_outputs):
            all_boxes, all_scores = [], []
            for i, grid_pred in enumerate(model_outputs):
                batch_size, grid_size = tf.shape(grid_pred)[0], self.grid_sizes[i]
                box_preds, cls_preds = grid_pred[..., :4 * self.reg_max], grid_pred[..., 4 * self.reg_max:]

                dfl_tensor = tf.reshape(box_preds, [-1, grid_size, grid_size, 4, self.reg_max])
                distances = tf.reduce_sum(tf.nn.softmax(dfl_tensor, axis=-1) * tf.range(self.reg_max, dtype=tf.float32),
                                          axis=-1)
                l, t, r, b = tf.split(distances, 4, axis=-1)

                grid_x, grid_y = tf.meshgrid(tf.range(grid_size, dtype=tf.float32),
                                             tf.range(grid_size, dtype=tf.float32))
                grid_x, grid_y = tf.reshape(grid_x, [1, grid_size, grid_size, 1]), tf.reshape(grid_y,
                                                                                              [1, grid_size, grid_size,
                                                                                               1])

                pred_y1 = ((grid_y + 0.5 - t) / float(grid_size)) * self.target_size[1]
                pred_x1 = ((grid_x + 0.5 - l) / float(grid_size)) * self.target_size[0]
                pred_y2 = ((grid_y + 0.5 + b) / float(grid_size)) * self.target_size[1]
                pred_x2 = ((grid_x + 0.5 + r) / float(grid_size)) * self.target_size[0]

                boxes = tf.concat([pred_y1, pred_x1, pred_y2, pred_x2], axis=-1)
                all_boxes.append(tf.reshape(boxes, [batch_size, -1, 4]))
                all_scores.append(tf.reshape(tf.math.sigmoid(cls_preds), [batch_size, -1, self.num_classes]))

            final_boxes = tf.concat(all_boxes, axis=1)
            final_scores = tf.concat(all_scores, axis=1)

            nms_out = tf.image.combined_non_max_suppression(
                boxes=tf.expand_dims(final_boxes, axis=2), scores=final_scores,
                max_output_size_per_class=300, max_total_size=300,
                iou_threshold=0.60, score_threshold=0.001, clip_boxes=False
            )
            y1, x1, y2, x2 = tf.split(nms_out.nmsed_boxes, 4, axis=-1)
            nms_boxes_cyxhwh = tf.concat([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=-1)
            mask = tf.sequence_mask(nms_out.valid_detections, maxlen=300)
            final_classes = tf.where(mask, nms_out.nmsed_classes, tf.fill(tf.shape(nms_out.nmsed_classes), -1.0))

            return {"boxes": nms_boxes_cyxhwh, "classes": final_classes, "confidence": nms_out.nmsed_scores}

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

                # Global Wheat target scaling fix for COCO metrics
                coco_metric.update_state({"boxes": val_boxes * [640, 640, 640, 640], "classes": val_classes},
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

        new_model = build_yolo11_model(variant="nano", input_shape=(640, 640, 3))
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