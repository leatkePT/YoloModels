import os
import sys
import time
import random
import numpy as np
import csv
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm

import tensorflow as tf
import keras_cv

# === SERVER OPTIMIZATIONS ===
# os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# tf.config.optimizer.set_jit(False)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from my_models.yolo_model11 import build_yolo11_model


# =======================================================
# SOTA YOLO LOSS & DECODER FUNCTIONS
# =======================================================
def bbox_ciou(b1_xy, b1_wh, b2_xy, b2_wh):
    b1_xmin, b1_ymin = b1_xy[..., 0] - b1_wh[..., 0] / 2.0, b1_xy[..., 1] - b1_wh[..., 1] / 2.0
    b1_xmax, b1_ymax = b1_xy[..., 0] + b1_wh[..., 0] / 2.0, b1_xy[..., 1] + b1_wh[..., 1] / 2.0
    b2_xmin, b2_ymin = b2_xy[..., 0] - b2_wh[..., 0] / 2.0, b2_xy[..., 1] - b2_wh[..., 1] / 2.0
    b2_xmax, b2_ymax = b2_xy[..., 0] + b2_wh[..., 0] / 2.0, b2_xy[..., 1] + b2_wh[..., 1] / 2.0
    inter_area = tf.maximum(tf.minimum(b1_xmax, b2_xmax) - tf.maximum(b1_xmin, b2_xmin), 0.0) * tf.maximum(
        tf.minimum(b1_ymax, b2_ymax) - tf.maximum(b1_ymin, b2_ymin), 0.0)
    iou = inter_area / tf.maximum((b1_wh[..., 0] * b1_wh[..., 1]) + (b2_wh[..., 0] * b2_wh[..., 1]) - inter_area, 1e-7)
    c_squared = tf.square(tf.maximum(b1_xmax, b2_xmax) - tf.minimum(b1_xmin, b2_xmin)) + tf.square(
        tf.maximum(b1_ymax, b2_ymax) - tf.minimum(b1_ymin, b2_ymin))
    center_dist_squared = tf.square(b1_xy[..., 0] - b2_xy[..., 0]) + tf.square(b1_xy[..., 1] - b2_xy[..., 1])
    factor = (4.0 / (np.pi ** 2))
    v = factor * tf.square(tf.math.atan(b1_wh[..., 0] / tf.maximum(b1_wh[..., 1], 1e-7)) - tf.math.atan(
        b2_wh[..., 0] / tf.maximum(b2_wh[..., 1], 1e-7)))
    alpha = v / tf.maximum((1.0 - iou) + v, 1e-7)
    return iou - (center_dist_squared / tf.maximum(c_squared, 1e-7)) - alpha * v


# =======================================================
# Η ΚΥΡΙΑ ΚΛΑΣΗ: YOLO11_OxfordPets
# =======================================================
class YOLO11_OxfordPets:

    def __init__(self):
        self.name = 'YOLO11_Nano_OxfordPets'

        # Υπερπαράμετροι ΜΟΝΟ για τη φάση ανάρρωσης (Pruning Fine-Tune)
        self.fine_tune_epochs = 5  # Λίγες εποχές, ίσα για να αναρρώσει το Pruned μοντέλο
        self.batch_size = 16

        # Πολύ μικρό Learning Rate για να μην καταστρέψει τη γνώση που του απέμεινε
        self.initial_learning_rate = 1e-4
        self.loss_function = 'YOLO_SOTA_Loss (CIoU + DFL + BCE)'

        # Optimizer προσαρμοσμένος για σταθερό και απαλό fine-tuning
        self.optimizer = tf.keras.optimizers.SGD(
            learning_rate=self.initial_learning_rate,
            momentum=0.9,
            weight_decay=0.0005
        )

        self.metrics = ['mAP_50', 'mAP_50_95']
        self.best_model_metrics = {'monitor': 'mAP_50', 'mode': 'max'}
        self.header = ['train_loss', 'val_loss', 'mAP_50', 'mAP_50_95']

        self.target_size = (640, 640)
        self.grid_sizes = [80, 40, 20]
        self.reg_max = 16
        self.classes = ["dog", "cat"]
        self.num_classes = len(self.classes)
        self.class_to_id = {name.lower(): i for i, name in enumerate(self.classes)}

        self.image_dir = "./datasets/oxford_pets/images"
        self.annot_path = "./datasets/oxford_pets/annotations/xmls"

        self.log_dir = "logs/oop_experiments/oxford"
        os.makedirs(self.log_dir, exist_ok=True)

        self.model_architecture = None
        self.model_history = None
        self.train_dataset = None
        self.val_dataset = None
        self.all_image_files = []

        print(f'\n[INIT] Αρχικοποίηση {self.name} (Service Class for Pruning)')
        print(f'   - Fine Tune Εποχές: {self.fine_tune_epochs}')
        print(f'   - Batch Size: {self.batch_size}')
        print(f'   - Learning Rate για Ανάρρωση: {self.initial_learning_rate}')

    def build(self, print_summary=True):
        print("[BUILD] Χτίσιμο αρχιτεκτονικής YOLO11-Nano...")
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
        print("[DATA] Εκκίνηση προετοιμασίας δεδομένων (Mosaic, Letterbox)...")

        if os.path.exists(self.image_dir):
            self.all_image_files = [f for f in os.listdir(self.image_dir) if f.lower().endswith('.jpg')]

        if not self.all_image_files:
            print(f"❌ ΣΦΑΛΜΑ: Δεν βρέθηκαν εικόνες στο {self.image_dir}")
            sys.exit(1)

        split_index = int(len(self.all_image_files) * 0.8)

        def load_raw(img_file):
            img_path = os.path.join(self.image_dir, img_file)
            xml_name = img_file.rsplit('.', 1)[0] + '.xml'
            xml_path = os.path.join(self.annot_path, xml_name)
            if not os.path.exists(xml_path): return None, None, None
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                boxes, classes = [], []
                for obj in root.findall('object'):
                    name = obj.find('name').text.lower().strip()
                    if name not in self.class_to_id: continue
                    boxes.append([float(obj.find('bndbox/xmin').text), float(obj.find('bndbox/ymin').text),
                                  float(obj.find('bndbox/xmax').text), float(obj.find('bndbox/ymax').text)])
                    classes.append(self.class_to_id[name])
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img, np.array(boxes, dtype=np.float32), np.array(classes, dtype=np.int32)
            except:
                return None, None, None

        def load_mosaic(img_file):
            w_tgt, h_tgt = self.target_size
            xc, yc = int(random.uniform(w_tgt // 4, 3 * w_tgt // 4)), int(random.uniform(h_tgt // 4, 3 * h_tgt // 4))
            indices = [img_file] + random.sample(self.all_image_files, 3)
            mosaic_img = np.full((h_tgt, w_tgt, 3), 128, dtype=np.uint8)
            mosaic_boxes, mosaic_classes = [], []
            for i, file in enumerate(indices):
                img, boxes, classes = load_raw(file)
                if img is None or len(boxes) == 0: continue
                h, w, _ = img.shape
                if i == 0:
                    x1a, y1a, x2a, y2a = 0, 0, xc, yc
                    x1b, y1b, x2b, y2b = w - xc, h - yc, w, h
                elif i == 1:
                    x1a, y1a, x2a, y2a = xc, 0, w_tgt, yc
                    x1b, y1b, x2b, y2b = 0, h - yc, w_tgt - xc, h
                elif i == 2:
                    x1a, y1a, x2a, y2a = 0, yc, xc, h_tgt
                    x1b, y1b, x2b, y2b = w - xc, 0, w, h_tgt - yc
                elif i == 3:
                    x1a, y1a, x2a, y2a = xc, yc, w_tgt, h_tgt
                    x1b, y1b, x2b, y2b = 0, 0, w_tgt - xc, h_tgt - yc
                x1b, y1b = max(0, x1b), max(0, y1b)
                x2b, y2b = min(w, x2b), min(h, y2b)
                pad_w, pad_h = (x2a - x1a), (y2a - y1a)
                if x2b - x1b > pad_w: x2b = x1b + pad_w
                if y2b - y1b > pad_h: y2b = y1b + pad_h
                mosaic_img[y1a:y1a + (y2b - y1b), x1a:x1a + (x2b - x1b)] = img[y1b:y2b, x1b:x2b]
                pad_x, pad_y = x1a - x1b, y1a - y1b
                for b_idx, box in enumerate(boxes):
                    xmin, ymin, xmax, ymax = box[0] + pad_x, box[1] + pad_y, box[2] + pad_x, box[3] + pad_y
                    xmin, ymin = max(x1a, min(xmin, x2a)), max(y1a, min(ymin, y2a))
                    xmax, ymax = max(x1a, min(xmax, x2a)), max(y1a, min(ymax, y2a))
                    if (xmax - xmin) > 5 and (ymax - ymin) > 5:
                        mosaic_boxes.append(
                            [(xmin + xmax) / 2 / w_tgt, (ymin + ymax) / 2 / h_tgt, (xmax - xmin) / w_tgt,
                             (ymax - ymin) / h_tgt])
                        mosaic_classes.append(classes[b_idx])
            return mosaic_img.astype(np.float32) / 255.0, mosaic_boxes, mosaic_classes

        def load_letterbox(img_file):
            img, boxes, classes = load_raw(img_file)
            if img is None: return None, None, None
            shape = img.shape[:2]
            r = min(self.target_size[0] / shape[0], self.target_size[1] / shape[1])
            new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
            dw, dh = (self.target_size[1] - new_unpad[0]) / 2, (self.target_size[0] - new_unpad[1]) / 2
            if shape[::-1] != new_unpad: img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128))
            tr_boxes = [[(((b[0] + b[2]) / 2) * r + left) / self.target_size[1],
                         (((b[1] + b[3]) / 2) * r + top) / self.target_size[0],
                         ((b[2] - b[0]) * r) / self.target_size[1], ((b[3] - b[1]) * r) / self.target_size[0]] for b in
                        boxes]
            return img.astype(np.float32) / 255.0, tr_boxes, classes

        def build_targs(boxes, classes):
            target_grids = [np.zeros((g, g, 5 + self.num_classes), dtype=np.float32) for g in self.grid_sizes]
            for i in range(len(boxes)):
                if boxes[i][2] == 0 and boxes[i][3] == 0: continue
                cx, cy, w, h = boxes[i]
                c = int(classes[i])
                max_dim = max(w * self.target_size[0], h * self.target_size[1])
                scale_idx = 0 if max_dim < 64 else 1 if max_dim < 128 else 2
                grid_size = self.grid_sizes[scale_idx]
                gx, gy = int(cx * grid_size), int(cy * grid_size)
                if 0 <= gx < grid_size and 0 <= gy < grid_size:
                    target_grids[scale_idx][gy, gx, 0] = 1.0
                    target_grids[scale_idx][gy, gx, 1:5] = [cx, cy, w, h]
                    target_grids[scale_idx][gy, gx, 5 + c] = 1.0
            return tuple(target_grids)

        def data_gen():
            for i, img_file in enumerate(self.all_image_files):
                is_val = 0 if i < split_index else 1
                if is_val == 0:
                    if random.random() > 0.3:
                        image, boxes, classes = load_mosaic(img_file)
                    else:
                        image, boxes, classes = load_letterbox(img_file)
                        if image is not None and random.random() > 0.5:
                            image = np.fliplr(image)
                            for b in boxes: b[0] = 1.0 - b[0]
                else:
                    image, boxes, classes = load_letterbox(img_file)

                if image is None or len(boxes) == 0: continue
                boxes_padded, classes_padded = np.zeros((100, 4), dtype=np.float32), np.zeros((100,),
                                                                                              dtype=np.float32) - 1.0
                num_boxes = min(len(boxes), 100)
                boxes_padded[:num_boxes], classes_padded[:num_boxes] = np.array(boxes[:num_boxes],
                                                                                dtype=np.float32), np.array(
                    classes[:num_boxes], dtype=np.float32)
                t1, t2, t3 = build_targs(boxes, classes)
                yield image, boxes_padded, classes_padded, t1, t2, t3, is_val

        @tf.function
        def augment_colors(image, t1, t2, t3):
            image = tf.image.random_brightness(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
            image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
            return tf.clip_by_value(image, 0.0, 1.0), t1, t2, t3

        dataset = tf.data.Dataset.from_generator(
            data_gen,
            output_signature=(
                tf.TensorSpec(shape=(640, 640, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(100, 4), dtype=tf.float32),
                tf.TensorSpec(shape=(100,), dtype=tf.float32),
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

        print(f"✅ Δεδομένα έτοιμα. Σύνολο εικόνων: {len(self.all_image_files)}")
        return self.train_dataset, self.val_dataset

    # =========================================================================
    # 1. TRAIN: Κενή (αφού παραδίδεις ήδη εκπαιδευμένο μοντέλο)
    # =========================================================================
    def train(self, fit_flag=True, print_summary=False):
        print("▶️ [TRAIN] Η μέθοδος train παρακάμπτεται επειδή χρησιμοποιούμε ήδη εκπαιδευμένο μοντέλο.")
        print("   (Σύμφωνα με τον κανόνα: 'if we have a pretrained model this method is not necessary')")
        return None

    # =========================================================================
    # 2. FINE_TUNE: Η Καρδιά του Pruning Experiment του Καθηγητή
    # =========================================================================
    def fine_tune(self, path_to_save, new_model=None, print_summary=False, new_fine_tune_epochs=None):
        print("\n" + "=" * 50)
        print("[FINE TUNE] Εκκίνηση Φάσης Ανάρρωσης (Pruning Fine-Tune)...")

        # ΕΔΩ ΕΙΝΑΙ ΟΛΗ Η ΛΟΓΙΚΗ!
        if new_model is None:
            print(
                "❌ ΣΦΑΛΜΑ: Για να τρέξει η fine_tune σε αυτό το API, πρέπει να δοθεί ένα μοντέλο (π.χ. το Pruned) στο όρισμα 'new_model'.")
            return None, None

        print("🔄 Φορτώθηκε το εξωτερικό (Pruned) μοντέλο επιτυχώς.")
        model = new_model

        fine_tune_epochs = new_fine_tune_epochs if new_fine_tune_epochs else self.fine_tune_epochs

        if not self.train_dataset:
            self.data_preprocessing()

        # Ορίζουμε τον SOTA Optimizer ΜΟΝΟ για την ανάρρωση
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
                max_output_size_per_class=100, max_total_size=100,
                iou_threshold=0.60, score_threshold=0.001, clip_boxes=False
            )
            y1, x1, y2, x2 = tf.split(nms_out.nmsed_boxes, 4, axis=-1)
            nms_boxes_cyxhwh = tf.concat([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=-1)
            mask = tf.sequence_mask(nms_out.valid_detections, maxlen=100)
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

            print(f"📊 Epoch {epoch + 1} -> Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | mAP@0.50: {map_50:.4f}")

            if epoch == 0 or map_50 >= max(history_dict['mAP_50'][:-1]):
                model.save_weights(path_to_save)

        best_info = self.get_best_epoch_info(history_dict)
        print(f"\n✅ Ανάρρωση Ολοκληρώθηκε! Καλύτερα Metrics: {best_info}")

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
            raise ValueError('mode should be either min or max')

        new_history = {key: hist_dict[key][best_epoch_ix] for key in self.header if key in hist_dict}
        if metrics: new_history = dict(new_history, **dict(metrics))
        new_history['best_epoch'] = best_epoch_ix + 1
        return new_history

    def extract_monitoring_metric_value(self, model=None):
        if not self.model_history:
            print("⚠️ Δεν υπάρχει history! Βεβαιωθείτε ότι έτρεξε η fine_tune.")
            return 0.0
        best_info = self.get_best_epoch_info(self.model_history)
        monitor_metric = self.best_model_metrics['monitor']
        if monitor_metric in best_info:
            metric_val = best_info[monitor_metric]
            print(f"[METRIC] Το {monitor_metric} του εκπαιδευμένου μοντέλου είναι: {metric_val:.4f}")
            return metric_val
        else:
            print(f"❌ ΣΦΑΛΜΑ: Η μετρική '{monitor_metric}' δεν βρέθηκε στο history!")
            return 0.0