import os
import sys
import time
import random
import numpy as np
import csv
import xml.etree.ElementTree as ET
import cv2
from tqdm import tqdm

# === OPTIMIZATIONS ΓΙΑ SERVER (AUTH HPC A100 / V100) ===
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

tf.config.optimizer.set_jit(False)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
# =======================================================

import keras_cv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from my_models.yolo_model11 import build_yolo11_model

# =======================================================
# 1. DATASET & TRANSFER LEARNING EXPERIMENT SETTINGS
# =======================================================
TARGET_SIZE = (640, 640)
BATCH_SIZE = 16
TOTAL_EPOCHS = 100

# Ιδανικό Learning Rate για SGD βάσει YOLO Paper
MAX_LEARNING_RATE = 0.01

DATASET_NAME = "Oxford_11n_Pretrained"
IMAGE_DIR = "./datasets/oxford_pets/images"
ANNOT_PATH = "./datasets/oxford_pets/annotations/xmls"
LOG_DIR = "./logs/yolo11n_logs"

CLASSES = ["dog", "cat"]

NUM_CLASSES = len(CLASSES)
class_to_id = {name.lower(): i for i, name in enumerate(CLASSES)}
REG_MAX = 16

os.makedirs(LOG_DIR, exist_ok=True)
GRID_SIZES = [80, 40, 20]

all_image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith('.jpg')] if os.path.exists(IMAGE_DIR) else []
if len(all_image_files) == 0:
    print(f"\n❌ ΣΦΑΛΜΑ: Δεν βρέθηκαν εικόνες στο {IMAGE_DIR}!")
    sys.exit(1)
print(f"\n✅ Βρέθηκαν {len(all_image_files)} εικόνες για εκπαίδευση στο Oxford Pets.")


# =======================================================
# 2. DATA LOADING & SOTA PREPROCESSING PIPELINE
# =======================================================
def load_raw_image_and_boxes(img_file):
    img_path = os.path.join(IMAGE_DIR, img_file)
    xml_name = img_file.rsplit('.', 1)[0] + '.xml'
    xml_path = os.path.join(ANNOT_PATH, xml_name)

    if not os.path.exists(xml_path): return None, None, None

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes, classes = [], []
        for obj in root.findall('object'):
            name = obj.find('name').text.lower().strip()
            if name not in class_to_id: continue
            current_class = class_to_id[name]
            bndbox = obj.find('bndbox')
            boxes.append(
                [float(bndbox.find('xmin').text), float(bndbox.find('ymin').text), float(bndbox.find('xmax').text),
                 float(bndbox.find('ymax').text)])
            classes.append(current_class)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, np.array(boxes, dtype=np.float32), np.array(classes, dtype=np.int32)
    except Exception:
        return None, None, None


def load_mosaic_dataset(img_file):
    w_target, h_target = TARGET_SIZE
    xc = int(random.uniform(w_target // 4, 3 * w_target // 4))
    yc = int(random.uniform(h_target // 4, 3 * h_target // 4))
    indices = [img_file] + random.sample(all_image_files, 3)
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

        x1_b, y1_b, x2_b, y2_b = max(0, x1_b), max(0, y1_b), min(w, x2_b), min(h, y2_b)
        pad_w, pad_h = (x2_a - x1_a), (y2_a - y1_a)
        if x2_b - x1_b > pad_w: x2_b = x1_b + pad_w
        if y2_b - y1_b > pad_h: y2_b = y1_b + pad_h
        mosaic_img[y1_a:y1_a + (y2_b - y1_b), x1_a:x1_a + (x2_b - x1_b)] = img[y1_b:y2_b, x1_b:x2_b]
        pad_x, pad_y = x1_a - x1_b, y1_a - y1_b

        for b_idx, box in enumerate(boxes):
            xmin, ymin, xmax, ymax = max(x1_a, min(box[0] + pad_x, x2_a)), max(y1_a, min(box[1] + pad_y, y2_a)), max(
                x1_a, min(box[2] + pad_x, x2_a)), max(y1_a, min(box[3] + pad_y, y2_a))
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
    r = min(TARGET_SIZE[0] / shape[0], TARGET_SIZE[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = (TARGET_SIZE[1] - new_unpad[0]) / 2, (TARGET_SIZE[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad: img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom, left, right = int(round(dh - 0.1)), int(round(dh + 0.1)), int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128))

    transformed_boxes = [
        [(((b[0] + b[2]) / 2) * r + left) / TARGET_SIZE[1], (((b[1] + b[3]) / 2) * r + top) / TARGET_SIZE[0],
         ((b[2] - b[0]) * r) / TARGET_SIZE[1], ((b[3] - b[1]) * r) / TARGET_SIZE[0]] for b in boxes]
    return img.astype(np.float32) / 255.0, transformed_boxes, classes


def build_targets_11(boxes, classes):
    target_grids = [np.zeros((g, g, 5 + NUM_CLASSES), dtype=np.float32) for g in GRID_SIZES]
    for i in range(len(boxes)):
        if boxes[i][2] == 0 and boxes[i][3] == 0: continue
        cx, cy, w, h = boxes[i]
        c = int(classes[i])
        max_dim = max(w * TARGET_SIZE[0], h * TARGET_SIZE[1])
        scale_idx = 0 if max_dim < 64 else 1 if max_dim < 128 else 2
        grid_size = GRID_SIZES[scale_idx]
        grid_x, grid_y = int(cx * grid_size), int(cy * grid_size)
        if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
            target_grids[scale_idx][grid_y, grid_x, 0] = 1.0
            target_grids[scale_idx][grid_y, grid_x, 1:5] = [cx, cy, w, h]
            target_grids[scale_idx][grid_y, grid_x, 5 + c] = 1.0
    return tuple(target_grids)


def dataset_generator():
    split_index = int(len(all_image_files) * 0.8)
    for i, img_file in enumerate(all_image_files):
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
        boxes_padded, classes_padded = np.zeros((100, 4), dtype=np.float32), np.zeros((100,), dtype=np.float32) - 1.0
        num_boxes = min(len(boxes), 100)
        boxes_padded[:num_boxes], classes_padded[:num_boxes] = np.array(boxes[:num_boxes], dtype=np.float32), np.array(
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
        tf.TensorSpec(shape=(100, 4), dtype=tf.float32),
        tf.TensorSpec(shape=(100,), dtype=tf.float32),
        tf.TensorSpec(shape=(GRID_SIZES[0], GRID_SIZES[0], 5 + NUM_CLASSES), dtype=tf.float32),
        tf.TensorSpec(shape=(GRID_SIZES[1], GRID_SIZES[1], 5 + NUM_CLASSES), dtype=tf.float32),
        tf.TensorSpec(shape=(GRID_SIZES[2], GRID_SIZES[2], 5 + NUM_CLASSES), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
)
train_dataset = dataset.filter(lambda img, box, cls, t1, t2, t3, is_val: is_val == 0).map(
    lambda img, box, cls, t1, t2, t3, is_val: (img, t1, t2, t3)).map(augment_colors,
                                                                     num_parallel_calls=tf.data.AUTOTUNE).batch(
    BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
val_dataset = dataset.filter(lambda img, box, cls, t1, t2, t3, is_val: is_val == 1).map(
    lambda img, box, cls, t1, t2, t3, is_val: (img, box, cls, t1, t2, t3)).batch(BATCH_SIZE,
                                                                                 drop_remainder=True).prefetch(
    tf.data.AUTOTUNE)


# =======================================================
# 3. LOSS & DECODER FUNCTIONS (SOTA ULTRALYTICS LOGIC)
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


@tf.function(jit_compile=False)
def compute_yolo11_loss(targets, model_outputs):
    loss = 0.0
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    for i, grid_pred in enumerate(model_outputs):
        target_grid, grid_size = targets[i], GRID_SIZES[i]
        box_preds, cls_preds = grid_pred[..., :4 * REG_MAX], grid_pred[..., 4 * REG_MAX:]
        mask, t_boxes, t_cls = target_grid[..., 0], target_grid[..., 1:5], target_grid[..., 5:]

        # === SOTA NORMALIZER (Αποτρέπει τη διαίρεση με το 0 σε άδεια Val batches!) ===
        normalizer = tf.maximum(tf.reduce_sum(mask), 1.0)

        # 1. Class Loss: ΧΩΡΙΣ MASK! Το μοντέλο ΠΡΕΠΕΙ να μαθαίνει το background!
        raw_cls_loss = bce(t_cls, cls_preds)
        cls_loss = tf.reduce_sum(raw_cls_loss) / normalizer

        # 2. Box & DFL Losses: Με MASK (υπολογίζονται ΜΟΝΟ εκεί που υπάρχουν πραγματικά αντικείμενα)
        dfl_tensor = tf.reshape(box_preds, [-1, grid_size, grid_size, 4, REG_MAX])
        distances = tf.reduce_sum(tf.nn.softmax(dfl_tensor, axis=-1) * tf.range(REG_MAX, dtype=tf.float32), axis=-1)
        l, t, r, b = [tf.squeeze(x, -1) for x in tf.split(distances, 4, axis=-1)]

        col, row = tf.range(grid_size, dtype=tf.float32), tf.range(grid_size, dtype=tf.float32)
        grid_x, grid_y = tf.reshape(tf.meshgrid(col, row)[0], [1, grid_size, grid_size]), tf.reshape(
            tf.meshgrid(col, row)[1], [1, grid_size, grid_size])
        pred_cx, pred_cy = (grid_x + 0.5 + (r - l) / 2.0) / float(grid_size), (grid_y + 0.5 + (b - t) / 2.0) / float(
            grid_size)
        pred_w, pred_h = (l + r) / float(grid_size), (t + b) / float(grid_size)

        ciou = bbox_ciou(tf.stack([pred_cx, pred_cy], axis=-1), tf.stack([pred_w, pred_h], axis=-1), t_boxes[..., 0:2],
                         t_boxes[..., 2:4])
        box_loss = tf.reduce_sum((1.0 - ciou) * mask) / normalizer

        loss += 0.5 * cls_loss + 7.5 * box_loss
    return loss


@tf.function
def decode_yolo11_outputs_sota(model_outputs):
    all_boxes, all_scores = [], []
    for i, grid_pred in enumerate(model_outputs):
        batch_size, grid_size = tf.shape(grid_pred)[0], GRID_SIZES[i]
        box_preds, cls_preds = grid_pred[..., :4 * REG_MAX], grid_pred[..., 4 * REG_MAX:]

        dfl_tensor = tf.reshape(box_preds, [-1, grid_size, grid_size, 4, REG_MAX])
        distances = tf.reduce_sum(tf.nn.softmax(dfl_tensor, axis=-1) * tf.range(REG_MAX, dtype=tf.float32), axis=-1)
        l, t, r, b = tf.split(distances, 4, axis=-1)

        grid_x, grid_y = tf.meshgrid(tf.range(grid_size, dtype=tf.float32), tf.range(grid_size, dtype=tf.float32))
        grid_x, grid_y = tf.reshape(grid_x, [1, grid_size, grid_size, 1]), tf.reshape(grid_y,
                                                                                      [1, grid_size, grid_size, 1])

        pred_y1 = ((grid_y + 0.5 - t) / float(grid_size)) * TARGET_SIZE[1]
        pred_x1 = ((grid_x + 0.5 - l) / float(grid_size)) * TARGET_SIZE[0]
        pred_y2 = ((grid_y + 0.5 + b) / float(grid_size)) * TARGET_SIZE[1]
        pred_x2 = ((grid_x + 0.5 + r) / float(grid_size)) * TARGET_SIZE[0]

        boxes = tf.concat([pred_y1, pred_x1, pred_y2, pred_x2], axis=-1)
        all_boxes.append(tf.reshape(boxes, [batch_size, -1, 4]))
        all_scores.append(tf.reshape(tf.math.sigmoid(cls_preds), [batch_size, -1, NUM_CLASSES]))

    final_boxes_yxyx = tf.concat(all_boxes, axis=1)
    final_scores = tf.concat(all_scores, axis=1)

    boxes_exp = tf.expand_dims(final_boxes_yxyx, axis=2)
    nms_out = tf.image.combined_non_max_suppression(
        boxes=boxes_exp, scores=final_scores,
        max_output_size_per_class=100, max_total_size=100,
        iou_threshold=0.60, score_threshold=0.001, clip_boxes=False
    )

    y1, x1, y2, x2 = tf.split(nms_out.nmsed_boxes, 4, axis=-1)
    nms_boxes_cyxhwh = tf.concat([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=-1)

    mask = tf.sequence_mask(nms_out.valid_detections, maxlen=100)
    final_classes_masked = tf.where(mask, nms_out.nmsed_classes, tf.fill(tf.shape(nms_out.nmsed_classes), -1.0))

    return {
        "boxes": nms_boxes_cyxhwh,
        "classes": final_classes_masked,
        "confidence": nms_out.nmsed_scores
    }


class CosineDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, max_lr, warmup_epochs, total_epochs, steps_per_epoch):
        super().__init__()
        self.max_lr, self.warmup_steps, self.total_steps = tf.cast(max_lr, tf.float32), tf.cast(
            warmup_epochs * steps_per_epoch, tf.float32), tf.cast(total_epochs * steps_per_epoch, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.max_lr * (step / tf.maximum(self.warmup_steps, 1.0))
        cosine_decay = 0.5 * (1.0 + tf.math.cos(3.14159265359 * tf.clip_by_value(
            (step - self.warmup_steps) / tf.maximum(self.total_steps - self.warmup_steps, 1.0), 0.0, 1.0)))
        return tf.where(step < self.warmup_steps, warmup_lr, self.max_lr * cosine_decay + 1e-6)


# =======================================================
# 4. UNIFIED TRAINING LOOP WITH AUTO-RESUME
# =======================================================
def train_yolo():
    log_file = os.path.join(LOG_DIR, f"training_metrics_{DATASET_NAME}.csv")
    LATEST_CHECKPOINT_PATH = f"{LOG_DIR}/oxford11_pretrain_latest_checkpoint.weights.h5"
    PRETRAINED_WEIGHTS_PATH = f"{LOG_DIR}/yolo11n_coco_pretrained.weights.h5"

    # --- 1. ΕΛΕΓΧΟΣ ΓΙΑ RESUME ΑΠΟ ΤΟ CSV ---
    start_epoch = 0
    if os.path.exists(log_file):
        try:
            with open(log_file, mode='r') as f:
                lines = list(csv.reader(f))
                if len(lines) > 1:
                    start_epoch = int(lines[-1][0])
                    print(f"\n🔄 ΑΝΙΧΝΕΥΤΗΚΕ ΔΙΑΚΟΠΗ: Συνέχιση από την Εποχή {start_epoch + 1}")
        except Exception as e:
            print(f"⚠️ Αδυναμία ανάγνωσης log αρχείου ({e}). Ξεκινάει από την αρχή.")

    if start_epoch == 0:
        with open(log_file, mode='w', newline='') as f:
            csv.writer(f).writerow(["Epoch", "Train_Loss", "Val_Loss", "mAP_50", "mAP_50_95", "Epoch_Time_s"])

    import my_models.yolo_model11 as yolo11_module
    yolo11_module.MODEL_CONFIGS["nano"]["nc"] = NUM_CLASSES
    model = build_yolo11_model(variant="nano", input_shape=(640, 640, 3))

    # --- 2. ΦΟΡΤΩΣΗ ΒΑΡΩΝ ---
    if start_epoch > 0 and os.path.exists(LATEST_CHECKPOINT_PATH):
        print(f"🔄 Φόρτωση τελευταίου Checkpoint για Resume: {LATEST_CHECKPOINT_PATH}")
        model.load_weights(LATEST_CHECKPOINT_PATH)
    else:
        if os.path.exists(PRETRAINED_WEIGHTS_PATH):
            print(f"\n🎬 ΕΝΑΡΞΗ ΝΕΑΣ ΕΚΠΑΙΔΕΥΣΗΣ. Φόρτωση COCO weights: {PRETRAINED_WEIGHTS_PATH}")
            model.load_weights(PRETRAINED_WEIGHTS_PATH, by_name=True, skip_mismatch=True)
            start_epoch = 0
        else:
            print(f"\n❌ ΣΦΑΛΜΑ: Δεν βρέθηκαν τα βάρη του COCO στο {PRETRAINED_WEIGHTS_PATH}!")
            sys.exit(1)

    # --- 3. ΡΥΘΜΙΣΗ STEPS & SGD OPTIMIZER ---
    num_train_images = int(len(all_image_files) * 0.8)
    steps_per_epoch = max(1, num_train_images // BATCH_SIZE)

    lr_schedule = CosineDecayWithWarmup(max_lr=MAX_LEARNING_RATE, warmup_epochs=5, total_epochs=TOTAL_EPOCHS,
                                        steps_per_epoch=steps_per_epoch)

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=lr_schedule,
        momentum=0.937,
        weight_decay=0.0005,
        use_ema=True,
        ema_momentum=0.999
    )

    # === ΔΙΚΛΕΙΔΑ ΑΣΦΑΛΕΙΑΣ ΓΙΑ ΤΟ RESUME ===
    dummy_img = tf.zeros((1, 640, 640, 3))
    _ = model(dummy_img, training=True)
    optimizer.build(model.trainable_variables)
    # ========================================

    initial_step = start_epoch * steps_per_epoch
    optimizer.iterations.assign(initial_step)
    print(f"ℹ️ Optimizer Advanced to Iteration: {initial_step} (Epoch {start_epoch})")

    coco_metric = keras_cv.metrics.BoxCOCOMetrics(bounding_box_format="center_xyWH", evaluate_freq=1)

    @tf.function(jit_compile=False)
    def train_step(images, targets):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = compute_yolo11_loss(targets, predictions)
        optimizer.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
        return loss

    # --- 4. ΚΥΡΙΟΣ ΒΡΟΧΟΣ ΕΚΠΑΙΔΕΥΣΗΣ ---
    for epoch in range(start_epoch, TOTAL_EPOCHS):
        epoch_start_time = time.time()
        train_loss_total, num_train_batches = 0.0, 0
        pbar_train = tqdm(train_dataset, desc=f"Epoch {epoch + 1}/{TOTAL_EPOCHS} [Train]", leave=True)

        for batch_images, t1, t2, t3 in pbar_train:
            loss = train_step(batch_images, [t1, t2, t3])
            train_loss_total += float(loss)
            num_train_batches += 1
            pbar_train.set_postfix(Loss=f"{train_loss_total / num_train_batches:.4f}",
                                   LR=f"{float(lr_schedule(optimizer.iterations)):.2e}")

        val_loss_total, num_val_batches = 0.0, 0
        coco_metric.reset_state()
        pbar_val = tqdm(val_dataset, desc=f"Epoch {epoch + 1}/{TOTAL_EPOCHS} [Eval]", leave=False)

        for val_images, val_boxes, val_classes, t1, t2, t3 in pbar_val:
            val_predictions = model(val_images, training=False)
            val_loss_total += float(compute_yolo11_loss([t1, t2, t3], val_predictions))
            num_val_batches += 1

            y_pred_dict = decode_yolo11_outputs_sota(val_predictions)
            coco_metric.update_state(
                {"boxes": val_boxes * [TARGET_SIZE[0], TARGET_SIZE[1], TARGET_SIZE[0], TARGET_SIZE[1]],
                 "classes": val_classes},
                y_pred_dict
            )

        metric_results = coco_metric.result()
        map_50 = float(metric_results['MaP@[IoU=50]'])
        map_50_95 = float(metric_results['MaP'])
        epoch_time = time.time() - epoch_start_time

        print(
            f"📊 Εποχή {epoch + 1} -> Train Loss: {train_loss_total / max(1, num_train_batches):.4f} | Val Loss: {val_loss_total / max(1, num_val_batches):.4f} | mAP@0.50: {map_50:.4f} | mAP@0.50:0.95: {map_50_95:.4f} | Χρόνος: {epoch_time:.1f}s")

        with open(log_file, mode='a', newline='') as f:
            csv.writer(f).writerow([epoch + 1, round(train_loss_total / max(1, num_train_batches), 4),
                                    round(val_loss_total / max(1, num_val_batches), 4), round(map_50, 4),
                                    round(map_50_95, 4), round(epoch_time, 2)])

        model.save_weights(LATEST_CHECKPOINT_PATH)

    model.save_weights(f"{LOG_DIR}/yolov11_nano_{DATASET_NAME}_final.weights.h5")
    print(f"\n🎉 Η εκπαίδευση ολοκληρώθηκε επιτυχώς!")


if __name__ == "__main__":
    train_yolo()