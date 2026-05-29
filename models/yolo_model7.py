import tensorflow as tf
from tensorflow.keras import layers, Model
import json
import os


# ==========================================
# 1. LOAD EXTERNAL CONFIGURATION
# Τώρα διαβάζει τα variants από το εξωτερικό JSON με απόλυτο path
# ==========================================
def load_config(config_filename="yolo7_config.json"):
    # Βρίσκει τον φάκελο στον οποίο βρίσκεται αυτό ακριβώς το script (δηλαδή τον φάκελο models)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Φτιάχνει το πλήρες path ενώνοντας τον φάκελο με το όνομα του json
    config_path = os.path.join(current_dir, config_filename)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"⚠️ Το αρχείο {config_path} δεν βρέθηκε! Βεβαιώσου ότι είναι στον ίδιο φάκελο.")

    with open(config_path, "r") as f:
        return json.load(f)


# Φόρτωση των ρυθμίσεων
MODEL_CONFIGS = load_config()


# ==========================================
# 2. BUILDING BLOCKS
# ==========================================
def CBS(x, filters, kernel_size, strides=1, name=None):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False,
                      name=name + '_conv' if name else None)(x)
    x = layers.BatchNormalization(name=name + '_bn' if name else None)(x)
    return layers.LeakyReLU(alpha=0.1, name=name + '_lrelu' if name else None)(x)


def ELAN_Block(x, out_filters, name=None):
    mid = out_filters // 2
    b1 = CBS(x, mid, 1)
    b2 = CBS(x, mid, 1)
    b3 = CBS(b2, mid, 3)
    b4 = CBS(b3, mid, 3)
    merged = layers.Concatenate(-1)([b1, b2, b3, b4])
    return CBS(merged, out_filters, 1, name=name)


def SPP_Block(x, filters, name="spp"):
    c1 = CBS(x, filters // 2, 1)
    p1 = layers.MaxPooling2D(5, 1, 'same')(c1)
    p2 = layers.MaxPooling2D(9, 1, 'same')(c1)
    p3 = layers.MaxPooling2D(13, 1, 'same')(c1)
    merged = layers.Concatenate(-1)([c1, p1, p2, p3])
    return CBS(merged, filters, 1, name=name)


# ==========================================
# 3. THE DYNAMIC MODEL BUILDER
# ==========================================
def build_yolo_model(variant="tiny", input_shape=(416, 416, 3), num_classes=20):
    if variant not in MODEL_CONFIGS:
        raise ValueError(f"Το variant '{variant}' δεν υπάρχει στο JSON. Διαθέσιμα: {list(MODEL_CONFIGS.keys())}")

    cfg = MODEL_CONFIGS[variant]
    inputs = tf.keras.Input(shape=input_shape)

    # === BACKBONE ===
    x = CBS(inputs, cfg["stem"][0], 3, 2, name="stem_1")
    x = CBS(x, cfg["stem"][1], 3, 2, name="stem_2")

    p3 = ELAN_Block(x, cfg["stages"][0], name="backbone_p3")
    x = layers.MaxPooling2D(2, 2, name="maxpool_1")(p3)

    p4 = ELAN_Block(x, cfg["stages"][1], name="backbone_p4")
    x = layers.MaxPooling2D(2, 2, name="maxpool_2")(p4)

    p5 = ELAN_Block(x, cfg["stages"][2], name="backbone_p5")

    # === NECK ===
    spp_out = SPP_Block(p5, cfg["stages"][2], name="spp")

    up_p5 = layers.UpSampling2D(2)(CBS(spp_out, cfg["stages"][1], 1))
    p4_fused = ELAN_Block(layers.Concatenate(-1)([CBS(p4, cfg["stages"][1], 1), up_p5]), cfg["stages"][1],
                          name="neck_p4")

    up_p4 = layers.UpSampling2D(2)(CBS(p4_fused, cfg["stages"][0], 1))
    neck_out_small = ELAN_Block(layers.Concatenate(-1)([CBS(p3, cfg["stages"][0], 1), up_p4]), cfg["stages"][0],
                                name="neck_out_small")

    down_p3 = layers.MaxPooling2D(2, 2)(neck_out_small)
    neck_out_medium = ELAN_Block(layers.Concatenate(-1)([down_p3, p4_fused]), cfg["stages"][1], name="neck_out_medium")

    down_p4 = layers.MaxPooling2D(2, 2)(neck_out_medium)
    neck_out_large = ELAN_Block(layers.Concatenate(-1)([down_p4, spp_out]), cfg["stages"][2], name="neck_out_large")

    # === HEAD ===
    out_channels = 3 * (5 + num_classes)
    out_small = layers.Conv2D(out_channels, 1, name="detect_small")(CBS(neck_out_small, cfg["head"][0], 3))
    out_medium = layers.Conv2D(out_channels, 1, name="detect_medium")(CBS(neck_out_medium, cfg["head"][1], 3))
    out_large = layers.Conv2D(out_channels, 1, name="detect_large")(CBS(neck_out_large, cfg["head"][2], 3))

    return Model(inputs, [out_small, out_medium, out_large], name=f"YOLOv7_{variant.capitalize()}")


# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    # Testάρουμε αν φορτώνει σωστά το config
    variant = "tiny"
    model = build_yolo_model(variant=variant)

    print(f"\n📊 --- Στατιστικά Μοντέλου: {variant.upper()} (Loaded from JSON) ---")
    print(f"Συνολικές Παράμετροι: {model.count_params():,}")

    # 1. Εμφανίζει τον αναλυτικό πίνακα με τα layers
    model.summary()

    # 2. Αποθηκεύει το μοντέλο σε αρχείο .h5
    save_filename = f"yolov7_{variant}_true_match.h5"
    model.save(save_filename)
    print(f"\n✅ Το μοντέλο αποθηκεύτηκε επιτυχώς στο αρχείο: {save_filename}")