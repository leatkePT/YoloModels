import tensorflow as tf
from tensorflow.keras import layers, Model
import json
import os

# =====================================================================
# 1. ΑΥΤΟΜΑΤΗ ΔΗΜΙΟΥΡΓΙΑ & ΦΟΡΤΩΣΗ CONFIGURATION (JSON)
# =====================================================================

DEFAULT_CONFIG = {
    "tiny": {
        "stem": [32, 64],
        "backbone_elan": [
            {"mid": 32, "out": 64},  # Stage P3
            {"mid": 64, "out": 128},  # Stage P4
            {"mid": 128, "out": 256},  # Stage P5
            {"mid": 256, "out": 512}  # Stage P6
        ],
        "spp": {"mid": 256, "out": 256},  # Διορθώθηκε: mid και out κανάλια ρητά
        "neck_elan": [
            {"mid": 64, "out": 128},  # Neck Upper
            {"mid": 32, "out": 64},  # Neck Middle
            {"mid": 64, "out": 128},  # Neck Down 1
            {"mid": 128, "out": 256}  # Neck Down 2
        ],
        "head_convs": [128, 256, 512]
    }
}


def load_or_create_config(config_filename="yolo7_config.json"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, config_filename)

    with open(config_path, "w") as f:
        json.dump(DEFAULT_CONFIG, f, indent=4)
    print(f"🔄 Το αρχείο ρυθμίσεων ενημερώθηκε επιτυχώς: {config_path}")

    with open(config_path, "r") as f:
        return json.load(f)


MODEL_CONFIGS = load_or_create_config()


# =====================================================================
# 2. CUSTOM LAYERS ΓΙΑ ΤΟ IMPLICIT DETECT HEAD (IDetect)
# =====================================================================

class ImplicitA(layers.Layer):
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels

    def build(self, input_shape):
        self.implicit = self.add_weight(
            shape=(1, 1, 1, self.channels),
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02),
            trainable=True,
            name="implicit_add"
        )

    def call(self, inputs):
        return inputs + self.implicit


class ImplicitM(layers.Layer):
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels

    def build(self, input_shape):
        self.implicit = self.add_weight(
            shape=(1, 1, 1, self.channels),
            initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02),
            trainable=True,
            name="implicit_mul"
        )

    def call(self, inputs):
        return inputs * self.implicit


# =====================================================================
# 3. ΔΟΜΙΚΑ ΣΤΟΙΧΕΙΑ (BUILDING BLOCKS)
# =====================================================================

def CBS(x, filters, kernel_size, strides=1, name=None):
    """Conv2D + BatchNorm + LeakyReLU (Alpha=0.1) με PyTorch-aligned Padding"""
    if strides == 2:
        x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_pad' if name else None)(x)
        padding_type = 'valid'
    else:
        padding_type = 'same'

    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding_type, use_bias=False,
                      name=name + '_conv' if name else None)(x)
    x = layers.BatchNormalization(momentum=0.03, epsilon=0.001, name=name + '_bn' if name else None)(x)
    return layers.LeakyReLU(negative_slope=0.1, name=name + '_lrelu' if name else None)(x)


def ELAN_Block(x, mid_filters, out_filters, name=None):
    """Το αυθεντικό ELAN Block με Split Channels και Concatenation 4 κλάδων"""
    b1 = CBS(x, mid_filters, 1, name=name + '_b1' if name else None)
    b2 = CBS(x, mid_filters, 1, name=name + '_b2' if name else None)
    b3 = CBS(b2, mid_filters, 3, name=name + '_b3' if name else None)
    b4 = CBS(b3, mid_filters, 3, name=name + '_b4' if name else None)

    merged = layers.Concatenate(axis=-1, name=name + '_concat' if name else None)([b4, b3, b2, b1])
    return CBS(merged, out_filters, 1, name=name + '_out' if name else None)


def SPPCSPC_Block(x, mid_filters, out_filters, name="sppcspc"):
    """CSP Spatial Pyramid Pooling (SPPCSPC) Διορθωμένο για το Tiny"""
    # Κλάδος 1 (Shortcut branch)
    y1 = CBS(x, mid_filters, 1, name=f"{name}_y1")

    # Κλάδος 2 (SPP branch)
    y2 = CBS(x, mid_filters, 1, name=f"{name}_y2")
    p1 = layers.MaxPooling2D(5, strides=1, padding='same', name=f"{name}_pool1")(y2)
    p2 = layers.MaxPooling2D(9, strides=1, padding='same', name=f"{name}_pool2")(y2)
    p3 = layers.MaxPooling2D(13, strides=1, padding='same', name=f"{name}_pool3")(y2)

    merged_spp = layers.Concatenate(axis=-1, name=f"{name}_spp_concat")([y2, p1, p2, p3])
    y2_out = CBS(merged_spp, mid_filters, 1, name=f"{name}_y2_out")

    # Ένωση Κλάδων
    final_concat = layers.Concatenate(axis=-1, name=f"{name}_final_concat")([y2_out, y1])

    # Εδώ επέστρεφα mid_filters κατά λάθος. Πρέπει να είναι out_filters!
    return CBS(final_concat, out_filters, 1, name=name)


# =====================================================================
# 4. MODEL BUILDER
# =====================================================================

def build_yolo7_model(variant="tiny", input_shape=(416, 416, 3), num_classes=20):
    if variant not in MODEL_CONFIGS:
        raise ValueError(f"Το variant '{variant}' δεν βρέθηκε στο αρχείο JSON.")

    cfg = MODEL_CONFIGS[variant]
    inputs = tf.keras.Input(shape=input_shape, name="input_layer")

    # -----------------------------------------------------------------
    # BACKBONE (Εξαγωγή Χαρακτηριστικών)
    # -----------------------------------------------------------------
    x = CBS(inputs, cfg["stem"][0], 3, strides=2, name="layer0")
    x = CBS(x, cfg["stem"][1], 3, strides=2, name="layer1")

    elan1 = ELAN_Block(x, cfg["backbone_elan"][0]["mid"], cfg["backbone_elan"][0]["out"], name="elan1")
    x = layers.MaxPooling2D(2, 2, name="layer8_mp")(elan1)

    elan2 = ELAN_Block(x, cfg["backbone_elan"][1]["mid"], cfg["backbone_elan"][1]["out"], name="elan2")
    x = layers.MaxPooling2D(2, 2, name="layer15_mp")(elan2)

    elan3 = ELAN_Block(x, cfg["backbone_elan"][2]["mid"], cfg["backbone_elan"][2]["out"], name="elan3")
    x = layers.MaxPooling2D(2, 2, name="layer22_mp")(elan3)

    elan4 = ELAN_Block(x, cfg["backbone_elan"][3]["mid"], cfg["backbone_elan"][3]["out"], name="elan4")

    # -----------------------------------------------------------------
    # NECK (Πυραμίδα Χαρακτηριστικών - PANet)
    # -----------------------------------------------------------------
    spp_out = SPPCSPC_Block(elan4, cfg["spp"]["mid"], cfg["spp"]["out"], name="layer35_spp")

    up_p5 = CBS(spp_out, 128, 1, name="layer38_conv")
    up_p5_scaled = layers.UpSampling2D(2, interpolation="nearest", name="layer39_upsample")(up_p5)
    elan3_reduced = CBS(elan3, 128, 1, name="layer40_conv")
    concat_p5 = layers.Concatenate(axis=-1, name="layer41_concat")([up_p5_scaled, elan3_reduced])
    elan_neck1 = ELAN_Block(concat_p5, cfg["neck_elan"][0]["mid"], cfg["neck_elan"][0]["out"], name="elan_neck1")

    up_p4 = CBS(elan_neck1, 64, 1, name="layer48_conv")
    up_p4_scaled = layers.UpSampling2D(2, interpolation="nearest", name="layer49_upsample")(up_p4)
    elan2_reduced = CBS(elan2, 64, 1, name="layer50_conv")
    concat_p4 = layers.Concatenate(axis=-1, name="layer51_concat")([up_p4_scaled, elan2_reduced])
    elan_neck2 = ELAN_Block(concat_p4, cfg["neck_elan"][1]["mid"], cfg["neck_elan"][1]["out"], name="elan_neck2")

    down_p4 = CBS(elan_neck2, 128, 3, strides=2, name="layer58_conv")
    concat_medium = layers.Concatenate(axis=-1, name="layer59_concat")([down_p4, elan_neck1])
    elan_neck3 = ELAN_Block(concat_medium, cfg["neck_elan"][2]["mid"], cfg["neck_elan"][2]["out"], name="elan_neck3")

    down_p5 = CBS(elan_neck3, 256, 3, strides=2, name="layer66_conv")
    concat_large = layers.Concatenate(axis=-1, name="layer67_concat")([down_p5, spp_out])
    elan_neck4 = ELAN_Block(concat_large, cfg["neck_elan"][3]["mid"], cfg["neck_elan"][3]["out"], name="elan_neck4")

    # -----------------------------------------------------------------
    # HEADS & IDETECT LAYER
    # -----------------------------------------------------------------
    head_small = CBS(elan_neck2, cfg["head_convs"][0], 3, name="layer74")
    head_medium = CBS(elan_neck3, cfg["head_convs"][1], 3, name="layer75")
    head_large = CBS(elan_neck4, cfg["head_convs"][2], 3, name="layer76")

    out_channels = 3 * (5 + num_classes)

    def idetect_branch(x, in_channels, name):
        x = ImplicitA(in_channels, name=f"{name}_implicit_add")(x)
        x = layers.Conv2D(out_channels, 1, strides=1, padding='same', use_bias=True, name=f"{name}_final_conv")(x)
        x = ImplicitM(out_channels, name=f"{name}_implicit_mul")(x)
        return x

    out_small = idetect_branch(head_small, cfg["head_convs"][0], "detect_small")
    out_medium = idetect_branch(head_medium, cfg["head_convs"][1], "detect_medium")
    out_large = idetect_branch(head_large, cfg["head_convs"][2], "detect_large")

    return Model(inputs, [out_small, out_medium, out_large], name=f"YOLOv7_{variant.capitalize()}")


# =====================================================================
# 5. ΕΚΤΕΛΕΣΗ & ΕΛΕΓΧΟΣ
# =====================================================================

if __name__ == "__main__":
    variant = "tiny"
    model = build_yolo7_model(variant=variant, input_shape=(416, 416, 3), num_classes=20)

    print(f"\n🚀 Στατιστικά Μοντέλου TensorFlow: YOLOv7-{variant.upper()}")
    print(f"Συνολικές Παράμετροι: {model.count_params():,}")
    print("-" * 65)

    model.summary()

    save_filename = f"yolov7_{variant}.keras"
    model.save(save_filename)
    print(f"\n💾 Το μοντέλο αποθηκεύτηκε επιτυχώς: {save_filename}")