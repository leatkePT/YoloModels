import tensorflow as tf
from tensorflow.keras import layers, Model
import json
import os


# =====================================================================
# 1. LOAD EXTERNAL CONFIGURATION
# =====================================================================

def load_config(config_filename="yolo11_config.json"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, config_filename)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"⚠️ Σφάλμα: Το αρχείο ρυθμίσεων '{config_path}' δεν βρέθηκε!")

    with open(config_path, "r") as f:
        return json.load(f)


MODEL_CONFIGS = load_config()


# =====================================================================
# 2. BUILDING BLOCKS (ΔΟΜΙΚΑ ΣΤΟΙΧΕΙΑ)
# =====================================================================

def CBS(x, filters, kernel_size, strides=1, groups=1, act=True, name=None):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same',
                      groups=groups, use_bias=False, name=name + '_conv' if name else None)(x)
    x = layers.BatchNormalization(name=name + '_bn' if name else None)(x)
    if act:
        x = layers.Activation('swish', name=name + '_swish' if name else None)(x)
    return x


def Bottleneck(x, c2, shortcut=True, g=1, e=0.5, name=None):
    c_ = int(c2 * e)
    y = CBS(x, c_, 3, 1, name=name + "_cv1")
    y = CBS(y, c2, 3, 1, groups=g, name=name + "_cv2")
    if shortcut:
        return layers.Add(name=name + "_add")([x, y])
    return y


def C3k(x, c2, n=1, shortcut=True, g=1, e=0.5, name=None):
    c_ = int(c2 * e)
    cv1 = CBS(x, c_, 1, name=name + "_cv1")
    cv2 = CBS(x, c_, 1, name=name + "_cv2")

    y = cv1
    for i in range(n):
        # To εσωτερικό bottleneck του C3k έχει e=1.0
        y = Bottleneck(y, c_, shortcut=shortcut, g=g, e=1.0, name=f"{name}_m_{i}")

    cat = layers.Concatenate(axis=-1, name=name + "_cat")([y, cv2])
    return CBS(cat, c2, 1, name=name + "_cv3")


def C3k2_Block(x, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True, name=None):
    c_ = int(c2 * e)
    cv1 = CBS(x, 2 * c_, 1, name=name + "_cv1")

    part1 = layers.Lambda(lambda t: t[:, :, :, :c_], name=name + "_split1")(cv1)
    part2 = layers.Lambda(lambda t: t[:, :, :, c_:], name=name + "_split2")(cv1)

    parts = [part1, part2]
    y = part2
    for i in range(n):
        if c3k:
            y = C3k(y, c_, n=2, shortcut=shortcut, g=g, name=f"{name}_m_{i}")
        else:
            y = Bottleneck(y, c_, shortcut=shortcut, g=g, e=0.5, name=f"{name}_m_{i}")
        parts.append(y)

    cat = layers.Concatenate(axis=-1, name=name + "_cat")(parts)
    return CBS(cat, c2, 1, name=name + "_cv2")


def Attention(x, dim, num_heads=4, attn_ratio=0.5, name=None):
    head_dim = dim // num_heads
    key_dim = int(head_dim * attn_ratio)
    nh_kd = key_dim * num_heads
    h = dim + nh_kd * 2

    # QKV με Batch Normalization
    qkv = layers.Conv2D(h, 1, use_bias=False, name=name + "_qkv_conv")(x)
    qkv = layers.BatchNormalization(name=name + "_qkv_bn")(qkv)

    # Positional Encoding (PE) με Batch Normalization
    pe = layers.Conv2D(dim, 3, padding='same', groups=dim, use_bias=False, name=name + "_pe_conv")(x)
    pe = layers.BatchNormalization(name=name + "_pe_bn")(pe)

    v = layers.Lambda(lambda t: t[:, :, :, nh_kd * 2:], name=name + "_v")(qkv)

    # Projection με Batch Normalization
    proj = layers.Conv2D(dim, 1, use_bias=False, name=name + "_proj_conv")(v)
    proj = layers.BatchNormalization(name=name + "_proj_bn")(proj)

    out = layers.Add(name=name + "_add_pe")([proj, pe])
    return out


def PSABlock(x, c, attn_ratio=0.5, num_heads=4, shortcut=True, name=None):
    attn_out = Attention(x, c, num_heads, attn_ratio, name=name + "_attn")
    x = layers.Add(name=name + "_add1")([x, attn_out]) if shortcut else attn_out

    ffn = layers.Conv2D(c * 2, 1, use_bias=False, name=name + "_ffn_cv1")(x)
    ffn = layers.BatchNormalization(name=name + "_ffn_bn1")(ffn)
    ffn = layers.Activation('swish', name=name + "_ffn_swish")(ffn)

    ffn = layers.Conv2D(c, 1, use_bias=False, name=name + "_ffn_cv2")(ffn)
    ffn = layers.BatchNormalization(name=name + "_ffn_bn2")(ffn)

    x = layers.Add(name=name + "_add2")([x, ffn]) if shortcut else ffn
    return x


def C2PSA(x, c2, n=1, e=0.5, name=None):
    c_ = int(c2 * e)
    cv1 = CBS(x, 2 * c_, 1, name=name + "_cv1")

    a = layers.Lambda(lambda t: t[:, :, :, :c_], name=name + "_split1")(cv1)
    b = layers.Lambda(lambda t: t[:, :, :, c_:], name=name + "_split2")(cv1)

    for i in range(n):
        b = PSABlock(b, c_, num_heads=c_ // 64, shortcut=True, name=f"{name}_psa_{i}")

    cat = layers.Concatenate(axis=-1, name=name + "_cat")([a, b])
    return CBS(cat, c2, 1, name=name + "_cv2")


def SPPF_Block(x, c2, pool_size=5, name="sppf"):
    c_ = c2 // 2
    cv1 = CBS(x, c_, 1, name=name + "_cv1")
    p1 = layers.MaxPooling2D(pool_size, 1, 'same', name=name + "_pool1")(cv1)
    p2 = layers.MaxPooling2D(pool_size, 1, 'same', name=name + "_pool2")(p1)
    p3 = layers.MaxPooling2D(pool_size, 1, 'same', name=name + "_pool3")(p2)
    cat = layers.Concatenate(axis=-1, name=name + "_cat")([cv1, p1, p2, p3])
    return CBS(cat, c2, 1, name=name + "_cv2")


def DWConv_Block(x, ch_out, name=None):
    """Depthwise-Separable Convolution για την κεφαλή κλάσεων cv3"""
    y = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same', use_bias=False, name=name + "_dw_conv")(x)
    y = layers.BatchNormalization(name=name + "_dw_bn")(y)
    y = layers.Activation('swish', name=name + "_dw_swish")(y)

    # FIX: Προσθήκη ρητά kernel_size=1
    y = CBS(y, filters=ch_out, kernel_size=1, strides=1, name=name + "_pw")
    return y


def Detect(x, num_classes=80, reg_max=16, name="detect"):
    ch0 = x[0].shape[-1]
    c2 = max(16, ch0 // 4, reg_max * 4)
    c3 = max(ch0, min(num_classes, 100))

    outputs = []
    for i, f in enumerate(x):
        # Bounding Box Branch
        box = CBS(f, c2, 3, name=f"{name}_cv2_{i}_0")
        box = CBS(box, c2, 3, name=f"{name}_cv2_{i}_1")
        box_out = layers.Conv2D(4 * reg_max, 1, use_bias=True, name=f"{name}_cv2_{i}_2")(box)

        # Classes Branch (Depthwise Convs)
        cls = DWConv_Block(f, c3, name=f"{name}_cv3_{i}_0")
        cls = DWConv_Block(cls, c3, name=f"{name}_cv3_{i}_1")
        cls_out = layers.Conv2D(num_classes, 1, use_bias=True, name=f"{name}_cv3_{i}_2")(cls)

        out = layers.Concatenate(axis=-1, name=f"{name}_out_{i}")([box_out, cls_out])
        outputs.append(out)
    return outputs


# =====================================================================
# 3. MODEL BUILDER
# =====================================================================

def build_yolo11_model(variant="nano", input_shape=(640, 640, 3)):
    if variant not in MODEL_CONFIGS:
        raise ValueError(f"Το variant '{variant}' δεν υπάρχει στο JSON. Διαθέσιμα: {list(MODEL_CONFIGS.keys())}")

    cfg = MODEL_CONFIGS[variant]
    nc = cfg["nc"]
    n = cfg["n"]
    ch = cfg["ch"]
    head_ch = cfg["head_ch"]

    # Διαβάζουμε τα δυναμικά c3k flags από το JSON.
    c3k_flags = cfg.get("c3k", [False, False, True, True, False, False, False, True])

    inputs = tf.keras.Input(shape=input_shape)

    #  BACKBONE
    x = CBS(inputs, ch[0], 3, 2, name="layer0")
    x = CBS(x, ch[1], 3, 2, name="layer1")
    x = C3k2_Block(x, ch[2], n=n, c3k=c3k_flags[0], e=0.25, name="layer2_c3k2")
    x = CBS(x, ch[3], 3, 2, name="layer3")

    p3 = C3k2_Block(x, ch[4], n=n, c3k=c3k_flags[1], e=0.25, name="layer4_c3k2")
    x = CBS(p3, ch[5], 3, 2, name="layer5")

    p4 = C3k2_Block(x, ch[6], n=n, c3k=c3k_flags[2], e=0.5, name="layer6_c3k2")
    x = CBS(p4, ch[7], 3, 2, name="layer7")

    x = C3k2_Block(x, ch[8], n=n, c3k=c3k_flags[3], e=0.5, name="layer8_c3k2")
    x = SPPF_Block(x, ch[9], name="layer9_sppf")
    p5 = C2PSA(x, ch[10], n=n, e=0.5, name="layer10_c2psa")

    #  NECK
    up1 = layers.UpSampling2D(2, interpolation="nearest", name="layer11_up")(p5)
    cat1 = layers.Concatenate(axis=-1, name="layer12_cat")([up1, p4])
    neck_p4 = C3k2_Block(cat1, head_ch[0], n=n, c3k=c3k_flags[4], e=0.5, name="layer13_c3k2")

    up2 = layers.UpSampling2D(2, interpolation="nearest", name="layer14_up")(neck_p4)
    cat2 = layers.Concatenate(axis=-1, name="layer15_cat")([up2, p3])
    out_small = C3k2_Block(cat2, head_ch[1], n=n, c3k=c3k_flags[5], e=0.5, name="layer16_c3k2")

    down1 = CBS(out_small, head_ch[1], 3, 2, name="layer17")
    cat3 = layers.Concatenate(axis=-1, name="layer18_cat")([down1, neck_p4])
    out_medium = C3k2_Block(cat3, head_ch[2], n=n, c3k=c3k_flags[6], e=0.5, name="layer19_c3k2")

    down2 = CBS(out_medium, head_ch[2], 3, 2, name="layer20")
    cat4 = layers.Concatenate(axis=-1, name="layer21_cat")([down2, p5])
    out_large = C3k2_Block(cat4, head_ch[3], n=n, c3k=c3k_flags[7], e=0.5, name="layer22_c3k2")

    #  DETECT HEAD
    detect_out = Detect([out_small, out_medium, out_large], num_classes=nc, reg_max=16, name="detect")

    return Model(inputs, detect_out, name=f"YOLO11_{variant.capitalize()}")


# =====================================================================
# 4. EXECUTION
# =====================================================================

if __name__ == "__main__":
    # Δοκίμασε τα όλα: "nano", "small", "medium", "large", "xlarge"
    variant = "xlarge"

    model = build_yolo11_model(variant=variant)

    print(f"\n🚀 Στατιστικά Μοντέλου TensorFlow: YOLO11-{variant.upper()}")
    print(f"Συνολικές Παράμετροι: {model.count_params():,}")
    print("-" * 65)

    model.summary()

    save_filename = f"yolo11_{variant}.keras"
    model.save(save_filename)
    print(f"\n💾 Το μοντέλο αποθηκεύτηκε επιτυχώς στο αρχείο: {save_filename}")