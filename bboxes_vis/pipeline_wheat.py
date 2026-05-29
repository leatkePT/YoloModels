import os
import ast
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf

# --- Configuration ---
TARGET_SIZE = (416, 416)
BASE_DIR = os.path.abspath(os.getcwd())
IMAGE_DIR = os.path.join(BASE_DIR, "train")
CSV_PATH = os.path.join(BASE_DIR, "train.csv")


def visualize_and_save_grid(image_ids, df_grouped, save_name="wheat_grid.png"):
    """Δημιουργεί το Grid για το Σιτάρι, κάνει Resize και ζωγραφίζει τα κουτιά."""
    num_images = len(image_ids)
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
    if num_images == 1: axes = [axes]

    for i, img_id in enumerate(image_ids):
        ax = axes[i]
        img_filename = f"{img_id}.jpg"
        img_path = os.path.join(IMAGE_DIR, img_filename)

        if not os.path.exists(img_path):
            ax.set_title("Missing Image", fontsize=10)
            ax.axis('off')
            continue

        # 1. Φόρτωση και Resize Εικόνας
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)

        # Πρέπει να βρούμε τις αρχικές διαστάσεις (συνήθως το σιτάρι είναι 1024x1024)
        orig_shape = tf.shape(image)
        orig_h, orig_w = float(orig_shape[0]), float(orig_shape[1])

        image = tf.cast(image, tf.float32) / 255.0
        image_resized = tf.image.resize(image, TARGET_SIZE)

        ax.imshow(image_resized.numpy())
        # Κόβουμε το όνομα γιατί τα IDs του Wheat είναι τεράστια (π.χ. b6ab77fd7)
        ax.set_title(img_id[:8] + "...", fontsize=12)
        ax.axis('off')

        # Υπολογισμός Κλίμακας (Scale)
        scale_x = TARGET_SIZE[0] / orig_w
        scale_y = TARGET_SIZE[1] / orig_h

        # 2. Ζωγραφική των Bounding Boxes
        group = df_grouped.get_group(img_id)  # Παίρνουμε όλα τα στάχυα για αυτή την εικόνα
        for bbox_str in group['bbox']:
            # Μετατρέπει το string "[x, y, w, h]" σε πραγματική λίστα [x, y, w, h]
            bbox = ast.literal_eval(bbox_str)
            xmin, ymin, w, h = bbox

            # Μαθηματικά: Προσαρμογή του κουτιού στο νέο 416x416 μέγεθος
            new_xmin = xmin * scale_x
            new_ymin = ymin * scale_y
            new_width = w * scale_x
            new_height = h * scale_y

            # Ζωγραφίζουμε το τετράγωνο (Επέλεξα Χρυσό/Κίτρινο χρώμα για το σιτάρι!)
            rect = patches.Rectangle(
                (new_xmin, new_ymin), new_width, new_height,
                linewidth=1.5, edgecolor='gold', facecolor='none'
            )
            ax.add_patch(rect)

    plt.tight_layout()
    save_path = os.path.join(BASE_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSUCCESS: Grid saved to {save_path}")
    plt.show()


# --- Execution ---
if __name__ == "__main__":
    if not os.path.exists(CSV_PATH) or not os.path.exists(IMAGE_DIR):
        print("ΣΦΑΛΜΑ: Το αρχείο train.csv ή ο φάκελος train/ δεν βρέθηκαν.")
    else:
        print("Διαβάζουμε το CSV αρχείο...")
        # Διαβάζουμε όλο το CSV με τη βιβλιοθήκη Pandas
        df = pd.read_csv(CSV_PATH)

        # Ομαδοποιούμε τις εγγραφές με βάση το ID της εικόνας
        grouped = df.groupby('image_id')
        unique_image_ids = list(grouped.groups.keys())

        print(f"Βρέθηκαν {len(unique_image_ids)} μοναδικές εικόνες στο CSV.")

        # Χρησιμοποιούμε το Slicing που μάθαμε!
        # Παίρνουμε από την 100ή εικόνα μέχρι την 300ή, ανά 50 εικόνες (άρα 4 εικόνες σύνολο)
        test_ids = unique_image_ids[100:300:50]

        print(f"Δημιουργία grid για {len(test_ids)} εικόνες...")
        visualize_and_save_grid(test_ids, grouped, save_name="wheat_grid.png")