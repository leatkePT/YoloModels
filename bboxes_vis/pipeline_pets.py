import os
import urllib.request
import tarfile
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf

# --- Configuration ---
TARGET_SIZE = (416, 416)
BASE_DIR = os.path.abspath(os.getcwd())
IMAGE_DIR = os.path.join(BASE_DIR, "images")
ANNOT_DIR = os.path.join(BASE_DIR, "annotations", "xmls")


def download_pets_annotations():
    """Κατεβάζει τα XML Bounding Boxes αν δεν υπάρχουν ήδη."""
    annot_tar_path = os.path.join(BASE_DIR, "annotations.tar.gz")
    if not os.path.exists(ANNOT_DIR):
        if not os.path.exists(annot_tar_path):
            print("Downloading Oxford Pets Annotations (19MB)...")
            url = "https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz"
            urllib.request.urlretrieve(url, annot_tar_path)

        print("Extracting Annotations...")
        with tarfile.open(annot_tar_path, 'r:gz') as tar_ref:
            tar_ref.extractall(BASE_DIR)
    else:
        print("Annotations already exist.")


def parse_voc_xml(xml_path):
    """Το Oxford Pets χρησιμοποιεί το ίδιο ακριβώς XML format με το PASCAL!"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find('size')
    orig_w = float(size.find('width').text)
    orig_h = float(size.find('height').text)

    boxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')

        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        boxes.append({"class": name, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})

    return orig_w, orig_h, boxes


def visualize_and_save_grid(image_filenames, save_name="pets_thesis_grid.png"):
    """Δημιουργεί το Grid (Όπως στο PASCAL)"""
    num_images = len(image_filenames)
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
    if num_images == 1: axes = [axes]

    for i, image_filename in enumerate(image_filenames):
        ax = axes[i]

        img_path = os.path.join(IMAGE_DIR, image_filename)
        xml_path = os.path.join(ANNOT_DIR, image_filename.replace('.jpg', '.xml'))

        orig_w, orig_h, raw_boxes = parse_voc_xml(xml_path)

        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32) / 255.0
        image_resized = tf.image.resize(image, TARGET_SIZE)

        ax.imshow(image_resized.numpy())
        ax.set_title(image_filename, fontsize=12)
        ax.axis('off')

        scale_x = TARGET_SIZE[0] / orig_w
        scale_y = TARGET_SIZE[1] / orig_h

        for box in raw_boxes:
            new_xmin = box["xmin"] * scale_x
            new_ymin = box["ymin"] * scale_y
            new_xmax = box["xmax"] * scale_x
            new_ymax = box["ymax"] * scale_y
            new_width = new_xmax - new_xmin
            new_height = new_ymax - new_ymin

            rect = patches.Rectangle(
                (new_xmin, new_ymin), new_width, new_height,
                linewidth=2, edgecolor='lime', facecolor='none'  # Πράσινο χρώμα για να ξεχωρίζει από το PASCAL!
            )
            ax.add_patch(rect)

            ax.text(new_xmin, new_ymin - 5, box["class"], color='black',
                    fontsize=10, weight='bold', backgroundcolor='lime')

    plt.tight_layout()
    save_path = os.path.join(BASE_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSUCCESS: Grid saved to {save_path}")
    plt.show()


# --- Execution ---
if __name__ == "__main__":
    download_pets_annotations()

    if not os.path.exists(IMAGE_DIR):
        print("Error: The 'images' folder is missing! Make sure your previous script downloaded the images.")
    else:
        print("Finding images that actually have XML bounding boxes...")

        # ΠΑΓΙΔΑ: Βρίσκουμε ΜΟΝΟ τις εικόνες που έχουν αντίστοιχο .xml!
        valid_images = []
        for img_file in os.listdir(IMAGE_DIR):
            if img_file.endswith('.jpg'):
                xml_file = img_file.replace('.jpg', '.xml')
                if os.path.exists(os.path.join(ANNOT_DIR, xml_file)):
                    valid_images.append(img_file)

        print(f"Found {len(valid_images)} images with valid bounding boxes out of {len(os.listdir(IMAGE_DIR))}.")

        # Ταξινομούμε πρώτα τη λίστα για να είμαστε σίγουροι ότι έχουν σταθερή σειρά
        # (αλλιώς τα Windows τα διαβάζουν με τυχαία σειρά κάθε φορά)
        valid_images.sort()

        # Εδώ βάζεις το [start:stop:step] σου!
        # Από την 300ή εικόνα έως την 400ή, ανά 10 εικόνες.
        test_images = valid_images[300:350:10]

        print(f"Επιλέχθηκαν {len(test_images)} εικόνες για το grid.")

        if len(test_images) > 0:
            visualize_and_save_grid(test_images, save_name="oxford_grid.png")
        else:
            print("Δώσατε όρια εκτός της λίστας!")