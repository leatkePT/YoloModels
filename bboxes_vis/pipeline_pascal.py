
import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf

# --- Configuration ---
TARGET_SIZE = (416, 416)
BASE_DIR = os.path.abspath(os.getcwd())
IMAGE_DIR = os.path.join(BASE_DIR, "VOCdevkit", "VOC2007", "JPEGImages")
ANNOT_DIR = os.path.join(BASE_DIR, "VOCdevkit", "VOC2007", "Annotations")


def parse_voc_xml(xml_path):
    """Parses a PASCAL VOC XML file and extracts bounding boxes and class names."""
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


def visualize_and_save_grid(image_filenames, save_name="pascal_preprocessed_grid.png"):
    """Creates a grid of preprocessed images with bounding boxes and saves it."""
    num_images = len(image_filenames)

    # 1. Setup the Grid (1 row, 'num_images' columns)
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))

    # Handle the case where there's only 1 image (axes is not an array)
    if num_images == 1:
        axes = [axes]

    for i, image_filename in enumerate(image_filenames):
        ax = axes[i]

        img_path = os.path.join(IMAGE_DIR, image_filename)
        xml_path = os.path.join(ANNOT_DIR, image_filename.replace('.jpg', '.xml'))

        if not os.path.exists(xml_path):
            ax.set_title(f"No XML: {image_filename}")
            ax.axis('off')
            continue

        # Parse and Preprocess
        orig_w, orig_h, raw_boxes = parse_voc_xml(xml_path)

        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32) / 255.0
        image_resized = tf.image.resize(image, TARGET_SIZE)

        # Plot Image on the specific grid cell
        ax.imshow(image_resized.numpy())
        ax.set_title(image_filename, fontsize=12)
        ax.axis('off')

        # Math: Scale Bounding Boxes
        scale_x = TARGET_SIZE[0] / orig_w
        scale_y = TARGET_SIZE[1] / orig_h

        for box in raw_boxes:
            new_xmin = box["xmin"] * scale_x
            new_ymin = box["ymin"] * scale_y
            new_xmax = box["xmax"] * scale_x
            new_ymax = box["ymax"] * scale_y

            new_width = new_xmax - new_xmin
            new_height = new_ymax - new_ymin

            # Draw Rectangle
            rect = patches.Rectangle(
                (new_xmin, new_ymin), new_width, new_height,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)

            # Add Class Label
            ax.text(new_xmin, new_ymin - 5, box["class"], color='white',
                    fontsize=10, weight='bold', backgroundcolor='red')

    plt.tight_layout()

    # 2. SAVE THE IMAGE TO YOUR HARD DRIVE
    save_path = os.path.join(BASE_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSUCCESS: Grid saved to {save_path}")

    # 3. Show it on screen
    plt.show()


# --- Execution ---
if __name__ == "__main__":
    if not os.path.exists(IMAGE_DIR):
        print("VOCdevkit folder not found! Make sure you ran the download script.")
    else:
        print("Processing PASCAL VOC images and generating grid...")
        # Grab the first 3 images to test the pipeline
        image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')][:3]

        # This will create a 1x3 grid and save it as a .png file!
        visualize_and_save_grid(image_files, save_name="pascal_grid.png")