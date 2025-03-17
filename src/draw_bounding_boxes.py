import os
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

def draw_bounding_boxes(input_folder, output_folder):
    # Create output folder structure
    for root_dir, dirs, files in os.walk(input_folder):
        for dir_name in dirs:
            relative_path = os.path.relpath(os.path.join(root_dir, dir_name), input_folder)
            output_subfolder = os.path.join(output_folder, relative_path)
            os.makedirs(output_subfolder, exist_ok=True)

    # Process all images
    for root_dir, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.jpg'):
                # Get image and XML paths
                image_path = os.path.join(root_dir, file)
                xml_path = os.path.join(root_dir, os.path.splitext(file)[0] + '.xml')

                if not os.path.exists(xml_path):
                    print(f"Warning: No XML file for {file}")
                    continue

                # Load image and XML
                image = Image.open(image_path)
                tree = ET.parse(xml_path)
                root = tree.getroot()

                # Draw bounding boxes
                draw = ImageDraw.Draw(image)
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    bbox = obj.find('bndbox')
                    xmin = int(bbox.find('xmin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymin = int(bbox.find('ymin').text)
                    ymax = int(bbox.find('ymax').text)
                    color = 'yellow' if name == '0' else 'purple'
                    draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)

                # Save to output folder
                relative_path = os.path.relpath(root_dir, input_folder)
                output_subfolder = os.path.join(output_folder, relative_path)
                output_image_path = os.path.join(output_subfolder, file)
                image.save(output_image_path)
                print(f"Processed: {image_path} -> {output_image_path}")

if __name__ == "__main__":
    input_folder = "C:\\Users\\HITECH\\Desktop\\projectpy\\hijab-detection\\data"  # Path to original data (contains train/val/test)
    output_folder = "C:\\Users\\HITECH\\Desktop\\projectpy\\hijab-detection\\data_with_bboxes"  # Output directory
    draw_bounding_boxes(input_folder, output_folder)