import os
import pandas as pd

def convert_csv_to_yolo(csv_file, image_folder, output_folder, image_width, image_height):
    # Reads the CSV file
    df = pd.read_csv(csv_file)

    for _, row in df.iterrows():
        image_name = row['filename']  
        xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        
        # Convert to YOLO format
        x_center = (xmin + xmax) / 2.0 / image_width
        y_center = (ymin + ymax) / 2.0 / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height
        
        yolo_annotation = f"0 {x_center} {y_center} {width} {height}\n"
        
        # Save annotation to a .txt file
        base_name = os.path.splitext(image_name)[0]
        output_path = os.path.join(output_folder, f"{base_name}.txt")
        with open(output_path, "w") as file:
            file.write(yolo_annotation)

# Example usage
convert_csv_to_yolo('bounding_boxes/train_labels.csv', 'data/train', 'data/train', 640, 480)
convert_csv_to_yolo('bounding_boxes/test_labels.csv', 'data/test', 'data/test', 640, 480)
