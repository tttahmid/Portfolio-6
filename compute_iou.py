# compute_iou.py
import cv2
import csv

def compute_iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0

def save_iou_results(results, output_csv):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["image_name", "confidence_value", "iou_value"])
        writer.writerows(results)

# Example usage
if __name__ == "__main__":
    # Mock data for testing
    boxes = [
        {"image_name": "image1.jpg", "confidence_value": 0.9, "box": [50, 50, 100, 100]},
        {"image_name": "image1.jpg", "confidence_value": 0.85, "box": [75, 75, 150, 150]},
        {"image_name": "image2.jpg", "confidence_value": 0.95, "box": [20, 20, 60, 60]},
        {"image_name": "image2.jpg", "confidence_value": 0.8, "box": [40, 40, 80, 80]},
    ]

    results = []
    for i in range(0, len(boxes), 2):
        box1 = boxes[i]["box"]
        box2 = boxes[i + 1]["box"]
        iou_value = compute_iou(box1, box2)
        results.append([boxes[i]["image_name"], boxes[i]["confidence_value"], iou_value])

    save_iou_results(results, "iou_results.csv")
    print("IoU results saved to iou_results.csv")
