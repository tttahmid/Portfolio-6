import cv2
import torch

def detect_graffiti_in_video(video_path):
    # Load the pre-trained YOLOv5 nano model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every 5th frame
        if frame_count % 5 == 0:
            results = model(frame)  # Perform inference

            # Draw bounding boxes and labels for each detection
            for det in results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = det
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, f'Confidence: {conf:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        frame_count += 1
        cv2.imshow("Graffiti Detection", frame)  # Display the frame with detections

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()  # Release the video capture object
    cv2.destroyAllWindows()  # Close all OpenCV windows

if __name__ == "__main__":
    video_path = r'D:\Uni\AI for Engineering\Week 6\sample video.mp4'  # Change this to your video path
    detect_graffiti_in_video(video_path)
