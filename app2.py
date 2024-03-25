from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

if __name__ == '__main__':
    # Load a model
    model = YOLO("yolov8n.yaml")  # build a new model from scratch

    # Use the model
    model.train(data=r"C:\Users\hegde\OneDrive\Desktop\componet Detection\data.yaml", epochs=3)  # train the model

    # Read the image
    img_path = r"C:\Users\hegde\OneDrive\Desktop\componet Detection\test\images\113-5-_png_jpg.rf.eddd2fbdf2dced27a6dbb26df846cbe8.jpg"
    img = cv2.imread(img_path)

    # Run prediction on the image
    results_list = model(img)

    # Iterate over each Results object in the list
    for results in results_list:
        # Check if there are any detections
        if len(results.boxes) > 0:
            # Access the bounding box coordinates
            for xmin, ymin, xmax, ymax, confidence, class_id in results.boxes[0].tolist():
                color = (0, 255, 0)  # Green color for bounding box
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                cv2.putText(img, f"{model.names[int(class_id)]} {confidence:.2f}", (int(xmin), int(ymin - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the image
    cv2.imshow("Detected Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
