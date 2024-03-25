from ultralytics import YOLO
from PIL import Image
import cv2

if __name__ == '__main__':
    # Load a model
    model = YOLO("yolov8n.yaml")  # build a new model from scratch

    # Use the model
    model.train(data=r"C:\Users\hegde\OneDrive\Desktop\componet Detection\data.yaml", epochs=30)  # train the model

    # Read the image
    img = cv2.imread(r"C:\Users\hegde\OneDrive\Desktop\componet Detection\test\images\113-5-_png_jpg.rf.eddd2fbdf2dced27a6dbb26df846cbe8.jpg")

    # Run prediction on the image
    results = model(img)

    print(results)
