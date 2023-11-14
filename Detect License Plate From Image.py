from PIL import Image
import pytesseract
import cv2
from yolov8 import YOLOv8
model_path = "models/best.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.2, iou_thres=0.3)
img=cv2.imread('audi.JPG')

# Detect Objects
boxes, scores, class_ids = yolov8_detector(img)

# Draw detections
combined_img = yolov8_detector.draw_detections(img)
cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
cv2.imshow("Detected Objects", combined_img)
cv2.imwrite("doc/img/detected_objects.jpg", combined_img)
cv2.waitKey(0)

# Load your image
image = cv2.imread('audi.JPG')

# Define the coordinates [x_min, y_min, x_max, y_max]
[coordinates] = boxes
# Extract the region defined by the coordinates
x_min, y_min, x_max, y_max = [int(coord) for coord in coordinates]
roi = image[y_min:y_max, x_min:x_max]

# Display or save the extracted region as an image
cv2.imshow("Extracted Region", roi)
cv2.imwrite('extract.jpg',roi)
cv2.waitKey(0)
cv2.destroyAllWindows()

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

image = Image.open('extract.jpg')
extracted_text = pytesseract.image_to_string(image)
extracted_text = pytesseract.image_to_string(image, lang='eng', config='--psm 6')
print(extracted_text)

from xlsx import store_data_to_csv
store_data_to_csv(extracted_text)

