from ultralytics import YOLO
import cv2
import os

# Model yükleme
model = YOLO("yolov8n.pt")

# Giriş resmini belirtme
input_path = "persons2.jpg" 
image = cv2.imread(input_path)
results = model(image)[0]

person_count = 0
for box in results.boxes.data:
    x1, y1, x2, y2, score, cls = box
    if int(cls) == 0 and score > 0.5:
        person_count += 1
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

# Yeniden boyutlandırma
display_width = 800
h, w = image.shape[:2]
scale = display_width / w
image_resized = cv2.resize(image, (display_width, int(h * scale)))

cv2.imshow("YOLO İnsan Tespiti", image_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(f"Toplam tespit edilen kişi: {person_count}") 