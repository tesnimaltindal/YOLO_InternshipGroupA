from ultralytics import YOLO
import cv2

# 1. YOLOv8 modeli yükleniyor (önceden eğitilmiş)
model = YOLO("yolov8n.pt")  # 'n' = nano, hızlı ve küçük

# 2. Kameradan görüntü alma
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Frame üzerinde tahmin yap
    results = model(frame, imgsz=640, conf=0.5)  # 0.5 confidence threshold

    # 4. Tahmin sonuçlarını çiz (bounding box'lar)
    annotated_frame = results[0].plot()

    # 5. Göster
    cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

    # Çıkış için ESC'ye bas
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
