from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np
import time

# Hız için küçük giriş boyutu ve büyük model seçildi
model = YOLO("yolov8x.pt")
tracker = DeepSort(max_age=30, n_init=1, nms_max_overlap=1.0)

input_path = "Test1.mp4"
cap = cv2.VideoCapture(input_path)
total_ids = set()

start_time = time.time()
frame_count = 0
last_detections = []

# FPS artışı için ayarlar
detect_width = 320
imgsz = 320
DETECT_EVERY = 3 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    h, w = frame.shape[:2]
    scale = detect_width / w
    frame_small = cv2.resize(frame, (detect_width, int(h * scale)))

    detections = []
    # Her 3 karede bir YOLO ile tespit etme
    if frame_count % DETECT_EVERY == 0:
        results = model(frame_small, imgsz=imgsz)[0]
        for box in results.boxes.data:
            x1, y1, x2, y2, score, cls = box
            if int(cls) == 0 and score > 0.5:
                # Orijinal boyuta geri ölçekleme
                x1o = x1.item() / scale
                y1o = y1.item() / scale
                x2o = x2.item() / scale
                y2o = y2.item() / scale
                detections.append(([x1o, y1o, x2o-x1o, y2o-y1o], score.item(), 'person'))
        last_detections = detections
    else:
        # Önceki tespitleri kullanıyoruz.(DeepSORT update için)
        detections = last_detections

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        total_ids.add(track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Yeniden boyutlandırma
    display_width = 800
    h, w = frame.shape[:2]
    scale_disp = display_width / w
    frame_resized = cv2.resize(frame, (display_width, int(h * scale_disp)))

    cv2.putText(frame_resized, f"Toplam Kisi Sayisi: {len(total_ids)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow("YOLO + DeepSORT İnsan Takip", frame_resized)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

elapsed = time.time() - start_time
fps = frame_count / elapsed if elapsed > 0 else 0
print(f"Toplam tespit edilen kişi sayısı: {len(total_ids)}")
print(f"Tespit edilen ID'ler: {sorted(list(total_ids))}")
print(f"Ortalama FPS: {fps:.2f}") 