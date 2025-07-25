import cv2
from ultralytics import YOLO
import argparse
import os

def detect_humans_on_image(image_path, output_path=None, conf_threshold=0.5):
    """
    Belirtilen resim dosyası üzerinde insan tespiti yapar.

    Args:
        image_path (str): Tespitin yapılacağı resim dosyasının yolu (örn: 'input/sample_image.jpg').
        output_path (str, optional): İşlenmiş çıktının kaydedileceği yol.
                                     Varsayılan olarak None (kaydedilmez).
        conf_threshold (float, optional): Tespit güvenilirlik eşiği.
                                          Bu eşikten düşük güvenilirliğe sahip
                                          tespitler filtrelenir. Varsayılan 0.5.
    """

    # YOLOv8 modelini yükle
    # 'yolov8s.pt' (small) boyutu daha küçük, daha hızlı ve yeterince doğru.
    # İlk çalıştırmada model otomatik olarak indirilir.
    model = YOLO('yolov8s.pt') 

    # Resim dosyasının varlığını kontrol et
    if not os.path.isfile(image_path):
        print(f"Hata: Resim dosyası bulunamadı: {image_path}")
        return
    
    # Desteklenen resim formatlarını kontrol et
    if not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
        print(f"Hata: Desteklenmeyen resim formatı: {image_path}. Lütfen bir resim (jpg, png vb.) dosyası girin.")
        return

    # YOLO sınıf etiketlerini al (0 genellikle "person"dır)
    class_names = model.names
    person_class_id = None
    for class_id, class_name in class_names.items():
        if class_name == 'person':
            person_class_id = class_id
            break

    if person_class_id is None:
        print("Uyarı: 'person' sınıfı modelin etiketlerinde bulunamadı. Lütfen modelin etiketlerini kontrol edin.")
        return

    # Renkler (BGR formatında)
    box_color = (0, 255, 0)  # Yeşil
    text_color = (255, 255, 255) # Beyaz

    print(f"Resim işleniyor: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Hata: Resim dosyası okunamadı: {image_path}")
        return
    
    # Tespit yap
    # verbose=False, konsol çıktısını azaltır
    results = model.predict(img, conf=conf_threshold, verbose=False)

    # Sadece 'person' sınıfından olanları filtrele ve çiz
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy() # [x1, y1, x2, y2]
        confs = r.boxes.conf.cpu().numpy()
        class_ids = r.boxes.cls.cpu().numpy()

        for box, conf, class_id in zip(boxes, confs, class_ids):
            if int(class_id) == person_class_id:
                x1, y1, x2, y2 = map(int, box)
                label = f"{class_names[int(class_id)]}: {conf:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
                
                # Metni kutunun üzerine çiz
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width, y1), box_color, -1) # Arka plan
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
    
    # Sonucu göster
    cv2.imshow("YOLOv8 Insan Tespiti", img)
    cv2.waitKey(0) # Kullanıcı bir tuşa basana kadar bekle
    cv2.destroyAllWindows() # Tüm pencereleri kapat

    # Sonucu kaydet
    if output_path:
        # Çıktı klasörünün varlığını kontrol et, yoksa oluştur
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        cv2.imwrite(output_path, img)
        print(f"İşlenmiş resim kaydedildi: {output_path}")
    
    print("İşlem tamamlandı.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 ile görüntüde insan tespiti.")
    parser.add_argument(
        "--source", 
        type=str, 
        required=True, 
        help="Tespitin yapılacağı resim dosyasının yolu (örn: 'input/sample_image.jpg')."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None, 
        help="İşlenmiş çıktının kaydedileceği yol (örn: 'output/result.jpg')."
    )
    parser.add_argument(
        "--conf", 
        type=float, 
        default=0.5, 
        help="Tespit güvenilirlik eşiği (0.0 ile 1.0 arası)."
    )

    args = parser.parse_args()

    detect_humans_on_image(args.source, args.output, args.conf)