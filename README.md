# 👁️ YOLO: Teorik Temeller, Sürümler ve Uygulama

## 1. YOLO Nedir?

**YOLO (You Only Look Once)**, gerçek zamanlı nesne tespiti için geliştirilmiş, tek geçişli (single-shot) ve son derece hızlı bir derin öğrenme algoritmasıdır. YOLO'nun temel yaklaşımı, bir görüntüyü tek seferde (tek bir ileri besleme ile) bölgelere ayırıp, her bölge için nesne sınıfı ve konumunu aynı anda tahmin etmektir. Bu sayede hem hız hem de doğruluk açısından günümüzün en popüler nesne tespit algoritmalarından biri olmuştur.

### YOLO'nun Temel Özellikleri
- **Tek Aşamalı (Single-Shot):** Görüntüdeki tüm nesneleri tek bir ileri besleme ile tespit eder.
- **Gerçek Zamanlı:** Yüksek FPS ile çalışabilir, gömülü sistemlerde ve canlı video akışlarında kullanılabilir.
- **Uçtan Uca Öğrenme:** Kutu koordinatları ve sınıf tahminleri aynı ağdan çıkar.
- **Genel Amaçlı:** İnsan, araç, hayvan gibi birçok nesne sınıfını aynı anda tespit edebilir.

---

## 2. YOLO Sürümleri ve Evrimi

YOLO algoritması, yıllar içinde birçok sürümle geliştirilmiştir. Her yeni sürüm, hız, doğruluk ve mimari açısından önemli yenilikler getirmiştir.

| Sürüm     | Yıl  | Temel Özellikler                                                                 |
|-----------|------|---------------------------------------------------------------------------------|
| YOLOv1    | 2016 | İlk sürüm, tek grid yaklaşımı, hızlı ama düşük doğruluk.                          |
| YOLOv2    | 2017 | Anchor box, batch normalization, daha iyi doğruluk.                              |
| YOLOv3    | 2018 | Darknet-53 backbone, çok ölçekli tespit, daha iyi küçük nesne tespiti.           |
| YOLOv4    | 2020 | CSPDarknet, Mosaic data augmentation, daha yüksek doğruluk ve hız.               |
| YOLOv5    | 2020 | PyTorch tabanlı, kolay kullanım, modüler yapı, topluluk desteği.                 |
| YOLOv7    | 2022 | Optimize edilmiş mimari, daha iyi hız/doğruluk dengesi.                          |
| YOLOv8    | 2023 | Anchor-free, yeni başlatıcılar, daha iyi genel doğruluk, sadeleştirilmiş yapı.   |

### YOLOv8'in Öne Çıkan Özellikleri
- **Anchor-free:** Kutu tahmininde sabit anchor box kullanılmaz, doğrudan kutu koordinatları tahmin edilir.
- **Daha Yüksek Doğruluk:** Özellikle küçük nesnelerde ve karmaşık sahnelerde daha iyi sonuçlar.
- **Kolay Entegrasyon:** Ultralytics tarafından sağlanan PyTorch tabanlı arayüz ile hızlı kurulum ve kullanım.

---

## 3. YOLO'nun Çalışma Prensibi

1. **Girdi Görüntüsü** sabit boyutlu bir grid'e bölünür (ör. 13x13).
2. Her grid hücresi, o bölgede nesne olup olmadığını ve varsa kutu koordinatlarını + sınıfını tahmin eder.
3. Tüm tahminler birleştirilir, gereksiz kutular NMS (Non-Maximum Suppression) ile elenir.
4. Sonuç olarak, her nesne için bir sınıf ve kutu koordinatı elde edilir.

---

## 4. Bu Projede YOLO Nasıl Kullanılıyor?

- **YOLOv8** modeli, hem resim hem de video üzerinde insan tespiti için kullanılır.
- Video modunda, tespit edilen kişilere DeepSORT algoritması ile benzersiz ID atanır ve takip edilir.
- Farklı YOLOv8 ağırlıkları (nano, medium, xlarge) ile hız/doğruluk dengesi ayarlanabilir.

---

## 5. Proje Kullanımı

### Kurulum
```bash
pip install ultralytics opencv-python deep_sort_realtime numpy
```

### Resimden İnsan Tespiti
```bash
python YOLO_Image_Detection.py
```

### Videodan İnsan Takibi
```bash
python YOLO_Video_Detection.py
```

---

## 6. Sıkça Sorulan Sorular (SSS)

**S: YOLO neden bu kadar hızlı?**
C: Tüm nesne tespitini tek bir ileri besleme ile yaptığı için, klasik R-CNN tabanlı yöntemlerden çok daha hızlıdır.

**S: Hangi sürümü kullanmalıyım?**
C: Donanımınız güçlüyse YOLOv8x, daha hızlı sonuç için YOLOv8n veya YOLOv8m kullanabilirsiniz.

**S: Sadece insan mı tespit ediliyor?**
C: Kodda sadece "person" (sınıf 0) filtreleniyor. Diğer nesneler için bu filtreyi değiştirebilirsiniz.

---

## 7. Kaynaklar
- [YOLOv8 Resmi Dokümantasyon](https://docs.ultralytics.com/)
- [YOLOv4 Paper](https://arxiv.org/abs/2004.10934)
- [YOLOv1 Paper](https://arxiv.org/abs/1506.02640)
- [DeepSORT Paper](https://arxiv.org/abs/1703.07402)

---

## 8. YOLO ve DeepSORT Nasıl Çalışır?

### YOLO (You Only Look Once) Algoritmasının Çalışma Prensibi

1. **Girdi Görüntüsünü Grid'lere Böler:** Görüntü, örneğin 13x13 gibi sabit boyutlu hücrelere ayrılır.
2. **Her Grid için Tahmin:** Her hücre, o bölgede nesne olup olmadığını, varsa kutu koordinatlarını (x, y, w, h) ve sınıf olasılıklarını tahmin eder.
3. **Tüm Tahminlerin Birleştirilmesi:** Tüm gridlerden gelen kutular birleştirilir.
4. **NMS (Non-Maximum Suppression):** Birbirine çok yakın ve aynı sınıfa ait kutular elenir, en yüksek skorlu olanlar bırakılır.
5. **Sonuç:** Her nesne için bir kutu ve sınıf etiketi elde edilir.

**Avantajı:** Tüm bu işlemler tek bir ileri besleme (forward pass) ile yapılır, bu da YOLO'yu çok hızlı yapar.

---

### DeepSORT Algoritmasının Çalışma Prensibi

**DeepSORT (Simple Online and Realtime Tracking with a Deep Association Metric)**, nesne tespitlerinden gelen kutuları (ör. YOLO'dan) alır ve her nesneye benzersiz bir ID atayarak, nesneleri kareler boyunca takip eder.

1. **Girdi:** Her karedeki tespit edilen kutular (bounding box) ve sınıf bilgisi.
2. **Özellik Çıkarımı:** Her kutu için bir görsel özellik vektörü (embedding) çıkarılır (derin öğrenme tabanlı).
3. **Tahmin (Prediction):** Her takip edilen nesne için bir sonraki karede nerede olacağı tahmin edilir (Kalman filtresi ile).
4. **Eşleştirme (Association):** Yeni tespitler ile mevcut izler (track) arasında, hem konum hem de görsel benzerliğe göre eşleştirme yapılır (Hungarian algoritması).
5. **ID Atama:** Eşleşen kutulara aynı ID atanır, yeni nesnelere yeni ID verilir.
6. **Kayıp Takipler:** Bir nesne birkaç kare boyunca görünmezse, takibi sonlandırılır.

**Avantajı:** Hem konumsal hem de görsel benzerlik kullandığı için, nesneler birbirine yaklaşsa bile ID karışıklığı az olur.

---

### Projede YOLO + DeepSORT Entegrasyonu

- YOLO, her karede insanları tespit eder ve kutu koordinatlarını çıkarır.
- DeepSORT, bu kutuları alır ve her kişiye benzersiz bir ID atar, kişileri kareler boyunca takip eder.
- Sonuç olarak, her kişiye ait kutu ve ID, video boyunca tutarlı şekilde gösterilir.
- Toplam tespit edilen kişi sayısı ve anlık takip edilen kişiler ekranda gösterilir.
