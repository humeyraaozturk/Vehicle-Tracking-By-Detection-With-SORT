# YOLOv8 + SORT ile Taşıt Takibi

Bu projede, video görüntüleri üzerinde **YOLOv8** kullanılarak insan ve taşıt tespiti yapılmakta,
tespit edilen nesneler **SORT (Simple Online and Realtime Tracking)** algoritması ile çoklu nesne takibine tabi tutulmaktadır.

## Kullanılan Teknolojiler

* Python 3.9
* YOLOv8 / YOLOv5 (Ultralytics)
* SORT Tracker
* OpenCV
* PyTorch
* NumPy

---

## Python Sürümü

Bu proje **Python 3.9** kullanılarak gerçekleştirilmiştir.

> Python 3.10 ve üzeri sürümler, bazı bağımlılıklar (özellikle NumPy, scikit-image ve SORT) ile
> uyumsuzluk sorunlarına yol açabilmektedir.

---

## Kurulum

### Sanal Ortam Oluşturma (Önerilir)

```
conda create -n yolov8_sort python=3.9 -y
conda activate yolov8_sort
```

### Gerekli Repoların Kurulması

```
https://github.com/abewley/sort.git
cd sort
pip install -r requirements.txt
```
```
https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
```

### Gerekli Kütüphanelerin Kurulması

```
pip install -r requirements.txt
```
---

## Model Ağırlıkları (Weights)

**Ağırlık dosyalarını [Drive](https://drive.google.com/drive/folders/1mgFSqSiPCqTaGy71pQKQMmTPm6L5tqiX?usp=sharing) üzerinden indirerek aşağıdaki klasöre yerleştirmeniz yeterlidir:**

```
weights/
 └── best_yolov8.pt
```

---

## Video Üzerinde Çalıştırma

Ana uygulama dosyası:

```
python main_sort_yolov8.py
```

Kod içerisinde:

* Video dosya yolu
* Confidence threshold

kolayca değiştirilebilir.

---

## GPU (CUDA) Desteği

GPU desteği için:

* NVIDIA ekran kartı
* Sisteme uygun CUDA sürümü
* CUDA sürümüne **uyumlu PyTorch** kurulumu gereklidir

CUDA versiyonunu öğrenmek için:
```
nvcc --version
```

CUDA durumunu kontrol etmek için:

```
import torch
print("CUDA available:", torch.cuda.is_available())
```

* `CUDA available: False` → CPU kullanılıyor
* `CUDA available: True` → GPU aktif
