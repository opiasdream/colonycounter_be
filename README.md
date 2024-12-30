# Colony Counter Backend

Bu proje, petri kaplarındaki bakteri kolonilerini otomatik olarak sayan bir API sunucusudur.

## Özellikler

- Petri kabı görüntülerini işleme
- Otomatik koloni sayımı
- Görsel sonuç çıktısı
- RESTful API arayüzü

## Gereksinimler

- Python 3.12
- Conda (önerilen) veya pip

## Kurulum

1. Projeyi klonlayın:
```bash
git clone [repository-url]
cd colony_counter_be
```

2. Conda ile yeni bir sanal ortam oluşturun ve gerekli paketleri yükleyin:
```bash
conda create --name colony_counter --file requirements.txt
conda activate colony_counter
```

Alternatif olarak pip ile kurulum:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac için
# veya
.\venv\Scripts\activate  # Windows için

pip install -r requirements.txt
```

## Kullanım

1. API sunucusunu başlatın:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
ya da
```bash
uvicorn main:app --reload
```

2. Sunucu varsayılan olarak http://localhost:8000 adresinde çalışacaktır.

3. API dokümantasyonuna erişmek için tarayıcınızda şu adresi açın:
```
http://localhost:8000/docs
```

## API Endpoint'leri

### POST /process-image/
Petri kabı görüntüsünü işler ve koloni sayımı yapar.

**Parametreler:**
- `file`: Görüntü dosyası (multipart/form-data)
- `isInverted`: Görüntünün ters çevrilip çevrilmeyeceği (boolean, varsayılan: true)

**Yanıt:**
- İşlenmiş görüntü (PNG formatında)

## Hata Ayıklama

Eğer kurulum veya çalıştırma sırasında sorunlarla karşılaşırsanız:

1. Python ve Conda sürümlerinizin uyumlu olduğundan emin olun
2. Sanal ortamın aktif olduğundan emin olun
3. Gerekli tüm paketlerin başarıyla yüklendiğini kontrol edin