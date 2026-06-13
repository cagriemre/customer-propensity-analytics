# Customer Propensity Analytics

E-ticaret platformunda müşteri satın alma eğilimini tahmin eden, kullanıcı davranışlarını analiz eden ve sonuçları interaktif panolarla görselleştiren uçtan uca bir veri bilimi projesidir.

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-02569B)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/DB-PostgreSQL-4169E1?logo=postgresql&logoColor=white)
![Power BI](https://img.shields.io/badge/BI-Power%20BI-F2C811?logo=powerbi&logoColor=black)

---

## Proje Özeti

455.401 satırlık bir e-ticaret clickstream verisi üzerinde çalışarak, müşterilerin satın alma olasılıklarını tahmin eden bir makine öğrenmesi pipeline'ı geliştirildi.

| Metrik | Değer |
|---|---|
| Veri Büyüklüğü | 607.056 satır (Train + Test) |
| Hedef Değişken | `satin_aldi` (Binary) |
| Sınıf Dengesizliği | %95.8 negatif, %4.2 pozitif |
| Seçilen Model | LightGBM |
| ROC-AUC | 0.9898 |
| F1 Score | 0.7609 |
| Threshold | 0.30 (varsayılan 0.5 yerine) |
| Early Stopping | 27. iterasyon |

---

## Proje Mimarisi

```
customer-propensity-analytics/
│
├── customer-propensity-analytics.ipynb   # Veri keşfi, EDA, model eğitimi
├── lgbm_model.pkl                        # Eğitilmiş LightGBM modeli
│
├── db_setup.py                           # PostgreSQL bağlantısı ve tablo oluşturma
├── feature_importance_export.py          # Feature importance tablosu export
├── behavior_segmentation.py             # Davranış segmentasyonu ve analiz
│
├── api/
│   ├── main.py                           # FastAPI REST API (7 endpoint)
│   └── requirements.txt                  # API bağımlılıkları
│
├── app.py                                # Streamlit web uygulaması (4 sekme)
│
├── customer_propensity_scores.csv        # Tahmin sonuçları (tüm kullanıcılar)
├── customer_propensity_side_report.csv   # Detaylı rapor
│
└── CustomerPropensityDashboard.pbix      # Power BI panosu (3 sayfa)
```

---

## Teknoloji Yığını

| Katman | Teknoloji | Kullanım Amacı |
|---|---|---|
| Veri Bilimi | Python, Pandas, scikit-learn, LightGBM | Model eğitimi ve feature engineering |
| Veritabanı | PostgreSQL | Analiz tabloları ve tahmin sonuçları |
| REST API | FastAPI, Uvicorn | Modeli production'a taşıma |
| Web Arayüzü | Streamlit, Plotly | İnteraktif tahmin ve analiz |
| BI Panosu | Power BI | Yönetici panoları ve raporlama |

---

## Kurulum

### Gereksinimler

- Python 3.10+
- PostgreSQL 14+
- Power BI Desktop (opsiyonel, pano için)

### 1. Repo'yu klonlayın

```bash
git clone https://github.com/<kullanici>/customer-propensity-analytics.git
cd customer-propensity-analytics
```

### 2. Bağımlılıkları yükleyin

```bash
pip install pandas numpy scikit-learn lightgbm joblib sqlalchemy psycopg2-binary
pip install streamlit plotly fastapi uvicorn
```

### 3. PostgreSQL veritabanını oluşturun

```sql
CREATE DATABASE customer_propensityDB;
```

### 4. Tabloları oluşturun ve doldurun

```bash
python db_setup.py
python feature_importance_export.py
python behavior_segmentation.py
```

### 5. Uygulamaları başlatın

```bash
# Streamlit arayüzü
streamlit run app.py

# FastAPI (ayrı terminal)
python -c "import uvicorn; uvicorn.run('api.main:app', host='127.0.0.1', port=8000, reload=True)"
```

---

## Özellikler

### 1. Makine Öğrenmesi Pipeline'ı

- **3 model karşılaştırıldı:** Logistic Regression, LightGBM, XGBoost
- **LightGBM seçildi:** XGBoost ile aynı ROC-AUC'ye sahip ancak 27. iterasyonda durdu (XGBoost 500 iterasyon çalıştı)
- **Threshold optimizasyonu:** Sınıf dengesizliği nedeniyle 0.30 kullanıldı
- **class_weight="balanced"** ile azınlık sınıfı ağırlıklandırıldı

### 2. Davranış Segmentasyonu

`behavior_segmentation.py` — 607K kullanıcıyı analiz eder ve PostgreSQL'e 4 tablo yazar:

| Tablo | İçerik |
|---|---|
| `aksiyon_donusum_oranlari` | 16 aksiyonun satın almaya dönüşüm oranı |
| `huni_analizi` | 5 aşamalı satın alma hunisi |
| `davranis_segmentleri` | 4 davranış segmenti |
| `cihaz_donusum` | Cihaz bazında dönüşüm oranları |

**Temel bulgular:**
- Ödeme sayfasını görenler %44.47 oranında satın alıyor
- %83.66 kullanıcı sepete bile eklemiyor
- PC kullanıcıları (%4.61) mobil kullanıcılardan (%2.70) daha fazla satın alıyor

### 3. FastAPI REST API

7 endpoint ile model ve analizler dışarıya açıktır:

| Endpoint | Method | Açıklama |
|---|---|---|
| `/health` | GET | Sistem sağlığı kontrolü |
| `/predict` | POST | Tekli müşteri tahmini |
| `/predict/batch` | POST | Toplu tahmin (max 10.000) |
| `/analytics/conversion-rates` | GET | Aksiyon dönüşüm oranları |
| `/analytics/funnel` | GET | Satın alma hunisi |
| `/analytics/segments` | GET | Davranış segmentleri |
| `/analytics/device` | GET | Cihaz bazında dönüşüm |

Swagger UI: `http://localhost:8000/docs`

### 4. Streamlit Web Uygulaması

4 sekmeli interaktif arayüz:

| Sekme | İçerik |
|---|---|
| Tekli Tahmin | Form ile müşteri davranışlarını girip olasılık tahmin etme (gauge chart) |
| Davranış Analizi | PostgreSQL'den çekilen 4 analitik grafik |
| Toplu Tahmin | CSV yükleme ile toplu tahmin + sonuç indirme |
| Model Bilgileri | ROC-AUC, F1, feature importance, model karşılaştırması |

### 5. Power BI Panosu (3 Sayfa)

| Sayfa | İçerik |
|---|---|
| Genel Bakış | KPI'lar, olasılık dağılımı, segment dağılımı, model performans özeti |
| Kullanıcı Analizi | Kullanıcı tablosu, filtreleme, histogram, segment sayıları, feature importance |
| Davranış Analizi | Satın alma hunisi, aksiyon dönüşüm oranları, davranış segmentleri, cihaz dönüşüm |

---

## Veritabanı Şeması

PostgreSQL `customer_propensityDB` veritabanındaki tablolar:

| Tablo | Satır | Kaynak |
|---|---|---|
| `propensity_scores` | 151.655 | Model tahminleri (test seti) |
| `propensity_report` | 151.655 | Tahmin + segment + top yüzde |
| `feature_importance` | 16 | Model özellik önemliliği |
| `aksiyon_donusum_oranlari` | 16 | Aksiyon bazında dönüşüm |
| `huni_analizi` | 5 | Satın alma hunisi |
| `davranis_segmentleri` | 4 | Davranış segmentleri |
| `cihaz_donusum` | 3 | Cihaz bazında dönüşüm |

---

## Model Detayları

### Feature'lar (16 adet)

| Feature | Açıklama |
|---|---|
| `sepet_ikonuna_tiklama` | Sepet ikonuna tıkladı mı |
| `listeden_sepet_ekleme` | Ürün listesinden sepete ekledi mi |
| `detaydan_sepet_ekleme` | Ürün detayından sepete ekledi mi |
| `hesap_sayfasina_tiklama` | Hesap sayfasına tıkladı mı |
| `kampanyaya_tiklama` | Kampanya sayfasına tıkladı mı |
| `favoriye_ekleme` | Ürünü favorilere ekledi mi |
| `beden_secimi` | Beden seçimi yaptı mı |
| `mini_sepet_kapatma` | Mini sepeti kapattı mı |
| `iade_detay_bakma` | İade detaylarına baktı mı |
| `giris_yapma` | Siteye giriş yaptı mı |
| `odeme_sayfasi_gorme` | Ödeme sayfasını gördü mü |
| `teslimat_bilgisi_gorme` | Teslimat bilgilerini gördü mü |
| `cihaz_mobil` | Mobil cihaz kullandı mı |
| `cihaz_pc` | PC kullandı mı |
| `cihaz_tablet` | Tablet kullandı mı |
| `geri_donen_kullanici` | Geri dönen kullanıcı mı |

### Çıkarılan Feature'lar

| Feature | Çıkarılma Nedeni |
|---|---|
| `beden_tablosu_gorme` | Yetersiz gözlem |
| `teslimat_detay_bakma` | Veri sızıntısı riski |

### Model Karşılaştırması

| Model | ROC-AUC | F1 Score | İterasyon |
|---|---|---|---|
| Logistic Regression | 0.9895 | 0.689 | 1000 (max_iter) |
| **LightGBM** | **0.9898** | **0.7609** | **27 (early stop)** |
| XGBoost | 0.9898 | 0.7610 | 500 |

LightGBM seçildi: Aynı performans, daha hızlı eğitim, daha düşük bellek kullanımı.

---

## Lisans

Bu proje eğitim ve portfolyo amacıyla geliştirilmiştir.
