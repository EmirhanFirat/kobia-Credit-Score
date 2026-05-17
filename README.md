# 💳 Kobia Credit Scorer

**Kredi kartı müşterilerinin bir sonraki ay temerrüde düşme olasılığını tahmin eden, uçtan uca makine öğrenmesi projesi.**

> UCI Credit Card Default veri seti üzerinde eğitilmiş XGBoost tabanlı sınıflandırma modeli; modüler mimari, otomatik ön işleme pipeline'ı ve FastAPI ile servis edilmeye hazır yapı.

---

## 📌 Proje Özeti

Bu proje, Tayvan'daki kredi kartı müşterilerine ait demografik bilgiler, ödeme geçmişi ve fatura tutarlarını kullanarak **temerrüt (default) riskini** tahmin eder. Projede SOLID prensiplerine uygun, üretime yakın bir yazılım mimarisi benimsenmiştir.

| Özellik | Detay |
|---|---|
| **Veri Seti** | UCI — Default of Credit Card Clients (30.000 kayıt, 24 özellik) |
| **Hedef Değişken** | `default_payment_next_month` (0: Ödeme yapacak, 1: Temerrüt) |
| **Model** | XGBoost (Gradient Boosting) |
| **Başarı Metrikleri** | Accuracy: **%81** · ROC-AUC: **0.755** |
| **Servis** | FastAPI REST API (geliştirme aşamasında) |

---

## 🏗️ Proje Mimarisi

```
kobia-credit-scorer/
│
├── data/                          # Veri yönetimi (proje kökü)
│   ├── raw/                       # Ham veri dosyaları (CSV, XLS)
│   ├── processed/                 # İşlenmiş veri
│   ├── external/                  # Harici veri kaynakları
│   └── loader.py                  # Veri yükleme fonksiyonları
│
├── models/                        # Eğitilmiş model artefaktları
│   ├── xgb_pipeline.joblib        # Serileştirilmiş model
│   └── xgb_pipeline.json          # Model metadata & performans raporu
│
├── notebooks/
│   └── EDA.ipynb                  # Keşifsel Veri Analizi
│
├── src/                           # Kaynak kod
│   ├── data/                      # Veri modülü
│   ├── features/
│   │   └── pipeline.py            # Ön işleme & özellik mühendisliği
│   ├── models/
│   │   ├── train.py               # Model eğitim script'i
│   │   ├── evaluate.py            # Değerlendirme metrikleri & görselleştirme
│   │   └── persistence.py         # Model kaydetme & metadata yönetimi
│   └── api/                       # FastAPI servisi (geliştirme aşamasında)
│
├── tests/                         # Birim testler
├── requirements.txt               # Python bağımlılıkları
└── README.md
```

---

## ⚙️ Teknik Detaylar

### Ön İşleme Pipeline'ı

Veri türüne göre üç ayrı pipeline ile otomatik dönüşüm uygulanır:

| Pipeline | Sütunlar | İşlemler |
|---|---|---|
| **Nümerik** | `limit_bal`, `age`, `bill_amt1-6`, `pay_amt1-6` | Median imputation → Standard Scaling |
| **Kategorik** | `sex`, `education`, `marriage` | Mode imputation → One-Hot Encoding |
| **Ordinal** | `pay_0`, `pay_2-6` | Mode imputation → Ordinal Encoding |

> **Tasarım Kararı:** `build_pipeline()` fonksiyonu **Dependency Injection** deseni ile çalışır — istenen herhangi bir sınıflandırıcı (XGBoost, LightGBM vb.) pipeline'a enjekte edilebilir. Bu sayede **SOLID — Open/Closed** prensibine uyulmuştur.

### Model Eğitimi

- **Train/Test Ayrımı:** %80 / %20 — `stratify` parametresi ile sınıf dağılımı korunur
- **Algoritma:** `XGBClassifier` (binary:logistic)
- **Değerlendirme:** Classification Report, Confusion Matrix, ROC-AUC eğrisi

### Model Persistence

Eğitilen model iki dosya olarak saklanır:
- **`.joblib`** — Serileştirilmiş pipeline (preprocessing + model)
- **`.json`** — Metadata: versiyon bilgileri, eğitim konfigürasyonu, hiperparametreler ve performans metrikleri

---

## 📊 Model Performansı

```
              precision    recall  f1-score   support

           0       0.84      0.94      0.88      4673
           1       0.62      0.36      0.45      1327

    accuracy                           0.81      6000
   macro avg       0.73      0.65      0.67      6000
weighted avg       0.79      0.81      0.79      6000
```

| Metrik | Değer |
|---|---|
| **Accuracy** | 0.81 |
| **ROC-AUC** | 0.755 |
| **Precision (Default=1)** | 0.62 |
| **Recall (Default=1)** | 0.36 |

**Confusion Matrix:**

|  | Tahmin: 0 | Tahmin: 1 |
|---|---|---|
| **Gerçek: 0** | 4382 | 291 |
| **Gerçek: 1** | 853 | 474 |

---

## 🚀 Kurulum & Çalıştırma

### Gereksinimler

- Python 3.10+
- pip

### Kurulum

```bash
# Repo'yu klonlayın
git clone https://github.com/<kullanici>/kobia-credit-scorer.git
cd kobia-credit-scorer

# Sanal ortam oluşturun
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Bağımlılıkları yükleyin
pip install -r requirements.txt
```

### Model Eğitimi

```bash
python src/models/train.py
```

Bu komut:
1. Ham veriyi `data/raw/` klasöründen yükler
2. Ön işleme pipeline'ını uygular
3. XGBoost modelini eğitir
4. Performans metriklerini ve ROC/Confusion Matrix grafiklerini gösterir
5. Modeli ve metadata'yı `models/` klasörüne kaydeder

---

## 🧰 Teknoloji Yığını

| Kategori | Teknoloji |
|---|---|
| **Veri İşleme** | pandas, NumPy |
| **ML Framework** | scikit-learn, XGBoost, LightGBM |
| **API** | FastAPI, Uvicorn, Pydantic |
| **Görselleştirme** | Matplotlib, missingno |
| **Test** | pytest |
| **Serileştirme** | joblib |
| **Konfigürasyon** | python-dotenv |

---

## 📂 Veri Seti

**Kaynak:** [UCI Machine Learning Repository — Default of Credit Card Clients](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

| Alan | Açıklama |
|---|---|
| `limit_bal` | Kredi limiti (NT dolar) |
| `sex` | Cinsiyet (1: Erkek, 2: Kadın) |
| `education` | Eğitim seviyesi (1: Lisansüstü, 2: Üniversite, 3: Lise, 4: Diğer) |
| `marriage` | Medeni durum (1: Evli, 2: Bekar, 3: Diğer) |
| `age` | Yaş |
| `pay_0 — pay_6` | Ödeme durumu (Eylül → Nisan) |
| `bill_amt1 — bill_amt6` | Fatura tutarları |
| `pay_amt1 — pay_amt6` | Ödeme tutarları |
| `default_payment_next_month` | **Hedef:** Temerrüt (1) veya değil (0) |

---

## 📜 Lisans

Bu proje özel kullanım amaçlıdır. Tüm hakları saklıdır.
