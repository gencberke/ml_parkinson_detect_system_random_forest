# Parkinson Hastalığı Tespiti — Random Forest ile Sınıflandırma

Oxford Parkinson's Disease Detection Dataset (UCI) üzerinde **Random Forest (Rastgele Orman)** algoritması kullanılarak Parkinson hastalığı tespiti yapılan bir makine öğrenmesi çalışmasıdır.

> **Ders:** Makine Öğrenmesi
> **Üniversite:** İstanbul Topkapı Üniversitesi — Yazılım Mühendisliği
> **Hazırlayan:** Berke Genç — `23040301058`

> **Not:** Bu repo, grup projesi kapsamındaki **bireysel Random Forest katkımı** içerir. Grup projesinin tamamında dört farklı algoritma (Random Forest, KNN, Decision Tree, Naive Bayes) karşılaştırılmıştır; burada yalnızca benim sorumlu olduğum Random Forest bölümü yer almaktadır.

---

## Proje Özeti

Parkinson hastalığı, beyindeki dopamin sistemini etkileyen nörodejeneratif bir hastalıktır ve ses üretimi üzerinde belirgin etkiler bırakır. Bu çalışma, **hastaların ses kayıtlarından türetilen akustik ölçümleri** kullanarak ikili sınıflandırma (sağlıklı / Parkinson) yapan bir Random Forest modelinin kurulmasını, optimize edilmesini ve değerlendirilmesini amaçlar.

### Problem Türü
- **İkili Sınıflandırma (Binary Classification)**
- Hedef değişken: `status` → `1 = Parkinson`, `0 = Sağlıklı`

### Veri Seti
- **Kaynak:** [UCI Machine Learning Repository — Parkinson's Dataset](https://archive.ics.uci.edu/dataset/174/parkinsons)
- **Gözlem sayısı:** 195 ses kaydı (31 bireyden)
- **Özellik sayısı:** 22 sayısal akustik ölçüm (MDVP:Fo, Jitter, Shimmer, HNR, RPDE, DFA, spread1, spread2, D2, PPE vb.)
- **Sınıf dağılımı:** 147 Parkinson (%75) · 48 Sağlıklı (%25) → **dengesiz veri seti**

---

## Klasör Yapısı

```
ML_Parkinson/
├── main.ipynb                  # Random Forest notebook (bu çalışma)
├── README.md                   # Bu dosya
└── data/
    ├── parkinsons_standardized.csv      # Z-score standardize edilmiş ana veri (195 × 24)
    ├── standardization_parameters.csv   # Her özellik için ortalama ve scale (std) değerleri
    ├── feature_summary_statistics.csv   # Orijinal verinin özet istatistikleri
    └── parkinsons.names                 # UCI veri seti açıklama dosyası
```

---

## Notebook İçeriği

`main.ipynb` şu bölümlerden oluşur:

| Bölüm | Başlık | İçerik |
|---|---|---|
| **1** | Veri Seti Tanıtımı | Problem tipi, hedef değişken, akustik özelliklerin klinik önemi |
| **2** | Kütüphaneler ve Veri Okuma | Import'lar, `parkinsons_standardized.csv` yüklemesi |
| **3** | İstatistiksel Analiz | Orijinal ve standardize veri üzerinde `describe()` çıktıları |
| **4** | Veri Hazırlama | Eksik değer, sınıf dengesizliği, standardizasyon, train-test split (80/20 stratified, `random_state=42`) |
| **5** | Random Forest Modelleme | Baseline model + GridSearchCV ile hiperparametre optimizasyonu + Baseline/Optimize karşılaştırması |
| **6** | Değerlendirme ve Analiz | Confusion matrix, performans metrikleri, feature importance, `n_estimators` etki analizi |
| **7** | Sonuçların Yorumlanması | Klinik değerlendirme, recall önemi, genel sonuç |

---

## Kullanılan Kütüphaneler

```python
numpy
pandas
matplotlib
seaborn
scikit-learn   # model_selection, ensemble, metrics
```

---

## Nasıl Çalıştırılır?

1. Python 3.9+ ve yukarıdaki kütüphaneleri yükleyin:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn jupyter
   ```
2. Depoyu klonlayın ve dizine girin:
   ```bash
   git clone <repo-url>
   cd ML_Parkinson
   ```
3. Jupyter başlatın ve notebook'u açın:
   ```bash
   jupyter notebook main.ipynb
   ```
4. `Kernel → Restart & Run All` ile notebook'u baştan sona çalıştırın.

> **Not:** `main.ipynb` içindeki tüm veri yolları `data/` alt klasörüne görelidir. Notebook'u proje kök dizininden açtığınızdan emin olun.

---

## Modelleme Yaklaşımı

### Baseline Random Forest
- Varsayılan hiperparametreler ile kurulmuş referans model
- `random_state=42`, `n_estimators=100`

### Hiperparametre Optimizasyonu (GridSearchCV)
- 5-fold cross-validation
- Aranan parametreler: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`
- Seçim kriteri: **F1-score** (dengesiz veri seti için uygun metrik)

### Ek Analizler
- **Feature importance** görselleştirmesi (en etkili akustik özellikler)
- **`n_estimators` etki grafiği** (ağaç sayısının doğruluk üzerindeki etkisi)
- **Performans bar chart** (accuracy, precision, recall, F1-score karşılaştırması)

---

## Değerlendirme Metrikleri

Veri dengesiz olduğundan (%75 Parkinson) tek başına **accuracy** yeterli değildir. Bu nedenle aşağıdaki dört metrik birlikte raporlanmıştır:

| Metrik | Anlamı | Neden Önemli? |
|---|---|---|
| **Accuracy** | Doğru tahminlerin oranı | Genel başarı göstergesi |
| **Precision** | "Parkinson" denenlerin ne kadarı gerçekten Parkinson | Yanlış alarm oranını ölçer |
| **Recall** | Parkinson hastalarının ne kadarı yakalandı | **Klinik açıdan en kritik metrik** — hasta kaçırma |
| **F1-Score** | Precision ve recall'un harmonik ortalaması | Dengesiz veride en bilgilendirici |

**Sağlık uygulamasında recall önceliklidir:** Bir Parkinson hastasını kaçırmak (False Negative), hastanın tedaviden mahrum kalmasına yol açar.

---

## Veri Ön İşleme Notu

Veri seti, ortak aşamada **Z-score standardizasyonu** uygulanarak hazırlanmıştır:

$$x' = \frac{x - \mu}{\sigma}$$

- Random Forest için ölçeklendirme **zorunlu değildir** (ağaç tabanlı modeller ölçek bağımsızdır), ancak pipeline tutarlılığı açısından standardize veri kullanılmıştır.
- Standardizasyon parametreleri `data/standardization_parameters.csv` dosyasında saklanmaktadır; bu sayede yeni veriler aynı dönüşümle işlenebilir.

---

## Lisans

Bu proje akademik amaçlı hazırlanmıştır.
Veri seti: *Little, M.A., McSharry, P.E., Hunter, E.J., Spielman, J. and Ramig, L.O. (2008). 'Suitability of dysphonia measurements for telemonitoring of Parkinson's disease.' IEEE Transactions on Biomedical Engineering.* UCI Machine Learning Repository.
