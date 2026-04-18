# Parkinson Hastalığı Tespiti — Makine Öğrenmesi Grup Projesi

Oxford Parkinson's Disease Detection Dataset (UCI) üzerinde **dört farklı sınıflandırma algoritması** ile Parkinson hastalığı tespiti yapılan karşılaştırmalı bir makine öğrenmesi çalışmasıdır.

> **Ders:** Makine Öğrenmesi  
> **Üniversite:** İstanbul Topkapı Üniversitesi — Yazılım Mühendisliği

---

## Ekip Üyeleri ve Katkıları

| Üye | Öğrenci No | Sorumlu Algoritma | Bölüm |
|:---|:---:|:---|:---:|
| **Berke Genç** | `23040301058` | Random Forest (Rastgele Orman) | Bölüm 5 |
| **Mert Ali Kızılkaya** | `23040301051` | K-Nearest Neighbors (KNN) | Bölüm 6 |
| **Yiğit Ali Çolakoğlu** | `23040301061` | Decision Tree (Karar Ağacı) | Bölüm 7 |
| **Efe Yaman Sir** | `23040301057` | Gaussian Naive Bayes | Bölüm 8 |

---

## Proje Özeti

Parkinson hastalığı, beyindeki dopamin sistemini etkileyen nörodejeneratif bir hastalıktır ve ses üretimi üzerinde belirgin etkiler bırakır. Bu proje, **hastaların ses kayıtlarından türetilen akustik ölçümleri** kullanarak ikili sınıflandırma (sağlıklı / Parkinson) yapan dört farklı modelin performansını karşılaştırmayı amaçlamaktadır.

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
├── main.ipynb                  # Birleştirilmiş grup notebook'u (dört algoritma)
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
| **4** | Veri Hazırlama | Eksik değer, sınıf dengesizliği, standardizasyon, **ortak train-test split** (80/20 stratified, `random_state=42`) |
| **5** | **Random Forest** (Berke Genç) | Baseline + GridSearchCV + feature importance + n_estimators analizi |
| **6** | **K-Nearest Neighbors** (Mert Ali Kızılkaya) | K=1..20 taraması, K=3 final model |
| **7** | **Decision Tree** (Yiğit Ali Çolakoğlu) | `max_depth` overfitting analizi + ağaç görseli |
| **8** | **Gaussian Naive Bayes** (Efe Yaman Sir) | GaussianNB baseline model |
| **9** | Model Karşılaştırma | 4 model × 4 metrik karşılaştırma tablosu + gruplu bar chart |
| **10** | Sonuçların Yorumlanması | Klinik değerlendirme, recall önemi, genel sonuç |

**Önemli tasarım kararı:** Dört model aynı eğitim ve test setini kullanır. Bu sayede karşılaştırma **adil** olur — performans farkları yalnızca algoritma seçiminden kaynaklanır.

---

## Kullanılan Kütüphaneler

```python
numpy
pandas
matplotlib
seaborn
scikit-learn   # model_selection, ensemble, neighbors, tree, naive_bayes, metrics
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

- KNN ve Naive Bayes gibi mesafe/olasılık tabanlı modeller için **zorunlu**
- Random Forest ve Decision Tree için **zorunlu değil** (ağaç tabanlı modeller ölçek bağımsızdır) ama pipeline tutarlılığı açısından uygulanmıştır

Standardizasyon parametreleri `data/standardization_parameters.csv` dosyasında saklanmaktadır; bu sayede yeni veriler aynı dönüşümle işlenebilir.

---

## Lisans

Bu proje akademik amaçlı hazırlanmıştır.  
Veri seti: *Little, M.A., McSharry, P.E., Hunter, E.J., Spielman, J. and Ramig, L.O. (2008). 'Suitability of dysphonia measurements for telemonitoring of Parkinson's disease.' IEEE Transactions on Biomedical Engineering.* UCI Machine Learning Repository.
