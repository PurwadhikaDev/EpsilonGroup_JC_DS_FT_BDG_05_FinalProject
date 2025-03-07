# EpsilonGroup_JC_DS_FT_BDG_05_FinalProject

# E-Commerce-Churn-Analysis
Oleh:
-  Azhar Maulana
- Eki Nakia Utami
- Naila Firdusi Putri Fadilah

**INTRODUCTION**

Project ini bertujuan untuk memprediksi churn pada ecommerce menggunakan machine learning. Di dalam project ini kita akan mencari cara cleaning terbaik, model terbaik, dan parameter terbaik agar menghasilkan prediksi yang terbaik yang didasarkan dari fitur-fitur telah disediakan dalam dataset. 
Daftar Isi 
1.	Business Problem Understanding
2.	Data Understanding
3.	Data Cleaning
4.	Data Analisis
5.	Data Pre-Processing
6.	Modelling
7.	Evaluation
8.	Conclusion
9.	Recommendation

**BUSINESS PROBLEM UNDERSTANDING**
1.	Context
- E-Commerce atau perdagangan elektronik adalah proses pembelian dan penjualan barang atau jasa melalui media elektronik, khususnya internet. Dalam e-commerce, transaksi dilakukan secara online tanpa interaksi fisik antara penjual dan pembeli. Keberadaan platform digital mempermudah perusahaan dalam menjangkau pasar yang lebih luas dengan biaya yang lebih efisien. Namun, dengan pesatnya pertumbuhan e-commerce, persaingan semakin ketat, sehingga perusahaan menghadapi tantangan dalam meningkatkan serta mempertahankan pelanggan.

2. Problem Statement
- Banyak perusahaan e-commerce menghadapi kesulitan dalam mengelola churn pelanggan karena kurangnya pendekatan berbasis data dalam strategi retensi. Tanpa analisis yang tepat, **perusahaan sering kali mengalokasikan sumber daya secara tidak efisien untuk mempertahankan pelanggan yang berisiko churn**. Selain itu, tidak adanya model prediksi churn yang akurat dapat menyebabkan keputusan bisnis yang kurang efektif dalam upaya mempertahankan pelanggan. **Oleh karena itu, diperlukan solusi berbasis machine learning untuk mengidentifikasi pelanggan yang berisiko churn dan mengembangkan strategi retensi yang lebih tepat sasaran.**

3. Rumusan Masalah
-	Bagaimana perilaku pelanggan yang berisiko churn dapat dianalisis secara efektif?
- Faktor-faktor apa saja yang paling berpengaruh dalam menentukan apakah pelanggan akan churn?
- Bagaimana model machine learning dapat digunakan untuk memprediksi kemungkinan pelanggan akan churn?
- Bagaimana strategi retensi yang efektif berdasarkan hasil prediksi churn?

4. Goals
- Dengan menganalisis perilaku pelanggan serta memprediksi kemungkinan churn menggunakan algoritma machine learning, diharapkan perusahaan dapat mengembangkan model prediksi churn yang lebih akurat. Model ini diharapkan dapat membantu manajemen dalam meningkatkan efektivitas strategi retensi pelanggan melalui personalisasi penawaran dan efisiensi operasional.
Selain strategi retensi, model yang dikembangkan juga dapat mendukung pengambilan keputusan dalam segmentasi pelanggan, perancangan program loyalitas, serta pengelolaan profitabilitas pendapatan. Dengan mengklasifikasikan pelanggan berdasarkan kemungkinan churn, perusahaan dapat secara proaktif menangani potensi churn, misalnya dengan menawarkan insentif khusus atau meningkatkan kualitas layanan bagi pelanggan berisiko tinggi

5. Analytic Approach
- Kita akan menganalisis data untuk menemukan pola yang membedakan pelanggan yang churn dan tidak. Setelah itu, kita akan membangun model klasifikasi untuk memprediksi kemungkinan pelanggan akan churn, sehingga perusahaan dapat mengambil langkah yang tepat.

6. Target
- Dalam analisis churn, target diklasifikasikan sebagai berikut:
  - 0: Customer tidak churn → Pelanggan masih aktif menggunakan layanan atau produk dalam periode tertentu.
  - 1: Customer churn → Pelanggan berhenti menggunakan layanan, ditandai dengan: 
      - Tidak ada transaksi dalam jangka waktu tertentu.
      - Berhenti berlangganan.
      - Tidak lagi menggunakan aplikasi/platform.
      -	Tidak merespons kampanye pemasaran.

7. Metric Evaluation
- Biaya Churn
   - Customer Acquisition Cost (CAC) → Biaya memperoleh pelanggan baru
   - Customer Retention Cost (CRC) → Biaya mempertahankan pelanggan lama
     
- Kesalahan Prediksi Churn
  - Confusion matrix mengklasifikasikan prediksi churn menjadi:
    - False Positive (FP) → Type 1 Error: Pelanggan tetap diprediksi churn → meningkatkan biaya retensi.
    - False Negative (FN) → Type 2 Error: Pelanggan churn diprediksi tetap → kehilangan pelanggan & meningkatkan CAC.
  - Dalam konteks analisis churn, **Type 2 Error lebih krusial untuk dihindari karena dampaknya yang lebih besar terhadap profitabilitas perusahaan**. Ketidakmampuan mengidentifikasi pelanggan yang berisiko churn dapat menyebabkan hilangnya pelanggan dan menignkatkan CAC. Sementara itu, meskipun **Type 1 Error juga perlu diperhatikan, efeknya cenderung lebih terkait pemborosan sumber daya tetapi tidak langsung mengurangi jumlah pelanggan**.

- Perbandingan CAC vs CRC
  - CAC lebih tinggi: Contoh, jika perusahaan menghabiskan $90 per pelanggan untuk akuisisi.
  - CRC lebih rendah: Contoh, hanya $40 per pelanggan untuk retensi.
  - Strategi retensi lebih efisien dalam jangka panjang dibandingkan akuisisi pelanggan baru.

- Optimasi Churn dengan F2-Score
  - F2-Score menekankan recall, sehingga lebih efektif dalam mengidentifikasi pelanggan yang berisiko churn.
  - Rumus F2-Score:
    - $$F_2= \frac{(1+2^2) \times \text{Precision} \times \text{Recall}}{2^2 \times \text{Precision} \times \text{Recall}}$$
  - F2-Score membantu mengurangi churn dengan menyeimbangkan kesalahan prediksi dan mengoptimalkan strategi retensi.
  - Accuracy tinggi → Model mampu memprediksi churn dan non-churn dengan baik.

  **Data Understanding**
  - Data Set
    - Data Source : https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction
      - Dataset tidak seimbang
      - Setiap baris data mempresentasikan informasi customer E-commerce
   - Secara umum kita bisa melihat data:
     - Dataset ini memiliki 5.630 baris(entri) dan 20 kolom.
     - `PreferredPaymentMode`: Terdapat metode pembayaran seperti Debit Card, UPI, dan CC (Credit Card). 
     - `CityTier`: Istilah ini sering digunakan dalam segmentasi kota di Amerika Serikat, di mana kota dibagi menjadi Tier 1, Tier 2, dan Tier 3.

    Atribut Information

    | Attribute | Data Type, Length | Description |
    | --- | --- | --- |
    | CustomerID | Integer | ID unik pelanggan|
    | Churn | Integer | status apakah pelanggan churn atau tidak  |
    | Tenure | Float | Durasi pelanggan dalam layanan |
    | PreferredLoginDevice | Object | perangkat yang digunakan ketika login |
    | CityTier | Integer | Tingkat kota customer|
    | WarehouseToHome | Float | Jarak antara gudang penyimpanan dan rumah pelanggan |
    | PreferredPaymentMode | Object | Metode pembayaran yang digunakan pelanggan |
    | Gender | Object | Jenis kelamin pelanggan  |
    | HourSpendOnApp | Float | lama waktu customer menggunakan aplikasi |
    | NumberOfDeviceRegistered | Integer | Jumlah perangkat yang terdaftar pada setiap customer|
    | PreferedOrderCat | Object | Kategori produk yang sering dipesan customer satu bulan terakhir |
    | SatisfactionScore | Integer | Skor kepuasan pelanggan terhadap layanan|
    | MaritalStatus | Object | Status pernikahan pelanggan|
    | NumberOfAddress | Integer | Jumlah alamat yang didaftarkan customer |
    | Complain | Integer | Apakah pelanggan mengajukan keluhan dalam satu bulan terakhir|
    | OrderAmountHikeFromlastYear | Float | Persentase peningkatan jumlah pesanan dibandingkan tahun lalu  |
    | CouponUsed | Float | Jumlah kupon/voucher yang telah digunakan customer selama satu bulan terakhir |
    | OrderCount | Float | Jumlah pesanan yang dilakukan selama satu bulan terakhir |
    | DaySinceLastOrder | Float | Jumlah hari sejak pesanan terakhir pelanggan|
    | CashbackAmount | Float | Rata-rata cashback yang diterima dalam satu bulan terakhir |

  **DATA CLEANING**
      Berikut ini hasil pada tahapan data cleaning.
    | Feature | Missing Value | Outlier | Inkonsistensi Data|
    | --- | --- | --- | --- |
    |`Tenure` | Median | Drop value > 31| - |
    | `WarehouseToHome` | Median | Drop value > 40 | - |
    | `HourSpendOnApp` | Median | Outlier di pertahankan | - |
    | `NumberOfDeviceRegistered` | - | Outlier di pertahankan | - |
    | `NumberOfAddress` | - | Outlier di pertahankan | - |
    | `OrderAmountHikeFromlastYear` | Median | Outlier di pertahankan | Mengubah nama kolom 'OrderAmountHikeFromlastYear' menjadi       `OrderAmountHikeFromLastYear `|
    | `CouponUsed` | Median berdasarkan OrderCount | Outlier di pertahankan | - |
    | `OrderCount` | Median berdasarkan CouponUsed | Outlier di pertahankan | - |
    | `DaySinceLastOrder` | Median berdasarkan OrderCount | Drop value > 31 | - |
    | `CashbackAmount` | - | Outlier di pertahankan | - |
    | `PreferredLoginDevice` | - | - | nilai "Phone" akan disesuaikan menjadi "Mobile Phone |  
    | `PreferredPaymentMode` | - | - | Mengubah nilai "COD" menjadi "Cash on Delivery dan nilai "CC" perlu disamakan menjadi "Credit Card  |
    | `PreferredOrderCat` |  |  | Mengubah nama kolom 'PreferedOrderCat' menjadi `PreferredOrderCat ` |

  **DATA ANALYSIS**
  - Menganalisis faktor-faktor yang berkontribusi terhadap churn menggunakan eksplorasi data dan teknik statistik:
    - Faktor Demografi
        - Marital status: Status pernikahan dapat mempengaruhi kebiasaan belanja dan preferensi layanan
        - Gender: Perbedaan preferensi antara pelanggan pria dan wanita dalam memilih layanan
        - City tier: Tingkat kota tempat tinggal pelanggan dapat memengaruhi aksesibilitas dan daya beli

    - Faktor Perilaku Pengguna
        - Tenure: Lama pelanggan menggunakan layanan sebelum churn
        - Preferred login device: Jenis perangkat yang digunakan untuk mengakses layanan
        - Number of device registered: Jumlah perangkat yang terdaftar pada akun pelanggan
        - Number of address: Banyaknya alamat yang terdaftar dalam akun pelanggan
        - HoursSpentOnApp: Waktu yang dihabiskan pelanggan dalam aplikasi

    - Faktor Kepuasan dan Pengalaman Pengguna
        - Satisfaction score: Skor kepuasan pelanggan terhadap layanan
        - Complaints: Jumlah keluhan pelanggan yang telah diajukan
        - Warehouse to home: Waktu pengiriman dari gudang ke alamat pelanggan

    - Faktor Perilaku Transaksi
        - Order count: Jumlah pesanan yang dilakukan pelanggan
        - Order amount like from last year: Perubahan jumlah pembelian dibandingkan tahun sebelumnya
        - Coupon used: Penggunaan kupon oleh pelanggan
    - RFM Analysis (Recency, Frequency, Monetary)
      - Menganalisis pola perilaku pelanggan berdasarkan:
        - Recency (R): Seberapa baru pelanggan melakukan transaksi terakhir
        - Frequency (F): Seberapa sering pelanggan bertransaksi dalam periode tertentu
        - Monetary (M): Total nilai transaksi pelanggan dalam periode tertentu
      - Langkah-langkah:  Menghitung nilai RFM score untuk setiap pelanggan. Melakukan segmen pelanggan berdasarkan RFM score.
      - Menentukan pelanggan dengan risiko churn tinggi berdasarkan low RFM score

  **MACHINE LEARNING**
    - Tahapan Pembuatan Model Machine Learning
      - Import Library: Menggunakan pandas, numpy, matplotlib, seaborn untuk eksplorasi data; sklearn, imblearn untuk preprocessing, modeling, evaluasi; statsmodels untuk analisis statistik.
      - Membaca dan Pembersihan Data: Menghapus duplikasi (CustomerID), menangani missing values (median), dan menghapus outlier (Tenure, WarehouseToHome).
      - Preprocessing Data: Standarisasi kategori (PreferredLoginDevice, PreferredPaymentMode), encoding kategori (OneHotEncoder), scaling numerik (RobustScaler).
      - Seleksi dan Analisis Fitur: Uji Chi-Square, Mann-Whitney U Test, dan Korelasi Spearman untuk pemilihan fitur.
      - Feature Engineering & Train-Test Split: RFM Analysis, pemilihan fitur, split data (80:20, stratifikasi Churn).
      - Penanganan Imbalance Data: Oversampling (SMOTE), Undersampling (NearMiss).

  - Pemilihan Model dan Evaluasi
    - Modeling & Eksperimen: Logistic Regression, KNN, Decision Tree, SVM, Random Forest, XGBoost, dll.
    - Hyperparameter Tuning: GridSearchCV, RandomizedSearchCV.
    Pelatihan Model Terbaik: Model optimal dari tuning dilatih ulang.
    - Evaluasi Model: F2-score, Classification Report, Precision-Recall Curve, Learning Curve.

  - Hasil
    - Berdasarkan hasil model benchmarking dan hyperparameter tuning, model **XGBoost Classification** menunjukkan performa yang baik dalam memprediksi *customer churn* di sebuah e-commerce, dengan nilai F2 Score sebelum tuning 0.90 dan **F2 Score seteleh tuning 0.92**.
   Berikut adalah perbandingan biaya yang dikeluarkan perusahaan untuk menangani customer churn:  
    - **Tanpa Prediksi Machine Learning**: $54,600  
    - **Dengan Prediksi Machine Learning**: $40,490  
    - Dengan menggunakan prediksi machine learning untuk mengidentifikasi pelanggan yang berisiko churn dan mengarahkan strategi retensi, perusahaan dapat menghemat sekitar **$14,110**, atau **25.8%** dibandingkan dengan tidak menggunakan prediksi sama sekali.  

**KESIMPULAN DAN REKOMENDASI**
- Hasil eksperimen model ditujukan untuk memprediksi apakah seorang *customer* akan melakukan churn atau tidak berdasarkan dari pola-pola yang ada dalam data, dengan rincian model sebagai berikut:

    - Metric utama yang digunakan pada percobaan adalah **F2 Score**. Hal ini mempertimbangkan di mana kesalahan dalam memprediksi *customer tidak churn* yang **aktualnya adalah churn**, dianggap lebih merugikan dibandingkan kesalahan saat memprediksi *customer churn* yang **aktualnya tidak churn**. Sehingga recall dianggap dua kali lebih penting daripada precision.
    
    - Berdasarkan hasil model benchmarking dan hyperparameter tuning, model **XGBoost Classification** menunjukkan performa yang baik dalam memprediksi *customer churn* di sebuah e-commerce, dengan nilai F2 Score sebelum tuning 0.90 dan **F2 Score seteleh tuning 0.92**. 

    - Berdasarkan model **XGBoost Classification**, fitur `HourspendOnApp`, `NumberOfDeviceRegistered`, `PreferredOrderCat`, dan `DaySinceLastOrder` merupakan lima fitur yang paling penting dalam memprediksi konsumen churn. 

    - Berdasarkan contoh perhitungan biaya, perusahaan berpotensi mendapatkan kerugian $54,600 dari hilangnya konsumen yang churn tanpa prediksi dan biaya retensi yang tidak efektif.

    - Untuk perhitungan biaya apabila perusahaan menerapkan machine learning, potensi kerugian yang dialami perusahaan sebesar $40,490 dari hilangnya konsumen yang terprediksi churn dan biaya retensi konsumen baik yang efektif maupun yang salah prediksi.

    - Melalui seluruh proses yang telah dilakukan, penerapan *machine learning* dalam memprediksi kemungkinan *churn* dan mengoptimalkan biaya retensi pelanggan memungkinkan perusahaan menghemat hingga 25.8%. Dengan jumlah pelanggan e-commerce yang bisa mencapai jutaan, potensi penghematan ini dapat semakin meningkat jika karakteristik pelanggan masih sesuai dengan cakupan data yang digunakan dalam pemodelan. Hal ini memungkinkan perusahaan untuk mengalokasikan biaya retensi dan akuisisi pelanggan secara lebih efisien serta meminimalkan potensi kerugian.

  **Tools**
  Berikut ini tools yang digunakan.
  - Python
  - Pandas
  - XGBoost
  - Streamlit
  - Pickle (untuk menyimpan dan memuat model)
  - Tableau (untuk dashboard)

