# Laporan Proyek Machine Learning - Ni Putu Sintia Wati

## Domain Proyek
mobil menjadi sarana transportasi yang dimiliki sebagian besar lapisan masyarakat. Terdapat berbagai merek dan jenis mobil dengan spesifik yang bervariasi. Model bisnis yang diterapkan yaitu sebagai tim marketing, yang bertugas untuk menentukan harga produk yang akan dipasarkan. Para konsumen mencari informasi baik dari situs asli atau katalog mobil yang tersedia. 

Untuk efisiensi, kita akan menerapkan teknik predictive modelling untuk memprediksi harga mobil tersebut sehingga dapat memudahkan konsumen dalam menentukan model mobil yang ingin dibelinya. Merek mobil yang dijadikan contoh yakni merek Audi. Dalam penjualannya, harga mobil dipengaruhi oleh fitur khusus, seperti model, tahun registrasi, harga (dalam satuan euro), jenis transmisi,jarak tempuh, jenis bahan bakar, pajak, mpg(penggunaan bahan bakar), dan kapasitas mesin. 

## Business Understanding
### Problem Statements
1. dari semua fitur yang ada, fitur apa yang paling berpengaruh dalam menentukan harga mobil?
2. berapa harga pasar mobil dengan fitur tententu?

### Goals
- Mengetahui fitur yang memiliki hubungan dengan harga mobil.
- Membuat Model ML yang dapat memprediksi harga berdasarkan fitur yang ada

### Solution statements
pada kasus ini, kami mengajukan tiga algoritma machine learning sebagai solusi permasalahan, yaitu KNN Algorithm, Random Forest dan Boosting Algorithm. 

- **KNN**. 
Cara kerja algoritma KNN yaitu mengklasifikasi sekumpulan data dengan menentukan jumlah tetangga terdekat lalu menghitung jarak objek terhadap data latih dengan perhitungan kuadrat jarak eucliden. KNN memiliki kelebihan, salah satunya mudah diimplementasi karena hanya menentukan objek dengan menghitung jarak antar instance. namun salah satu kekurangan dari algoritma ini yaitu perlu menunjukan parameter k dan menentukan nilai k yang sesuai untuk menghindari outlier dalam KNN

- **Random Forest**. 
Cara kerja dari Random Forest yaitu memecah data ke dalam decision tree secara acak. lalu akan dilakukan pemilihan untuk setiap kelas dari data sampel. kemudian mengkombinasikan hasil suara dari setiap kelas untuk diambil yang terbanyak. salah satu kelebihan dari random forest yaitu dapat mengatasi missing value data pada jumlah yang besar. namun salah satu kelemahan dari Random Forest antara lain proses learning yang lambat karena karena bergantung pada parameter yang digunanakan.

- **Boosting Algorithm**. 
Cara kerja dari algoritma boosting yaitu membangun model dari data latih lali membuat model kedua untuk memperbaiki kesalahan di model pertama. penambahan model dilakukan hingga mencapai jumlah maksimum model untuk ditambahkan. 

## Data Understanding
Data yang digunakan untuk projek kali ini yaitu audi car dataset yang diunduh dari kaggle. 
(https://www.kaggle.com/adityadesai13/used-car-dataset-ford-and-mercedes).
Dataset ini memiliki dimensi 10668 x 9 dengan variable-variable sebagai berikut:
- year : tahun registrasi 
- price : harga dalam £
- transmission : tipe gearbox
- mileage : jarak tempuh
- fuelType : tipe bahan bakar mesin
- tax : pajak
- mpg : miles per gallon
- engineSize : ukuran mesin

sebelum masuk ke tahap selanjutnya, terdapat tahapan yang harus dilakukan diantaranya Data Loading dan Data Analysis. 

### Data Loading

Sebelum masuk ke data loading, import library yang akan digunakan seperti **pandas** untuk membaca data, **numpy** untuk numeratical data serta **matplotlib** dan **seaborn** untuk visualisasi data. 

Data Loading yaitu memuat data yang akan diolah pada proses Modeling nanti. dataset yang digunakan yaitu audi.csv. Tambahkan fungsi **.head()** untuk nenampilkan data teratas. hasilnya seperti dibawah ini. 

Untuk menentukan ukuran dari dataset yang akan digunakan, dapat digunakan fungsi **.shape**. hasilnya akan berupa bilangan desimal yang dimana bagian depan disebut baris sedangkan dibelakang koma disebut jumlah kolom.

### Data Analysis

Exploratory Data Analysis (EDA) merupakan proses pengenalan data untuk menganalisis karakteristik, menemukan pola, anomali dan memeriksa asumsi data. teknik tersebut juga menggunakan bantuan statistik dan visualisasi grafis. 

**Deskripsi Variable** \
Pada pembuka Data Understanding, telah dijelaskan variable yang akan digunakan. selanjutnya akan kita cek informasi pada dataset dengan beberapa fungsi **.info**. 

 ![hasil info](https://github.com/sintiasnn/car-price-prediction/blob/main/picture-01.jpg?raw=true)
 
Hasilnya terdapat tipe data yang digunakan, diantaranya object (model, transsmission, fuelType), int(year, price, mileage, tax), dan float(mpg dan engineSize). Jika dikelompokan, maka:

 - data kategorial (model, transmission, dan fuelType)
 - data numerial (year, price, mileage, tax, mpg dan engineSize)

**Penanganan Missing Value** \
untuk memastikan ada tidaknya missing value, kita dapat melakukan deskripsi statistik dengan penggunakan fungsi **describe()**.

![hasil deskripsi](https://github.com/sintiasnn/car-price-prediction/blob/main/picture-02.jpg?raw=true)

Berdasarkan output diatas, terdapat dua variable dengan nilai minimum 0 yaitu pada variabel tax dan engineSize. Hal tersebut terindikasi adanya missing value. Selanjutnya kita cek pada kedua variabel jumlah data yang memiliki nilai 0 dengan bantuan fungsi **.sum**. 

![jumlah nilai 0](https://github.com/sintiasnn/car-price-prediction/blob/main/picture-03.jpg?raw=true)

Untuk memastikan kembali, kita dapat mengetahui data yang memiliki nilai 0 tersebut dengan fungsi loc[] dengan kondisi nilaiVariable==0.

![nilai 0 pada tax](https://github.com/sintiasnn/car-price-prediction/blob/main/picture-04.jpg?raw=true)

![nilai 0 pada engineSize](https://github.com/sintiasnn/car-price-prediction/blob/main/picture-05.jpg?raw=true)
![nilai 0 pada engineSize](https://github.com/sintiasnn/car-price-prediction/blob/main/picture-06.jpg?raw=true)

Jika telah ditelusuri, kita dapat menghapus data tersebut lalu melakukan kembali deskripsi statistik dengan fungsi **describe()**. 

![hasil deskripsi kedua](https://github.com/sintiasnn/car-price-prediction/blob/main/picture-07.jpg?raw=true)

Hasilnya nilai nimimum berubah menjadi 1 dan tidak terdapat missing value pada dataset tersebut. 

**Unvariate Analysis** \
Unvariate analysis merupakan proses untuk mengeksplorasi dan menjelaskan setiap variabel dalam kumpulan data secara terpisah. 

Sebelum melakukan unvariate analysis, terlebih dahulu kita membagi semua fitur menjadi dua kelompok fitur yaitu fitur numerik dan fitur kategorikal. 

Kemudian dilanjutkan dengan menganalisis fitur kategorial. Diawali dengan fitur model

![total fitur model](https://github.com/sintiasnn/car-price-prediction/blob/main/picture-08.jpg?raw=true)
![bar chart fitur](https://github.com/sintiasnn/car-price-prediction/blob/main/picture-09.jpg?raw=true)

fitur transmission \

![bar chart transmission](https://github.com/sintiasnn/car-price-prediction/blob/main/picture-10.jpg?raw=true)

fitur fuelType \

![bar chart fuelType](https://github.com/sintiasnn/car-price-prediction/blob/main/picture-11.jpg?raw=true)

Selanjutnya, lakukan analisis pada fitur numerikal. \

![bar chart fitur numerikal](https://github.com/sintiasnn/car-price-prediction/blob/main/picture-12.jpg?raw=true)

**Multivariate Analysis** \
Multivariate analysis merupakan proses eksplorasi yang melibatkan banyak (dua atau lebih) variabel pada data. Dalam hal ini kita akan menganalisis keterkaitan/korelasi antara fitur target (price) dengan fitur lainnya. 

Fitur kategorikal \

![bar chart fitur kategorikal](https://github.com/sintiasnn/car-price-prediction/blob/main/picture-13.jpg?raw=true)
![bar chart fitur](https://github.com/sintiasnn/car-price-prediction/blob/main/picture-14.jpg?raw=true)
![bar chart fitur](https://github.com/sintiasnn/car-price-prediction/blob/main/picture-15.jpg?raw=true)

Hasil yang didapat : 

- untuk fitur 'model', sebagian besar model memiliki rata-rata harga yang bervariasi. kirasan 20000-40000 euro. namun ada juga model yang memiliki harga yang cukup tinggi yaitu model R8. hal ini menunjukan bahwa model memiliki pengaruh yang cukup kecil terhadap harga.

- untuk fitur 'transmission', automatic transmission dan semi auto memiliki harga yang cukup tinggi.

- untuk FuelType, hybrid memiliki harga yang tinggi, sedangkan pada petrol dan diesel memiliki selisih harga yang tidak beda jauh.

- dapat disimpulkan bahwa ketiga fitur kategori mempengaruhi rata-rata harga. 

Fitur numerikal \
menggunakan **pairplot()** serta menggunakan **corr()** untuk menghitung korelasi antar fitur target dengan fitur numerik.

![bar chart fitur](https://github.com/sintiasnn/car-price-prediction/blob/main/picture-16.jpg?raw=true)

Hasil nya :
Pada kasus diatas, terjadi relasi antar fitur numerik dengan fitur target(price). pada grafik year dan engineSize terjadi korelasi positif, kemudian pada grafik mileage dan mpg terjadi korelasi negatif. sedangkan pada grafik tax tidak memiliki korelasi.

Untuk melihat hasil korelasi secara numerik, dapat menggunakan corr()

![correlation matrix](https://github.com/sintiasnn/car-price-prediction/blob/main/picture-17.jpg?raw=true)

Perhatikan pada baris price karena kita akan mengamati korelasi antara fitur price dengan fitur numerik lainnya.  pada fitur year dan engineSize memiliki nilai korelasi mendekati 1, lalu pada fitur mileage dan mpg memiliki nilai korelasi mendekati -1 sedangkan fitur tax memiliki nilai korelasi mendekati 0 (korelasi lemah). 

Oleh karena itu, fitur tax tidak berkorelasi dengan fitur price sehingga fitur tax dapat di drop. 

![drop result](https://github.com/sintiasnn/car-price-prediction/blob/main/picture-18.jpg?raw=true)

## Data Preparation
- Encoding Categorical Feature \
Mengubah fitur kategori menjadi fitur numerik dengan teknik one-hot-encoding. Hal ini dilakukan karena mesin hanya dapat memproses data yang bernilai numerik khusunya bernilai 0 dan 1. kita dapat menggunakan library **sklearn** dan library **pandas** dengan fungsi get_dummies()

![encoding result](https://github.com/sintiasnn/car-price-prediction/blob/main/picture-19.jpg?raw=true)


- train test split \
Teknik digunakan untuk membagi dataset menjadi data latih dan data uji. dalam kasus ini, pembagiannya 90:10 yakni 90% untuk data latih dan 10% untuk data uji.


Cek jumlah sampel ada masing-masing bagian.
![hasil split](https://github.com/sintiasnn/car-price-prediction/blob/main/picture-19a.jpg?raw=true)

- standaritation \
Tujuan standarisasi adalah agar performa algoritma Machine Learning lebih baik serta dapat mempermudah data diolah oleh algoritma. Standarisasi dilakukan dengan mengurangi mean kemudian membaginya dengan milai standar deviasi untuk menggeser distribusi. Kita akan menggunakan library scikitlearn dengan teknik **StandarScaler**.

Kita terapkan standarisasi pada data latih, sedangkan pada tahap evaluasi diterapkan standarisasi pada data uji.

![standarisasi](https://github.com/sintiasnn/car-price-prediction/blob/main/picture-20.jpg?raw=true)

Hasil standarisasi: 

![standarisasi](https://github.com/sintiasnn/car-price-prediction/blob/main/picture-21.jpg?raw=true)

## Modeling
Model yang akan digunakan dalam menyelesaikan permasalahan prediksi harga mobil ini, diantaranya KNN, Random Forest, dan Boosting. 

Sebelum masuk ke tahapan modeling dengan ketiga model diatas, terlebih dahulu siapkan dataframe untuk analisa model nantinya. 


* K-Nearest Neighbor \

cara kerja :
      a. tentukan parameter k \
      b. Hitung jarak antara data yang akan dievaluasi dengan semua pelatihan \
      c. urutkan jarak yang berbentuk menaik \
      d. tentukan jarak trdekat sampai urutan k \
      e. pasangkan kelas yang bersesuaian \
      f. cari jumlah kelas dari tetangga yang terdekat dan tetapkan kelas tersebut sebagai kelas data yang akan dievaluasi \

penerapan pada python :
```python
from sklearn.neighbors import KNeighborsRegressor
 
knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_train)
```

* Random Forest \

cara kerja : 
Random forest  adalah kombinasi dari  masing – masing tree yang baik kemudian dikombinasikan  ke dalam satu model. Random Forest bergantung pada sebuah nilai vector random dengan distribusi yang sama pada semua pohon yang masing masing decision tree memiliki kedalaman yang maksimal. 

penerapan pada python:
```python
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
 

RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
RF.fit(X_train, y_train)
 
models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train) 
```

* Boosting\
cara kerja : 
Algoritma boosting bekerja dengan membangun model dari data latih. Kemudian ia membuat model kedua yang bertugas memperbaiki kesalahan dari model pertama. Model ditambahkan sampai data latih terprediksi dengan baik atau telah mencapai jumlah maksimum model untuk ditambahkan. 

penerapan pada python : 
```python
from sklearn.ensemble import AdaBoostRegressor
 
boosting = AdaBoostRegressor(n_estimators=50, learning_rate=0.05, random_state=55)                             
boosting.fit(X_train, y_train)
models.loc['train_mse','Boosting'] = mean_squared_error(y_pred=boosting.predict(X_train), y_true=y_train)
```

## Evaluation
Karena masalah yang diselesaikan termasuk regresi, maka metrik yang akan kita gunakan pada prediksi ini adalah MSE atau Mean Squared Error yang menghitung selisih rata-rata nilai sebenarnya dengan nilai prediksi. 

Sebelum menghitung nilai MSE, kita akan lakukan scaling fitur numerik pada data uji. Tujuannya untuk menghindari kebocoraan data. Selanjutnya dilakukan evaluasi terhadap 3 model algoritma machine learning dengan metrik MSE. 

![hasil evaluasi](https://github.com/sintiasnn/car-price-prediction/blob/main/picture-22.jpg?raw=true)

Visualisasikan ke dalam bentuk bar chart.
![bar chart metrik](https://github.com/sintiasnn/car-price-prediction/blob/main/picture-23.jpg?raw=true)

Untuk mengujinya, kita dapat membuat prediksi dengan beberapa harga dari dataset. Hasil Uji akan seperti dibawah. 

![hasil uji](https://github.com/sintiasnn/car-price-prediction/blob/main/picture-24.jpg?raw=true)

Hasil uji menunjukkan bahwa prediksi dengan Random Forest memberikan hasil yang mendekati. 
