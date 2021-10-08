# Laporan Proyek Machine Learning - Ni Putu Sintia Wati

## Domain Proyek
mobil sudah menjadi sara transportasi yang sebagian besar orang telah memilikinya. Terdapat berbagai merek dan jenis mobil dengan spesifik yang memadai. Model bisnis yang diterapkan yaitu sebagai tim marketing, yang bertugas untuk menentukan harga produk yang akan dipasarkan. Para konsumen mencari informasi baik dari situs asli atau katalog mobil yang tersedia. Untuk efisiensi, kita akan menerapkan teknik predictive modelling untuk memprediksi harga mobil tersebut sehingga dapat memudahkan konsumen dalam menentukan model mobil yang ingin dibelinya. Merek mobil yang dijadikan contoh yakni merek Audi.

Dalam penjualannya, harga mobil dipengaruhi oleh fitur khusus, seperti model, tahun registrasi, harga (dalam satuan euro), jenis transmisi,jarak tempuh, jenis bahan bakar, pajak, mpg(penggunaan bahan bakar), dan kapasitas mesin. 

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
cara kerja algoritma KNN yaitu mengklasifikasi sekumpulan data dengan menentukan jumlah tetangga terdekat lalu menghitung jarak objek terhadap data latih dengan perhitungan kuadrat jarak eucliden. KNN memiliki kelebihan, salah satunya mudah diimplementasi karena hanya menentukan objek dengan menghitung jarak antar instance. namun salah satu kekurangan dari algoritma ini yaitu perlu menunjukan parameter k dan menentukan nilai k yang sesuai untuk menghindari outlier dalam KNN

- **Random Forest**. 
cara kerja dari Random Forest yaitu memecah data ke dalam decision tree secara acak. lalu akan dilakukan pemilihan untuk setiap kelas dari data sampel. kemudian mengkombinasikan hasil suara dari setiap kelas untuk diambil yang terbanyak. salah satu kelebihan dari random forest yaitu dapat mengatasi missing value data pada jumlah yang besar. namun salah satu kelemahan dari Random Forest antara lain proses learning yang lambat karena karena bergantung pada parameter yang digunanakan.

- **Boosting Algorithm**. 
cara kerja dari algoritma boosting yaitu membangun model dari data latih lali membuat model kedua untuk memperbaiki kesalahan di model pertama. penambahan model dilakukan hingga mencapai jumlah maksimum model untuk ditambahkan. 

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

## Data Preparation
- Encoding Categorical Feature
teknik ini digunakan untuk mengubah niai pada fitur kategori menjadi tipe numerik agar dapat di proses. perintah yang digunakan yaitu get_dummies(). 

- train test split
teknik digunakan untuk membagi dataset menjadi data latih dan data uji. dalam kasus ini, pembagiannya 90:10 

- standarisasi 
dengan melakukan pengurangan mean kemudian membaginya dengan nilai standar deviasi. 

## Modeling


## Evaluation
Bagian ini menjelaskan mengenai metrik evaluasi yang digunakan untuk mengukur kinerja model. Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan dan bagaimana formulanya
- Kelebihan dan kekurangan metrik
- Bagaimana cara menerapkannya ke dalam kode.



**---Ini adalah bagian akhir laporan---**




