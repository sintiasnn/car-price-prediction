# Laporan Proyek Machine Learning - Ni Putu Sintia Wati

## Domain Proyek
mobil sudah menjadi sarana transportasi yang sebagian besar orang telah memilikinya. Terdapat berbagai merek dan jenis mobil dengan spesifik yang memadai. Model bisnis yang diterapkan yaitu sebagai tim marketing, yang bertugas untuk menentukan harga produk yang akan dipasarkan. Para konsumen mencari informasi baik dari situs asli atau katalog mobil yang tersedia. 

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

sebelum masuk ke tahap selanjutnya, terdapat tahapan yang harus dilakukan diantaranya Data Loading dan Data Analysis. 

### Data Loading

Data Loading yaitu memuat data yang akan diolah pada proses Modeling nanti. 

   memuat dataset ke notebook
   ```python
    data = 'C:/Users/HP/Downloads/car-price-prediction/audi.csv'
    cars = pd.read_csv(data)
    cars.head(2)
   ```
   
### Data Analysis

Exploratory Data Analysis (EDA) merupakan proses pengenalan data untuk menganalisis karakteristik, menemukan pola, anomali dan memeriksa asumsi data. teknik tersebut juga menggunakan bantuan statistikdan visualisasi grafis. 

- **Deskripsi Variable**.
Pada pembuka Data Understanding, telah dijelaskan variable yang akan digunakan. selanjutnya akan kita cek informasi pada dataset dengan beberapa perintah dibawah ini. 


  cek informasi pada dataset:
  ```python
  cars.info()
  ```
 ![hasil info](https://github.com/sintiasnn/car-price-prediction/blob/main/picture-01.jpg?raw=true)
 
Hasilnya terdapat tipe data yang digunakan, diantaranya object (model, transsmission, fuelType), int(year, price, mileage, tax), dan float(mpg dan engineSize). Jika dikelompokan, maka:

 1. data kategorial (model, transmission, dan fuelType)
 2. data numerial (year, price, mileage, tax, mpg dan engineSize)

- **Penanganan Missing Value**.
untuk memastikan ada tidaknya missing value, kita dapat melakukan deskripsi statistik dengan penggunakan fungsi describe()

```python
cars.info()
```

![hasil deskripsi](https://github.com/sintiasnn/car-price-prediction/blob/main/picture-02.jpg?raw=true)

Berdasarkan output diatas, terdapat dua variable dengan nilai minimum 0 yaitu pada variabel tax dan engineSize. Hal tersebut terindikasi adanya missing value. Selanjutnya kita cek pada kedua variabel jumlah data yang memiliki nilai 0. 

```python
tax = (cars.tax == 0).sum()
engineSize = (cars.engineSize == 0).sum()
print("nilai 0 pada tax : ",tax)
print("nilai 0 pada engineSize : ",engineSize)
```
![jumlah nilai 0](https://github.com/sintiasnn/car-price-prediction/blob/main/picture-03.jpg?raw=true)

untuk memastikan kembali, kita dapat mengetahui data yang memiliki nilai 0 tersebut dengan fungsi loc[] dengan kondisi nilaiVariable==0

```python
cars.loc[(cars['tax']==0)]
```
![nilai 0 pada tax](https://github.com/sintiasnn/car-price-prediction/blob/main/picture-04.jpg?raw=true)

```python
cars.loc[(cars['engineSize']==0)]
```
![nilai 0 pada engineSize](https://github.com/sintiasnn/car-price-prediction/blob/main/picture-05.jpg?raw=true)
![nilai 0 pada engineSize](https://github.com/sintiasnn/car-price-prediction/blob/main/picture-06.jpg?raw=true)

Jika telah ditelusuri, kita dapat menghapus data tersebut lalu melakukan kembali deskripsi statistik dengan fungsi describe()

```python
cars = cars.loc[(cars[['tax']] >= 1).all(axis=1)]
cars = cars.loc[(cars[['engineSize']] >= 1).all(axis=1)]
```

```python
cars.describe()
```
![hasil deskripsi kedua](https://github.com/sintiasnn/car-price-prediction/blob/main/picture-07.jpg?raw=true)

hasilnya nilai nimimum berubah menjadi 1 dan tidak terdapat missing value pada dataset tersebut. 

- **Unvariate Analysis**.
Unvariate analysis merupakan proses untuk mengeksplorasi dan menjelaskan setiap variabel dalam kumpulan data secara terpisah. 

sebelum melakukan unvariate analysis, terlebih dahulu kita membagi semua fitur menjadi dua kelompok fitur yaitu fitur numerik dan fitur kategorikal. 

```python
numerical_features = ['price', 'year', 'mileage', 'tax', 'mpg', 'engineSize']
categorical_features = ['model', 'transmission', 'fuelType']
```
lalu dimulai dengan menganalisis fitur kategorial. diawali dengan fitur model

```python
feature = categorical_features[0]
count = cars[feature].value_counts()
percent = 100*cars[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);
```
![total fitur model](https://github.com/sintiasnn/car-price-prediction/blob/main/picture-08.jpg?raw=true)
![bar chart fitur](https://github.com/sintiasnn/car-price-prediction/blob/main/picture-09.jpg?raw=true)

fitur transmission
```python
feature = categorical_features[1]
count = cars[feature].value_counts()
percent = 100*cars[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);
```
![bar chart fitur](https://github.com/sintiasnn/car-price-prediction/blob/main/picture-10.jpg?raw=true)

fitur fuelType
```python
feature = categorical_features[2]
count = cars[feature].value_counts()
percent = 100*cars[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);
```
![bar chart fitur](https://github.com/sintiasnn/car-price-prediction/blob/main/picture-11.jpg?raw=true)

Selanjutnya, pada fitur numerikal
```python
cars.hist(bins=50, figsize=(20,15))
plt.show()
```
![bar chart fitur](https://github.com/sintiasnn/car-price-prediction/blob/main/picture-12.jpg?raw=true)

- **Multivariate Analysis**.
Multivariate analysis merupakan proses eksplorasi yang melibatkan banyak (dua atau lebih) variabel pada data. dalam hal ini kita akan menganalisis keterkaitan/korelasi antara fitur target (price) dengan fitur lainnya. 

fitur kategorikal.

```python
cat_features = cars.select_dtypes(include='object').columns.to_list()
 
for col in cat_features:
  sns.catplot(x=col, y="price", kind="bar", dodge=False, height = 4, aspect = 3,  data=cars, palette="Set3")
  plt.title("Rata-rata 'price' Relatif terhadap - {}".format(col))
```
![bar chart fitur](https://github.com/sintiasnn/car-price-prediction/blob/main/picture-13.jpg?raw=true)
![bar chart fitur](https://github.com/sintiasnn/car-price-prediction/blob/main/picture-14.jpg?raw=true)
![bar chart fitur](https://github.com/sintiasnn/car-price-prediction/blob/main/picture-15.jpg?raw=true)


fitur numerikal.
```python
sns.pairplot(cars, diag_kind = 'kde')
```
![bar chart fitur](https://github.com/sintiasnn/car-price-prediction/blob/main/picture-16.jpg?raw=true)


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




