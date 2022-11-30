import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from numpy import array
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from collections import OrderedDict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import altair as alt
from sklearn.utils.validation import joblib

st.title("PENAMBANGAN DATA")
st.write("##### Nama  : R.Bella Aprilia Damayanti ")
st.write("##### Nim   : 200411100082 ")
st.write("##### Kelas : Penambangan Data ")
data_set_description, upload_data, preporcessing, modeling, implementation = st.tabs(["Data Set Description", "Upload Data", "Prepocessing", "Modeling", "Implementation"])

with data_set_description:
    st.write("""# Data Set Description """)
    st.write("###### Data Set Ini Adalah : Body Fat Prediction Dataset (Kumpulan Data Prediksi Lemak Tubuh) ")
    st.write("###### Sumber Data Set dari Kaggle : https://www.kaggle.com/datasets/adityakadiwal/water-potability")
    st.write("""###### Penjelasan setiap kolom : """)
    st.write("""1.ph value (nilai pH):PH merupakan parameter penting dalam mengevaluasi keseimbangan asam-basa air. Ini juga merupakan indikator kondisi status air asam atau basa. WHO telah merekomendasikan batas pH maksimum yang diizinkan dari 6,5 hingga 8,5. Rentang investigasi saat ini adalah 6,52-6,83 yang berada dalam kisaran standar WHO. (pH 1. air (0 hingga 14))""")
    st.write("""2.Hardness (kekerasan): Kekerasan terutama disebabkan oleh garam kalsium dan magnesium. Garam-garam ini larut dari endapan geologis yang dilalui air. Lamanya waktu air bersentuhan dengan bahan penghasil kesadahan membantu menentukan berapa banyak kesadahan yang ada dalam air baku. Kesadahan awalnya didefinisikan sebagai kapasitas air untuk mengendapkan sabun yang disebabkan oleh Kalsium dan Magnesium. (Kapasitas air untuk mengendapkan sabun dalam mg/L).""")
    st.write("""3.Solids (padatan): Air memiliki kemampuan untuk melarutkan berbagai mineral atau garam anorganik dan beberapa organik seperti kalium, kalsium, natrium, bikarbonat, klorida, magnesium, sulfat, dll. Mineral ini menghasilkan rasa yang tidak diinginkan dan warna encer seperti air. Ini adalah parameter penting untuk penggunaan air. Air dengan nilai TDS tinggi menunjukkan bahwa air sangat termineralisasi. Batas yang diinginkan untuk TDS adalah 500 mg/l dan batas maksimum adalah 1000 mg/l yang ditentukan untuk tujuan minum. (Total dissolved solids in ppm)""")
    st.write("""4.Chloramines (kloramin): Klorin dan kloramin adalah disinfektan utama yang digunakan dalam sistem air publik. Chloramines paling sering terbentuk ketika amonia ditambahkan ke klorin untuk mengolah air minum. Tingkat klorin hingga 4 miligram per liter (mg/L atau 4 bagian per juta (ppm)) dianggap aman dalam air minum. (Jumlah kloramin dalam ppm).""")
    st.write("""5.Sulfate (sulfat): Sulfat adalah zat alami yang ditemukan dalam mineral, tanah, dan batuan. Mereka hadir di udara sekitar, air tanah, tumbuhan, dan makanan. Penggunaan komersial utama sulfat adalah dalam industri kimia. Konsentrasi sulfat dalam air laut sekitar 2.700 miligram per liter (mg/L). Ini berkisar dari 3 sampai 30 mg/L di sebagian besar pasokan air tawar, meskipun konsentrasi yang jauh lebih tinggi (1000 mg/L) ditemukan di beberapa lokasi geografis. (Jumlah Sulfat terlarut dalam mg/L).""")
    st.write("""6.Conductivity (daya konduksi) : Air murni bukanlah penghantar arus listrik yang baik, melainkan isolator yang baik. Peningkatan konsentrasi ion meningkatkan konduktivitas listrik air. Umumnya, jumlah padatan terlarut dalam air menentukan konduktivitas listrik. Konduktivitas listrik (EC) sebenarnya mengukur proses ionik suatu larutan yang memungkinkannya mentransmisikan arus. Menurut standar WHO, nilai EC tidak boleh melebihi 400 μS/cm. (Konduktivitas listrik air dalam μS/cm).""")
    st.write("""7.Organic_carbon (Karbon_organik):Total Karbon Organik (TOC) di perairan sumber berasal dari bahan organik alami (NOM) yang membusuk serta sumber sintetis. TOC adalah ukuran jumlah total karbon dalam senyawa organik dalam air murni. Menurut US EPA < 2 mg/L sebagai TOC dalam air olahan / air minum, dan < 4 mg/Lit dalam sumber air yang digunakan untuk pengolahan. (Jumlah karbon organik dalam ppm). """)
    st.write("""8.Trihalomethanes : THM adalah bahan kimia yang dapat ditemukan dalam air yang diolah dengan klorin. Konsentrasi THM dalam air minum bervariasi sesuai dengan kadar bahan organik dalam air, jumlah klorin yang dibutuhkan untuk mengolah air, dan suhu air yang diolah. Tingkat THM hingga 80 ppm dianggap aman dalam air minum. (Jumlah Trihalometana dalam μg/L).""")
    st.write("""9.Turbidity (kekeruhan): Kekeruhan air tergantung pada jumlah zat padat yang ada dalam keadaan tersuspensi. Ini adalah ukuran sifat air yang memancarkan cahaya dan tes ini digunakan untuk menunjukkan kualitas pembuangan limbah sehubungan dengan materi koloid. Nilai rata-rata kekeruhan yang diperoleh Kampus Wondo Genet (0,98 NTU) lebih rendah dari nilai rekomendasi WHO sebesar 5,00 NTU. (Ukur properti pemancar cahaya air di NTU).""")
    st.write("###### Aplikasi ini untuk : kualitas air. ")
    st.write("###### Source Code Aplikasi ada di Github anda bisa acces di link :  ")
    st.write("###### Untuk Wa saya anda bisa hubungi nomer ini : http://wa.me/6289658567766 ")

with upload_data:
    st.write("""# Upload File""")
    uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        st.write("Nama File Anda = ", uploaded_file.name)
        st.dataframe(df)

with preporcessing:
    st.write("""# Preprocessing""")
    df[["ph","Hardness","Solids","Chloramines","Sulfate","Conductivity","Organic_carbon","Trihalomethanes","Turbidity"]].agg(['min','max'])

    df.Potability.value_counts()
    df = df.drop(columns=["ph"])

    X = df.drop(columns="Potability")
    y = df.Potability
    "### Membuang fitur yang tidak diperlukan"
    df

    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    "### Transformasi Label"
    y

    le.inverse_transform(y)

    labels = pd.get_dummies(df.Potability).columns.values.tolist()

    "### Label"
    labels

    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    "### Normalize data transformasi"
    X

    X.shape, y.shape

    le.inverse_transform(y)

    labels = pd.get_dummies(df.Potability).columns.values.tolist()
    
    "### Label"
    labels

    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X

    X.shape, y.shape

with modeling:
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    st.write("""# Modeling """)
    st.subheader("Berikut ini adalah pilihan untuk Modeling")
    st.write("Pilih Model yang Anda inginkan untuk Cek Akurasi")
    naive = st.checkbox('Naive Bayes')
    kn = st.checkbox('K-Nearest Neighbor')
    des = st.checkbox('Decision Tree')
    mod = st.button("Modeling")

    # NB
    GaussianNB(priors=None)

    # Fitting Naive Bayes Classification to the Training set with linear kernel
    nvklasifikasi = GaussianNB()
    nvklasifikasi = nvklasifikasi.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = nvklasifikasi.predict(X_test)
    
    y_compare = np.vstack((y_test,y_pred)).T
    nvklasifikasi.predict_proba(X_test)
    akurasi = round(100 * accuracy_score(y_test, y_pred))
    # akurasi = 10

    # KNN 
    K=10
    knn=KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)

    skor_akurasi = round(100 * accuracy_score(y_test,y_pred))

    # DT

    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    # prediction
    dt.score(X_test, y_test)
    y_pred = dt.predict(X_test)
    #Accuracy
    akurasiii = round(100 * accuracy_score(y_test,y_pred))

    if naive :
        if mod :
            st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(akurasi))
    if kn :
        if mod:
            st.write("Model KNN accuracy score : {0:0.2f}" . format(skor_akurasi))
    if des :
        if mod :
            st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(akurasiii))
    
    eval = st.button("Evaluasi semua model")
    if eval :
        # st.snow()
        source = pd.DataFrame({
            'Nilai Akurasi' : [akurasi,skor_akurasi,akurasiii],
            'Nama Model' : ['Naive Bayes','KNN','Decision Tree']
        })

        bar_chart = alt.Chart(source).mark_bar().encode(
            y = 'Nilai Akurasi',
            x = 'Nama Model'
        )

        st.altair_chart(bar_chart,use_container_width=True)

with implementation:
    st.write("# Implementation")
    ph=st.number_input("nilai ph : ")
    Hardness=st.number_input("kekerasan : ")
    Solids=st.number_input("Padatan) : ")
    Chloramines=st.number_input("kloramin: ")
    Sulfate=st.number_input("(sulfat : ")
    Conductivity=st.number_input("daya konduksi : ")
    Organic_carbon=st.number_input("Karbon organik : ")
    Trihalomethanes=st.number_input(" : ")
    Turbidity=st.number_input("kekeruhan : ")
    Potability=st.number_input("Sifat dpt diminum : ")
    AspectRation=st.number_input("Rasio Aspek : ")	


    def submit():
        # input
        inputs = np.array([[
            ph,
            Hardness,
            Solids,
            Chloramines,
            Sulfate,
            Conductivity,
            Organic_carbon,
            Trihalomethanes,
            Turbidity,
            Potability,
            AspectRation
            ]])
        le = joblib.load("le.save")
        model1 = joblib.load("knn.joblib")
        y_pred3 = model1.predict(inputs)
        st.write(f"Berdasarkan data yang di masukkan, maka prediksinya dapat diminum: {le.inverse_transform(y_pred3)[0]}")

    all = st.button("Submit")
    if all :
        st.balloons()
        submit()

