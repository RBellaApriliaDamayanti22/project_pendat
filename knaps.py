import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
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
    st.write("###### Data Set Ini Adalah : Harumanis Mango Physical Measurements (Pengukuran Fisik Mangga Harumanis ")
    st.write("###### Sumber Data Set dari Kaggle : https://www.kaggle.com/datasets/mohdnazuan/harumanis-mango-physical-measurement")
    st.write("""###### Penjelasan setiap kolom : """)
    st.write("""1.Weight(Berat):Berat mangga dalam gram (g)""")
    st.write("""2.Length (Panjang):Panjang buah mangga dalam centimeter (cm). Sebuah mangga arumanis panjangnya sekitar 15 cm dengan berat per buah 450 gram (rata-rata).  """)
    st.write("""3.Circumference (Lingkar): Lingkar mangga dalam sentimeter (cm)""")
    st.write("###### Aplikasi ini untuk : Pengukuran Fisik Mangga Harumanis")
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
    X = df.drop(labels = 'Grade',axis = 1)
    y = df['Grade'].map({'A': 0, 'B': 1})
    #df[["No","Weight","Length","Circumference"]].agg(['min','max'])

    #df.Grade.value_counts()
    #df = df.drop(columns=["No"])

    #X = df.drop(columns="Grade")
    #y = df.Grade
    #"### Membuang fitur yang tidak diperlukan"
    #df

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 88)

    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    "### Transformasi Label"
    y

    le.inverse_transform(y)

    labels = pd.get_dummies(df.Grade).columns.values.tolist()

    "### Label"
    labels

    scaler1 = MinMaxScaler()
    scaler1.fit(X)
    X = scaler1.transform(X)
    "### Normalize data transformasi"
    X

    X.shape, y.shape

    le.inverse_transform(y)

    labels = pd.get_dummies(df.Grade).columns.values.tolist()
    
    "### Label"
    labels

    scaler1 = MinMaxScaler()
    scaler1.fit(X)
    X = scaler1.transform(X)
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
    lr = st.checkbox('LogisticRegression')
    kn = st.checkbox('K-Nearest Neighbor')
    des = st.checkbox('Decision Tree')
    mod = st.button("Modeling")

    # LR
    metode1 = LogisticRegression()
    metode1.fit(scaled_X_train,y_train)
    metode1.coef_
    y_pred = metode1.predict(scaled_X_test)

    akurasi = round(100 * accuracy_score(y_test, y_pred))
    #GaussianNB(priors=None)

    # Fitting Naive Bayes Classification to the Training set with linear kernel
    #nvklasifikasi = GaussianNB()
    #nvklasifikasi = nvklasifikasi.fit(X_train, y_train)

    # Predicting the Test set results
    #y_pred = nvklasifikasi.predict(X_test)
    
    #y_compare = np.vstack((y_test,y_pred)).T
    #nvklasifikasi.predict_proba(X_test)
    #akurasi = round(100 * accuracy_score(y_test, y_pred))
    # akurasi = 10

    # KNN 
    # Build a function for choosing reasonable K value

    error_rates = []

    scaler = StandardScaler()
    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_test = scaler.transform(X_test)

    for k in range(1,30):
        knn_model = KNeighborsClassifier(n_neighbors=k)
        knn_model.fit(scaled_X_train,y_train)
        y_knn_predict = knn_model.predict(scaled_X_test)
        
        error_k = 1 - accuracy_score(y_test,y_knn_predict)
        
        error_rates.append(error_k)

        knn_model = KNeighborsClassifier(n_neighbors=5)
        knn_model.fit(scaled_X_train,y_train)
        y_knn_predict = knn_model.predict(scaled_X_test)

        skor_akurasi = round(100 * accuracy_score(y_test,y_knn_predict))
    #K=10
    #knn=KNeighborsClassifier(n_neighbors=K)
    #knn.fit(X_train,y_train)
    #y_pred=knn.predict(X_test)

    #skor_akurasi = round(100 * accuracy_score(y_test,y_pred))

    # DT

    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    # prediction
    dt.score(X_test, y_test)
    y_pred = dt.predict(X_test)
    #Accuracy
    akurasiii = round(100 * accuracy_score(y_test,y_pred))

    if lr :
        if mod :
            st.write('Model Logistic Regression accuracy score: {0:0.2f}'. format(akurasi))
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
            'Nama Model' : ['Logistic Regression','KNN','Decision Tree']
        })

        bar_chart = alt.Chart(source).mark_bar().encode(
            y = 'Nilai Akurasi',
            x = 'Nama Model'
        )

        st.altair_chart(bar_chart,use_container_width=True)

with implementation:
    st.write("# Implementation")
    Weight=st.number_input("Besar : ")
    Length=st.number_input("Panjang : ")
    Circumference=st.number_input("Lingkar) : ")


    def submit():
        # input
        inputs = np.array([[
            Weight,
            Length,
            Circumference
            ]])
        le = joblib.load("le.save")
        model1 = joblib.load("knn.joblib")
        y_pred3 = model1.predict(inputs)
        st.write(f"Berdasarkan data yang di masukkan, maka prediksinya dapat A: {le.inverse_transform(y_pred3)[0]}")

    all = st.button("Submit")
    if all :
        st.balloons()
        submit()

