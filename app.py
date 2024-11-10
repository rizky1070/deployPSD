import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Setting tampilan
st.title("Prediksi Jumlah Pengunjung Menggunakan Bagging Ensemble")
st.write("Aplikasi ini menggunakan Bagging Ensemble Model untuk memprediksi jumlah pengunjung berdasarkan data historis.")

# Membaca data
df = pd.read_csv("data_pengunjung.csv")
# st.write("Data Asli:", df.head())

# Data Preprocessing
df["Date"] = pd.to_datetime(df["Date"])
df = df.dropna(subset=['Pengunjung'])

# Menyiapkan data dengan sliding window
def sliding_window(data, lag):
    series = data['Pengunjung']
    result = pd.DataFrame()
    for l in lag:
        result[f'xt-{l}'] = series.shift(l)
    result['xt'] = series[lag[-1]:]
    result = result.dropna()
    result.index = series.index[lag[-1]:]
    return result

df_slide = sliding_window(df, [1, 2, 3])
df_slide["Date"] = df["Date"].iloc[len(df["Date"]) - len(df_slide):].values
df_slide = df_slide[['Date', 'xt', 'xt-1', 'xt-2', 'xt-3']].set_index('Date')

# Menghapus outlier
def remove_outliers(df, features):
    df_clean = df.copy()
    for col in features:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

features = ['xt-3', 'xt-2', 'xt-1', 'xt']
df_slide_cleaned = remove_outliers(df_slide, features)

# Normalisasi Data
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df_slide_cleaned), columns=df_slide_cleaned.columns, index=df_slide_cleaned.index)

# Melatih model dengan seluruh data
X_train_model = df_normalized.drop(columns=['xt'])
y_train_model = df_normalized['xt']

bagging_model = BaggingRegressor(estimator=LinearRegression(), n_estimators=10, random_state=42)
bagging_model.fit(X_train_model, y_train_model)

# Menambahkan input untuk jumlah bulan yang akan diprediksi
st.subheader("Prediksi Jumlah Pengunjung di Masa Depan")
num_months = st.number_input("Masukkan jumlah bulan yang ingin diprediksi:", min_value=1, value=1, step=1)

# Prediksi untuk beberapa bulan ke depan
df_denormalized = pd.DataFrame(scaler.inverse_transform(df_normalized), columns=df_normalized.columns, index=df_normalized.index)
predictions = []
last_data = df_denormalized.iloc[-1][1:].values.reshape(1, -1)

for _ in range(num_months):
    pred = bagging_model.predict(last_data)[0]
    predictions.append(pred)
    
    # Update last_data untuk prediksi bulan berikutnya
    last_data = np.roll(last_data, -1)
    last_data[0, -1] = pred  # Tambahkan prediksi terbaru sebagai input

# Menampilkan hasil prediksi
st.subheader(f"Hasil Prediksi untuk {num_months} Bulan Mendatang")
for i, pred in enumerate(predictions, start=1):
    st.write(f"Prediksi bulan ke-{i}: {pred:.1f} pengunjung")
