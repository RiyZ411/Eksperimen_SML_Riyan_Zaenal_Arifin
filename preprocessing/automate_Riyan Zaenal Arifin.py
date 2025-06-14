import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os


try:
    df = pd.read_csv('data/heart_raw.csv')
except FileNotFoundError:
    raise FileNotFoundError("File heart_raw.csv tidak ditemukan.")

def encode_categorical(df):
    try:
        kategori_fitur = df.select_dtypes(include=['object', 'category']).columns
        encoded = pd.get_dummies(df, columns=kategori_fitur)
        encoded = encoded.astype(int)
        return encoded
    except Exception as e:
        raise ValueError(f"Error saat encoding: {str(e)}")

def scale_features(df, col):
    try:
        if col not in df.columns:
            raise KeyError(f"Kolom target '{col}' tidak ditemukan.")
        X = df.drop(columns=[col])
        y = df[col]
        scaler = StandardScaler()
        scaled = scaler.fit_transform(X)
        scaled = pd.DataFrame(scaled, columns=X.columns)
        scaled = pd.concat([scaled.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
        return scaler, scaled
    except Exception as e:
        raise ValueError(f"Error saat scaling: {str(e)}")

encoded = encode_categorical(df)
scaler, scaled = scale_features(encoded, 'HeartDisease')

try:
    joblib.dump(scaler, './scaler.joblib')
    print('Berhasil menyimpan hasil skala normalisasi')
except NameError:
    raise NameError("Objek 'scaler' tidak didefinisikan.")

try:
    scaled.to_csv('./heart_preprocessing.csv')
    print('Berhasil menyimpan hasil normalisasi')
except NameError:
    raise NameError("Objek 'scaled' tidak didefinisikan.")