import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

try:
    df = pd.read_csv('../heart_raw.csv')
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
            raise KeyError(f"Kolom '{col}' tidak ditemukan di DataFrame.")
        X = df.drop(columns=[col])
        y = df[col]
        scaler = StandardScaler()
        scaled = scaler.fit_transform(X)
        scaled = pd.DataFrame(scaled, columns=X.columns)
        scaled = pd.concat([scaled.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
        return scaler, scaled
    except Exception as e:
        raise ValueError(f"Error saat scaling: {str(e)}")
    
def split_data(df, col, test_size):
    try:
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan di DataFrame.")

        X = df.drop(columns=[col])
        y = df[col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise ValueError(f"Error saat spliting data: {str(e)}")

encoded = encode_categorical(df)
scaler, scaled = scale_features(encoded, 'HeartDisease')
X_train, X_test, y_train, y_test = split_data(scaled, 'HeartDisease', 0.3)

try:
    joblib.dump(scaler, 'scaler.joblib')
    print('Berhasil menyimpan hasil skala normalisasi')
except NameError:
    raise NameError("Objek scaler tidak didefinisikan.")

try:
    scaled.to_csv('heart_preprocessing/heart_preprocessing.csv')
    print('Berhasil menyimpan hasil normalisasi')
except NameError:
    raise NameError("Objek scaled tidak didefinisikan.")

try:
    scaled.to_csv('heart_preprocessing/X_train.csv')
    print('Berhasil menyimpan hasil spliting X_train')
except NameError:
    raise NameError("Objek X_train tidak didefinisikan.")

try:
    scaled.to_csv('heart_preprocessing/X_test.csv')
    print('Berhasil menyimpan hasil spliting X_test')
except NameError:
    raise NameError("Objek X_test tidak didefinisikan.")

try:
    scaled.to_csv('heart_preprocessing/y_train.csv')
    print('Berhasil menyimpan hasil spliting y_train')
except NameError:
    raise NameError("Objek y_train tidak didefinisikan.")

try:
    scaled.to_csv('heart_preprocessing/y_test.csv')
    print('Berhasil menyimpan hasil spliting y_test')
except NameError:
    raise NameError("Objek y_test tidak didefinisikan.")