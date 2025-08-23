import pandas as pd
from sklearn.preprocessing import StandardScaler
def fit_scaler(X_train):
    """
    학습 데이터에 맞춰 StandardScaler 학습(fit) 후 반환
    - X_train: (N, T, F) 형태 시계열 데이터
    """
    B, T, F = X_train.shape
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train.reshape(-1, F)).reshape(B, T, F)
    return scaler, X_scaled

def transform_scaler(X, scaler):
    """
    이미 학습된 scaler로 새로운 데이터 변환
    - X: (N, T, F) 형태 시계열 데이터
    """
    B, T, F = X.shape
    X_scaled = scaler.transform(X.reshape(-1, F)).reshape(B, T, F)
    return X_scaled

def inverse_scaler(X_scaled, scaler):
    """
    스케일링된 데이터를 원래 값으로 되돌림
    - X_scaled: (N, T, F) 형태 시계열 데이터
    """
    B, T, F = X_scaled.shape
    X_orig = scaler.inverse_transform(X_scaled.reshape(-1, F)).reshape(B, T, F)
    return X_orig