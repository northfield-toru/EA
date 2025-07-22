import json
import numpy as np
import pandas as pd

def load_config(path: str) -> dict:
    """
    JSON形式の設定ファイルを読み込む

    Parameters:
        path (str): 設定ファイルのパス

    Returns:
        dict: 設定内容を含む辞書
    """
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config

def prepare_sequences(features: pd.DataFrame, labels: np.ndarray, sequence_length: int):
    """
    特徴量とラベルをシーケンス形式に変換

    Parameters:
        features (pd.DataFrame): 特徴量データフレーム
        labels (np.ndarray): ラベル配列
        sequence_length (int): シーケンスの長さ

    Returns:
        X (np.ndarray): 入力データ [num_samples, sequence_length, num_features]
        y (np.ndarray): 出力ラベル [num_samples]
    """
    X, y = [], []
    for i in range(sequence_length, len(features)):
        X.append(features.iloc[i - sequence_length:i].values)
        y.append(labels[i])
    return np.array(X), np.array(y)
