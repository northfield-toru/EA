import json
import numpy as np
import matplotlib.pyplot as plt

def load_config(path: str) -> dict:
    """
    JSON形式の設定ファイルを読み込む

    Parameters:
        path (str): 設定ファイルのパス

    Returns:
        dict: 設定情報
    """
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def prepare_sequences(features: np.ndarray, labels: np.ndarray, sequence_length: int):
    """
    特徴量とラベルをシーケンス形式に変換（LSTMなどの入力用）

    Parameters:
        features (np.ndarray): 特徴量（行数: 時系列長）
        labels (np.ndarray): ラベル
        sequence_length (int): シーケンスの長さ

    Returns:
        Tuple[np.ndarray, np.ndarray]: 入力シーケンス, 対応するラベル
    """
    X, y = [], []
    for i in range(sequence_length, len(features)):
        X.append(features[i-sequence_length:i])
        y.append(labels[i])
    return np.array(X), np.array(y)


def save_history_plot(history, path: str):
    """
    学習履歴（損失・精度）を画像として保存

    Parameters:
        history: モデル学習履歴（historyオブジェクト）
        path (str): 保存先パス（例: "history.png"）
    """
    plt.figure(figsize=(12, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
