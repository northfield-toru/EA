import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_sequences(features: np.ndarray, labels: np.ndarray, sequence_length: int):
    X, y = [], []
    for i in range(sequence_length, len(features)):
        X.append(features[i-sequence_length:i])
        y.append(labels[i])
    return np.array(X, dtype=np.float32), np.array(y)


def save_history_plot(history, path: str):
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


def load_tick_data(path, time_format="combined", timezone="Asia/Tokyo"):
    df = pd.read_csv(path, sep="\t")

    # ✅ 応急処置：カラム名の自動リネーム
    df.columns = [col.strip("<>").lower() for col in df.columns]

    # ✅ 時間列の処理
    if time_format == "combined":
        df["time"] = pd.to_datetime(df["time"])
    elif time_format == "separated":
        df["time"] = pd.to_datetime(df["date"] + " " + df["time"])
        df.drop(columns=["date"], inplace=True)
    else:
        raise ValueError("time_column_format must be 'combined' or 'separated'")

    # ✅ bid/ask 数値化とNaN除去
    df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
    df["ask"] = pd.to_numeric(df["ask"], errors="coerce")
    df = df.dropna(subset=["bid", "ask"])

    # ✅ タイムゾーン処理とソート
    df["time"] = df["time"].dt.tz_localize("UTC").dt.tz_convert(timezone)
    df = df.sort_values("time").reset_index(drop=True)

    # ✅ mid計算とインデックス化
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df = df.set_index("time")

    return df
