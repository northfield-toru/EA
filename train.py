import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from sklearn.model_selection import train_test_split

from data_loader import load_tick_data
from feature_engineering import generate_features
from labeling import create_labels
from model import build_model
from utils import load_config

# 設定ファイル読み込み
config = load_config("config.json")

TP_PIPS = config["tp_pips"]
SL_PIPS = config["sl_pips"]
CONFIDENCE_THRESHOLD = config["confidence_threshold"]
SEQUENCE_LENGTH = config["sequence_length"]

EPOCHS = config["train"]["epochs"]
BATCH_SIZE = config["train"]["batch_size"]
TEST_SIZE = config["train"]["test_size"]
VAL_SIZE = config["train"]["val_size"]
LEARNING_RATE = config["train"]["learning_rate"]
MODEL_SAVE_PATH = config["train"]["model_save_path"]
DATA_PATH = config["train"]["data_path"]

def prepare_sequences(features, labels, sequence_length):
    """
    時系列データをシーケンス形式に変換
    """
    X, y = [], []
    for i in range(sequence_length, len(features)):
        X.append(features[i-sequence_length:i])
        y.append(labels[i])
    return np.array(X), np.array(y)

def main():
    print("📥 ティックデータ読み込み中...")
    tick_data = load_tick_data(DATA_PATH)
    
    print("🛠️ 特徴量生成中...")
    features = generate_features(tick_data)

    print("🏷️ ラベル生成中...")
    labels = create_labels(tick_data)

    print("📐 シーケンス準備中...")
    X, y = prepare_sequences(features, labels, SEQUENCE_LENGTH)

    print("📊 データ分割中...")
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=False)
    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, shuffle=False)

    print("🧠 モデル構築...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape, learning_rate=LEARNING_RATE)

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True)
    ]

    print("🚀 学習開始...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )

    print("✅ 学習完了！モデル保存済み:", MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()
