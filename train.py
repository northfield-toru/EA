import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import json

from feature_engineering import generate_features
from labeling import create_labels
from utils import load_config, prepare_sequences, save_history_plot

# 設定ファイルの読み込み
config = load_config("config.json")
DATA_PATH = config["data"]["tick_path"]
SEQUENCE_LENGTH = config["train"]["sequence_length"]
TP_PIPS = config["train"]["tp_pips"]
SL_PIPS = config["train"]["sl_pips"]
CONFIDENCE_THRESHOLD = config["train"]["confidence_threshold"]
DOWNSAMPLE_RATIO = config["train"]["downsample_no_trade_ratio"]
MODEL_PATH = config["model"]["save_path"]
HISTORY_PLOT_PATH = config["model"]["history_plot_path"]

def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    print("📥 ティックデータ読み込み中...")
    df = pd.read_csv(DATA_PATH, sep="\t")

    print("🛠️ 特徴量生成中...")
    features = generate_features(df)

    print("🏷️ ラベル生成中...")
    labels = create_labels(df, tp_pips=TP_PIPS, sl_pips=SL_PIPS, confidence_threshold=CONFIDENCE_THRESHOLD)

    # One-hot形式なら整数ラベルに変換
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        label_values = np.argmax(labels, axis=1)
    else:
        label_values = labels

    print("🔍 ラベル分布（元）:", Counter(label_values))

    # NO_TRADE(=2)のダウンサンプリング
    df_combined = features.copy()
    df_combined["label"] = label_values

    no_trade_df = df_combined[df_combined["label"] == 2]
    buy_df = df_combined[df_combined["label"] == 0]
    sell_df = df_combined[df_combined["label"] == 1]

    no_trade_sample = no_trade_df.sample(frac=DOWNSAMPLE_RATIO, random_state=42)
    df_balanced = pd.concat([buy_df, sell_df, no_trade_sample]).sample(frac=1, random_state=42)

    features = df_balanced.drop(columns=["label"])
    labels = df_balanced["label"].values

    print("🔍 ラベル分布（調整後）:", Counter(labels))

    print("📐 シーケンス準備中...")
    X, y = prepare_sequences(features, labels, SEQUENCE_LENGTH)

    print("📊 データ分割中...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # One-hotエンコーディング
    y_train_cat = to_categorical(y_train, num_classes=3)
    y_val_cat = to_categorical(y_val, num_classes=3)

    print("🧠 モデル構築中...")
    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]), num_classes=3)

    # クラス重みの自動計算
    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=np.unique(y_train),
                                         y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    print("📏 クラス重み:", class_weight_dict)

    print("🚀 学習開始...")
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, save_best_only=True)
    ]

    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=50,
        batch_size=64,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    print(f"✅ 学習完了！モデル保存済み: {MODEL_PATH}")
    save_history_plot(history, HISTORY_PLOT_PATH)

if __name__ == "__main__":
    main()
