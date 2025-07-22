import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

from data_loader import load_tick_data
from feature_engineering import generate_features
from labeling import create_labels
from utils import load_config, prepare_sequences

# 設定読み込み
config = load_config("config.json")
TRAIN_CONFIG = config["train"]

MODEL_SAVE_PATH = TRAIN_CONFIG["model_save_path"]
DATA_PATH = TRAIN_CONFIG["data_path"]
SEQUENCE_LENGTH = config["sequence_length"]
TP_PIPS = config["tp_pips"]
SL_PIPS = config["sl_pips"]
CONFIDENCE_THRESHOLD = config["confidence_threshold"]
EPOCHS = TRAIN_CONFIG["epochs"]
BATCH_SIZE = TRAIN_CONFIG["batch_size"]
VAL_RATIO = config["val_ratio"]

def main():
    print("📥 ティックデータ読み込み中...")
    tick_data = load_tick_data(DATA_PATH)

    print("🛠️ 特徴量生成中...")
    features = generate_features(tick_data)

    print("🏷️ ラベル生成中...")
    labels = create_labels(tick_data, tp_pips=TP_PIPS, sl_pips=SL_PIPS)

    print("🔍 ラベル分布（元）:", Counter(labels.flatten() if labels.ndim > 1 else labels))

    # データ整合性のため特徴量とラベルの行数を一致させる
    min_len = min(len(features), len(labels))
    features = features.iloc[-min_len:].reset_index(drop=True)
    labels = labels[-min_len:]

    # クラスバランス調整（NO_TRADEダウンサンプリング）
    df_combined = pd.DataFrame(features)
    df_combined["label"] = labels

    no_trade_df = df_combined[df_combined["label"] == 2]
    other_df = df_combined[df_combined["label"] != 2]

    no_trade_sampled = no_trade_df.sample(n=min(len(no_trade_df), len(other_df)*3), random_state=42)
    balanced_df = pd.concat([other_df, no_trade_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)

    features = balanced_df.drop("label", axis=1)
    labels = balanced_df["label"]

    print("🔍 ラベル分布（調整後）:", Counter(labels))

    print("📐 シーケンス準備中...")
    X, y = prepare_sequences(features, labels, SEQUENCE_LENGTH)

    print("📊 データ分割中...")
    val_size = int(len(X) * VAL_RATIO)
    X_train, X_val = X[:-val_size], X[-val_size:]
    y_train, y_val = y[:-val_size], y[-val_size:]

    print("🧠 モデル構築中...")
    model = Sequential()
    model.add(LSTM(64, input_shape=(SEQUENCE_LENGTH, X.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # クラス重み計算
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(np.argmax(y_train, axis=1)), y=np.argmax(y_train, axis=1))
    class_weights_dict = {i: w for i, w in enumerate(class_weights)}
    print("📏 クラス重み:", class_weights_dict)

    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, verbose=1)

    print("🚀 学習開始...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights_dict,
        callbacks=[checkpoint],
        verbose=1
    )

    print(f"✅ 学習完了！モデル保存済み: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
