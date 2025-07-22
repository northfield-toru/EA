import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from collections import Counter

from data_loader import load_tick_data
from feature_engineering import generate_features
from labeling import create_labels
from model import build_model
from utils import load_config

# 設定ファイル読み込み
config = load_config("config.json")

# 設定の読み込み
TP_PIPS = config["tp_pips"]
SL_PIPS = config["sl_pips"]
CONFIDENCE_THRESHOLD = config["confidence_threshold"]
SEQUENCE_LENGTH = config["sequence_length"]
DATA_PATH = config["tick_data_path"]

EPOCHS = config["train"]["epochs"]
BATCH_SIZE = config["train"]["batch_size"]
TEST_SIZE = config["train"]["test_size"]
VAL_SIZE = config["train"]["val_size"]
LEARNING_RATE = config["train"]["learning_rate"]
MODEL_SAVE_PATH = config["train"]["model_save_path"]

def prepare_sequences(features, labels, sequence_length):
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
    labels = create_labels(tick_data, tp_pips=TP_PIPS, sl_pips=SL_PIPS)
    print("🔍 ラベル分布（元）:", Counter(labels))

    # ✅ NO_TRADE（クラス2）をダウンサンプリングしてクラスバランスを調整
    labels = np.array(labels)
    features = features.reset_index(drop=True)

    idx_no_trade = np.where(labels == 2)[0]
    idx_buy = np.where(labels == 0)[0]
    idx_sell = np.where(labels == 1)[0]

    max_no_trade = (len(idx_buy) + len(idx_sell)) * 2
    np.random.seed(42)
    selected_no_trade = np.random.choice(idx_no_trade, size=min(max_no_trade, len(idx_no_trade)), replace=False)

    selected_indices = np.concatenate([idx_buy, idx_sell, selected_no_trade])
    selected_indices.sort()

    features = features.iloc[selected_indices]
    labels = labels[selected_indices]
    print("🔍 ラベル分布（調整後）:", Counter(labels))

    print("📐 シーケンス準備中...")
    X, y = prepare_sequences(features, labels, SEQUENCE_LENGTH)

    print("📊 データ分割中...")
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=False)
    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, shuffle=False)

    print("🧠 モデル構築中...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape, learning_rate=LEARNING_RATE)

    # ✅ クラス重みを自動計算
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = {i: w for i, w in enumerate(class_weights)}
    print("📏 クラス重み:", class_weights_dict)

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True)
    ]

    print("🚀 学習開始...")
    history = model.fit(
        X_train, tf.keras.utils.to_categorical(y_train),
        validation_data=(X_val, tf.keras.utils.to_categorical(y_val)),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights_dict
    )

    print("✅ 学習完了！モデル保存済み:", MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()
