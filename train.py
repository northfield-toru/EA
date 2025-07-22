import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from feature_engineering import generate_features
from labeling import create_labels
from model import build_model
from utils import (
    load_config,
    prepare_sequences,
    save_history_plot,
    load_tick_data,
)

# 設定の読み込み
config = load_config("config.json")

# 設定の展開
DATA_PATH = config["train"]["data_path"]
MODEL_PATH = config["train"]["model_save_path"]
SEQUENCE_LENGTH = config["train"]["sequence_length"]
TP_PIPS = config["train"]["tp_pips"]
SL_PIPS = config["train"]["sl_pips"]
CONFIDENCE_THRESHOLD = config["train"]["confidence_threshold"]
EPOCHS = config["train"]["epochs"]
BATCH_SIZE = config["train"]["batch_size"]
TEST_SIZE = config["train"]["test_size"]
VAL_SIZE = config["train"]["val_size"]
LEARNING_RATE = config["train"]["learning_rate"]
DOWNSAMPLE_RATIO = config["train"].get("downsample_no_trade_ratio", 1.0)
MAX_SAMPLES = config["train"].get("max_samples", None)
TIME_COLUMN_FORMAT = config.get("time_column_format", "combined")
TIMEZONE = config.get("timezone", "Asia/Tokyo")

def main():
    print("📥 ティックデータ読み込み中...")
    df = load_tick_data(DATA_PATH, time_format=TIME_COLUMN_FORMAT, timezone=TIMEZONE)
    df["mid"] = (df["bid"] + df["ask"]) / 2

    print("🛠️ 特徴量生成中...")
    features = generate_features(df)
    print("✅ 特徴量 shape:", features.shape)

    print("🏷️ ラベル生成中...")
    labels = create_labels(df, TP_PIPS, SL_PIPS, CONFIDENCE_THRESHOLD)
    print("🔍 ラベル分布（元）:", Counter(labels.ravel()))

    if DOWNSAMPLE_RATIO < 1.0 and 2 in labels:
        print(f"⚖️ NO_TRADEクラスをダウンサンプリング中... (比率: {DOWNSAMPLE_RATIO})")
        df_combined = features.copy()
        df_combined["label"] = labels
        no_trade_df = df_combined[df_combined["label"] == 2]
        other_df = df_combined[df_combined["label"] != 2]
        no_trade_sampled = no_trade_df.sample(frac=DOWNSAMPLE_RATIO, random_state=42)
        df_downsampled = pd.concat([no_trade_sampled, other_df]).sample(frac=1, random_state=42)
        features = df_downsampled.drop(columns=["label"])
        labels = df_downsampled["label"].values
        print("🔍 ラベル分布（調整後）:", Counter(labels.tolist()))

    min_len = min(len(features), len(labels))
    features = features.iloc[:min_len]
    labels = labels[:min_len]

    if MAX_SAMPLES and len(features) > MAX_SAMPLES:
        print(f"📉 最大サンプル数 {MAX_SAMPLES} に制限中...")
        sampled_indices = np.random.choice(len(features), size=MAX_SAMPLES, replace=False)
        features = features.iloc[sampled_indices]
        labels = labels[sampled_indices]

    print("📐 シーケンス準備中...")
    X, y = prepare_sequences(features, labels, SEQUENCE_LENGTH)
    print("✅ シーケンス件数:", len(X))

    if len(X) == 0:
        raise ValueError("❌ シーケンス長がデータ数より長すぎます。SEQUENCE_LENGTHを減らすか、もっと多くのデータを読み込んでください。")

    print("📊 データ分割中...")
    y = np.ravel(y)  # flatten（ここで明示的に1次元に）
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    val_ratio_adjusted = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_adjusted, random_state=42, stratify=y_temp
    )

    print("🧠 モデル構築中...")
    model = build_model(input_shape=X.shape[1:], learning_rate=LEARNING_RATE)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(enumerate(class_weights))
    print("📏 クラス重み:", class_weights_dict)

    checkpoint = ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1)

    print("🚀 学習開始...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights_dict,
        callbacks=[checkpoint],
        verbose=1,
    )

    print(f"✅ 学習完了！モデル保存済み: {MODEL_PATH}")
    save_history_plot(history, path="results/training_history.png")

if __name__ == "__main__":
    main()
