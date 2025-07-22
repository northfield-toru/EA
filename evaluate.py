import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

from data_loader import load_tick_data
from feature_engineering import generate_features
from labeling import create_labels
from train import prepare_sequences
from utils import load_config

# 設定ファイル読み込み
config = load_config("config.json")

TP_PIPS = config["tp_pips"]
SL_PIPS = config["sl_pips"]
SEQUENCE_LENGTH = config["sequence_length"]
MODEL_PATH = config["model_path"]
DATA_PATH = config["tick_data_path"]
TEST_SIZE = config.get("eval_test_size", 0.2)

def main():
    print("📥 ティックデータ読み込み...")
    tick_data = load_tick_data(DATA_PATH)

    print("🛠️ 特徴量・ラベル生成...")
    features = generate_features(tick_data)
    labels = create_labels(tick_data, tp_pips=TP_PIPS, sl_pips=SL_PIPS)

    print("📐 シーケンス形式へ変換...")
    X, y = prepare_sequences(features, labels, SEQUENCE_LENGTH)

    # 時系列順にテストデータ抽出
    test_size = int(len(X) * EVAL_TEST_SIZE)
    X_test = X[-test_size:]
    y_test = y[-test_size:]

    print("🧠 モデル読み込み:", MODEL_PATH)
    model = load_model(MODEL_PATH)

    print("🔮 予測実行...")
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)

    print("📊 分類レポート:")
    print(classification_report(y_true_labels, y_pred_labels, target_names=["NO_TRADE", "BUY", "SELL"]))

    print("🧮 混同行列:")
    print(confusion_matrix(y_true_labels, y_pred_labels))

if __name__ == "__main__":
    main()
