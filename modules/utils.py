import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, Any, Tuple, Optional
import tensorflow as tf
from typing import Dict, Any, List

def load_config(config_path: str = 'config.json') -> Dict[str, Any]:
    """設定ファイルを読み込む"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"設定ファイルの形式が正しくありません: {e}")

def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """ログ設定を初期化"""
    
    # 外部ライブラリのログレベルを制限
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def setup_gpu_memory_growth():
    """GPU メモリの段階的使用を設定"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU memory growth setup failed: {e}")
    else:
        print("No GPU detected, using CPU")

def create_directories(config: Dict[str, Any]):
    """必要なディレクトリを作成"""
    dirs_to_create = [
        'data',
        'models',
        'modules',
        'logs',
        config['data']['output_dir'],
        os.path.dirname(config['data']['input_file'])
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)

def calculate_mid_price(bid: float, ask: float) -> float:
    """MID価格を計算（スプレッド補正の基準）"""
    return (bid + ask) / 2.0

def apply_spread_correction(price: float, spread_pips: float, pip_value: float, direction: str = 'buy') -> float:
    """
    スプレッド補正を適用
    direction: 'buy' または 'sell'
    """
    spread_value = spread_pips * pip_value
    if direction.lower() == 'buy':
        return price + (spread_value / 2)  # 買いの場合はスプレッドの半分を加算
    else:
        return price - (spread_value / 2)  # 売りの場合はスプレッドの半分を減算

def pips_to_price(pips: float, pip_value: float) -> float:
    """pips値を価格変動に変換"""
    return pips * pip_value

def price_to_pips(price_diff: float, pip_value: float) -> float:
    """価格変動をpips値に変換"""
    return price_diff / pip_value

def parse_datetime(date_str: str, time_str: str) -> datetime:
    """日付と時刻文字列をdatetimeオブジェクトに変換"""
    try:
        datetime_str = f"{date_str} {time_str}"
        return pd.to_datetime(datetime_str, format='%Y.%m.%d %H:%M:%S.%f')
    except ValueError:
        # フォーマットが異なる場合の対応
        return pd.to_datetime(f"{date_str} {time_str}")

def validate_data_integrity(df: pd.DataFrame, config: Dict[str, Any]) -> bool:
    """
    データの整合性をチェック
    """
    required_columns = config['data']['use_columns']
    
    # 必要なカラムの存在確認
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"必要なカラムが不足: {missing_columns}")
    
    # データ型のチェック
    if not pd.api.types.is_numeric_dtype(df['BID']):
        raise ValueError("BIDカラムが数値型ではありません")
    if not pd.api.types.is_numeric_dtype(df['ASK']):
        raise ValueError("ASKカラムが数値型ではありません")
    
    # 価格データの妥当性チェック
    if (df['BID'] <= 0).any() or (df['ASK'] <= 0).any():
        raise ValueError("価格データに0以下の値が含まれています")
    
    if (df['ASK'] < df['BID']).any():
        raise ValueError("ASK < BIDの異常なデータが含まれています")
    
    return True

def memory_usage_mb() -> float:
    """現在のメモリ使用量をMBで返す"""
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def save_model_metadata(model_path: str, config: Dict[str, Any], 
                       training_history: Dict[str, Any], 
                       feature_names: List[str] = None,
                       scaling_params: Dict[str, Any] = None):
    """
    モデルメタデータをJSONファイルに保存（スケーリングパラメータ対応）
    """
    metadata = {
        "model_path": model_path,
        "config": config,
        "created_at": datetime.now().isoformat(),
        "feature_names": feature_names or [],
        "training_history": training_history
    }
    
    # スケーリングパラメータを追加
    if scaling_params:
        metadata["scaling_params"] = scaling_params
    
    # メタデータファイルパス
    metadata_path = model_path.replace('.h5', '_metadata.json')
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"モデルメタデータ保存完了: {metadata_path}")
    
    # 訓練ログCSVも保存
    if training_history:
        log_path = model_path.replace('.h5', '_training_log.csv')
        df_history = pd.DataFrame(training_history)
        df_history.to_csv(log_path, index=False)
        logger.info(f"訓練ログCSV保存完了: {log_path}")
    
    return metadata_path

def load_model_metadata(model_path: str) -> Dict[str, Any]:
    """モデルのメタデータを読み込む"""
    metadata_path = model_path.replace('.h5', '_metadata.json')
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def set_random_seed(seed: int):
    """再現性のためのシード設定"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    import random
    random.seed(seed)

def time_series_split(n_samples: int, validation_split: float, test_split: float) -> Tuple[slice, slice, slice]:
    """
    時系列データ用の分割インデックスを生成（昇順維持）
    """
    test_size = int(n_samples * test_split)
    val_size = int(n_samples * validation_split)
    train_size = n_samples - test_size - val_size
    
    train_slice = slice(0, train_size)
    val_slice = slice(train_size, train_size + val_size)
    test_slice = slice(train_size + val_size, n_samples)
    
    return train_slice, val_slice, test_slice

def format_prediction_output(predictions: np.ndarray, timestamps: np.ndarray, 
                           config: Dict[str, Any]) -> pd.DataFrame:
    """予測結果をMT5連携用フォーマットに変換"""
    class_names = config['labels']['class_names']
    
    results = []
    for i, (pred, timestamp) in enumerate(zip(predictions, timestamps)):
        predicted_class = np.argmax(pred)
        confidence = np.max(pred)
        
        results.append({
            'timestamp': timestamp,
            'predicted_class': predicted_class,
            'predicted_label': class_names[predicted_class],
            'confidence': confidence,
            'buy_prob': pred[0],
            'sell_prob': pred[1],
            'no_trade_prob': pred[2]
        })
    
    return pd.DataFrame(results)