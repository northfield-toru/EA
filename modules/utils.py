import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight

def load_config(config_path="config.json"):
    """設定ファイルを読み込む"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

def setup_logging(log_dir):
    """ログ設定を初期化"""
    os.makedirs(log_dir, exist_ok=True)
    
    # ログファイル設定
    log_file = os.path.join(log_dir, 'training.log')
    
    # ログフォーマット設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def create_directories():
    """必要なディレクトリを作成"""
    directories = ['data', 'models', 'logs', 'modules']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def load_tick_data(file_path, sample_rate=1):
    """
    ティックデータを読み込む
    
    Args:
        file_path: CSVファイルパス
        sample_rate: サンプリングレート（1なら全データ、2なら半分など）
    
    Returns:
        DataFrame: 読み込んだティックデータ
    """
    try:
        # タブ区切りCSVを読み込み
        df = pd.read_csv(file_path, sep='\t')
        
        # ヘッダー名を正規化（<>を除去）
        df.columns = [col.strip('<>') for col in df.columns]
        
        # 必要カラムの存在確認
        required_columns = ['DATE', 'TIME', 'BID', 'ASK']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # データ型変換
        df['BID'] = pd.to_numeric(df['BID'], errors='coerce')
        df['ASK'] = pd.to_numeric(df['ASK'], errors='coerce')
        
        # MID価格を計算
        df['MID'] = (df['BID'] + df['ASK']) / 2
        
        # NaNを除去
        df = df.dropna(subset=['BID', 'ASK', 'MID'])
        
        # 時系列順にソート（未来リーク防止）
        df = df.sort_values(['DATE', 'TIME']).reset_index(drop=True)
        
        # サンプリング
        if sample_rate > 1:
            df = df.iloc[::sample_rate].reset_index(drop=True)
        
        logging.info(f"Loaded {len(df)} tick records from {file_path}")
        logging.info(f"Sample rate: {sample_rate}, Data range: {df['DATE'].min()} to {df['DATE'].max()}")
        
        return df
        
    except Exception as e:
        logging.error(f"Error loading tick data: {e}")
        raise

def calculate_class_weights(labels):
    """
    クラス重みを自動計算
    
    Args:
        labels: ラベル配列
        
    Returns:
        dict: クラス重み辞書
    """
    # ユニークなクラスを取得
    unique_classes = np.unique(labels)
    
    # クラス重みを計算
    class_weights = compute_class_weight(
        'balanced',
        classes=unique_classes,
        y=labels
    )
    
    # 辞書形式で返す
    class_weight_dict = dict(zip(unique_classes, class_weights))
    
    return class_weight_dict

def save_class_weights(class_weights, log_dir):
    """クラス重みをJSONファイルで保存"""
    class_weights_file = os.path.join(log_dir, 'class_weights.json')
    
    # numpy型をPython標準型に変換
    serializable_weights = {str(k): float(v) for k, v in class_weights.items()}
    
    with open(class_weights_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_weights, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Class weights saved: {serializable_weights}")

def plot_label_distribution(labels, log_dir):
    """ラベル分布をプロットして保存"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ラベル名マッピング
    label_names = {0: 'BUY', 1: 'SELL', 2: 'NO_TRADE'}
    
    # カウント計算
    unique, counts = np.unique(labels, return_counts=True)
    percentages = counts / len(labels) * 100
    
    # 棒グラフ
    bars = ax1.bar([label_names[i] for i in unique], counts)
    ax1.set_title('Label Distribution (Count)')
    ax1.set_ylabel('Count')
    
    # 数値を棒グラフに表示
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                str(count), ha='center', va='bottom')
    
    # 円グラフ
    ax2.pie(counts, labels=[f'{label_names[i]}\n({count}件, {pct:.1f}%)' 
                           for i, count, pct in zip(unique, counts, percentages)],
            autopct='', startangle=90)
    ax2.set_title('Label Distribution (Percentage)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'label_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # CSV形式でも保存
    distribution_df = pd.DataFrame({
        'Label': [label_names[i] for i in unique],
        'Count': counts,
        'Percentage': percentages
    })
    distribution_df.to_csv(os.path.join(log_dir, 'label_distribution.csv'), index=False)
    
    logging.info("Label distribution saved")
    return distribution_df

def plot_threshold_analysis(y_true, y_pred_proba, log_dir):
    """閾値別F1スコア分析"""
    from sklearn.metrics import f1_score
    
    thresholds = np.arange(0.1, 1.0, 0.1)
    results = []
    
    for threshold in thresholds:
        # 最大確率が閾値を超える場合のみ予測、そうでなければNO_TRADE（2）
        y_pred_thresh = np.where(
            np.max(y_pred_proba, axis=1) >= threshold,
            np.argmax(y_pred_proba, axis=1),
            2  # NO_TRADE
        )
        
        # F1スコア計算
        f1_macro = f1_score(y_true, y_pred_thresh, average='macro')
        f1_buy = f1_score(y_true, y_pred_thresh, labels=[0], average='macro')
        f1_sell = f1_score(y_true, y_pred_thresh, labels=[1], average='macro')
        f1_no_trade = f1_score(y_true, y_pred_thresh, labels=[2], average='macro')
        
        results.append({
            'Threshold': threshold,
            'F1_Macro': f1_macro,
            'F1_BUY': f1_buy,
            'F1_SELL': f1_sell,
            'F1_NO_TRADE': f1_no_trade,
            'Pred_BUY': np.sum(y_pred_thresh == 0),
            'Pred_SELL': np.sum(y_pred_thresh == 1),
            'Pred_NO_TRADE': np.sum(y_pred_thresh == 2)
        })
    
    results_df = pd.DataFrame(results)
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # F1スコア推移
    ax1.plot(results_df['Threshold'], results_df['F1_Macro'], 'o-', label='Macro F1', linewidth=2)
    ax1.plot(results_df['Threshold'], results_df['F1_BUY'], 's-', label='BUY F1', alpha=0.7)
    ax1.plot(results_df['Threshold'], results_df['F1_SELL'], '^-', label='SELL F1', alpha=0.7)
    ax1.plot(results_df['Threshold'], results_df['F1_NO_TRADE'], 'd-', label='NO_TRADE F1', alpha=0.7)
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('F1 Score vs Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 予測分布
    ax2.plot(results_df['Threshold'], results_df['Pred_BUY'], 'o-', label='BUY', linewidth=2)
    ax2.plot(results_df['Threshold'], results_df['Pred_SELL'], 's-', label='SELL', linewidth=2)
    ax2.plot(results_df['Threshold'], results_df['Pred_NO_TRADE'], '^-', label='NO_TRADE', linewidth=2)
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Prediction Count')
    ax2.set_title('Prediction Distribution vs Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'threshold_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # CSV保存
    results_df.to_csv(os.path.join(log_dir, 'threshold_analysis.csv'), index=False)
    
    logging.info("Threshold analysis saved")
    return results_df

def save_model_summary(config, metrics, log_dir):
    """モデルサマリーをCSVに追記保存"""
    summary_file = 'logs/model_summary.csv'
    
    # 新しい行のデータ
    new_row = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M'),
        'tp_pips': config['trading']['tp_pips'],
        'sl_pips': config['trading']['sl_pips'],
        'threshold': config['trading']['trade_threshold'],
        'feature_window': config['data']['feature_window'],
        'future_window': config['data']['future_window'],
        'sample_rate': config['data']['sample_rate'],
        'model_type': config['model']['type'],
        'learning_rate': config['model']['learning_rate'],
        'epochs': config['model']['epochs'],
        'batch_size': config['model']['batch_size'],
        'val_accuracy': metrics.get('val_accuracy', 0),
        'f1_macro': metrics.get('f1_macro', 0),
        'buy_ratio': metrics.get('buy_ratio', 0),
        'sell_ratio': metrics.get('sell_ratio', 0),
        'no_trade_ratio': metrics.get('no_trade_ratio', 0),
        'log_dir': log_dir
    }
    
    # DataFrameに変換
    new_df = pd.DataFrame([new_row])
    
    # 既存ファイルがあれば追記、なければ新規作成
    if os.path.exists(summary_file):
        existing_df = pd.read_csv(summary_file)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
    
    # 保存
    combined_df.to_csv(summary_file, index=False)
    logging.info(f"Model summary saved to {summary_file}")

def check_gpu_availability():
    """GPU利用可能性をチェック"""
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            logging.info(f"GPU available: {len(gpus)} device(s)")
            for i, gpu in enumerate(gpus):
                logging.info(f"  GPU {i}: {gpu}")
            return True
        else:
            logging.info("No GPU available, using CPU")
            return False
    except Exception as e:
        logging.warning(f"Error checking GPU availability: {e}")
        return False

def memory_usage_check():
    """メモリ使用量をチェック"""
    import psutil
    memory = psutil.virtual_memory()
    logging.info(f"Memory usage: {memory.percent}% ({memory.used // (1024**3):.1f}GB / {memory.total // (1024**3):.1f}GB)")
    
    if memory.percent > 80:
        logging.warning("High memory usage detected! Consider reducing batch size or data size.")
    
    return memory.percent