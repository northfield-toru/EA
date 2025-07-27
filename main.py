#!/usr/bin/env python3
"""
USDJPYスキャルピング向けAI学習モデル
メインスクリプト
"""

import os
import sys
import warnings
import traceback
from datetime import datetime

# 警告を抑制
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.utils import (
    load_config, setup_logging, create_directories, 
    load_tick_data, plot_label_distribution, save_model_summary,
    check_gpu_availability, memory_usage_check
)
from modules.feature_engineering import FeatureEngineer
from modules.labeling import LabelCreator
from modules.train import ModelTrainer

def main():
    """メイン実行関数"""
    
    print("=" * 60)
    print("USDJPYスキャルピング向けAI学習モデル 2025年版")
    print("=" * 60)
    
    try:
        # 1. 初期設定
        print("\n1. 初期設定...")
        
        # ディレクトリ作成
        create_directories()
        
        # 設定読み込み
        config = load_config()
        print(f"設定読み込み完了: {config['trading']['pair']}")
        
        # ログ設定
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        log_dir = f"logs/{timestamp}"
        logger = setup_logging(log_dir)
        
        logger.info("=" * 60)
        logger.info("USDJPYスキャルピング向けAI学習モデル開始")
        logger.info("=" * 60)
        
        # GPU確認
        gpu_available = check_gpu_availability()
        logger.info(f"GPU利用可能: {gpu_available}")
        
        # メモリ使用量確認
        memory_usage_check()
        
        # 設定情報をログ出力
        logger.info(f"設定情報:")
        logger.info(f"  通貨ペア: {config['trading']['pair']}")
        logger.info(f"  TP/SL: {config['trading']['tp_pips']}/{config['trading']['sl_pips']} pips")
        logger.info(f"  スプレッド: {config['trading']['spread_pips']} pips")
        logger.info(f"  サンプリングレート: {config['data']['sample_rate']}")
        logger.info(f"  特徴量ウィンドウ: {config['data']['feature_window']}")
        logger.info(f"  未来ウィンドウ: {config['data']['future_window']}")
        logger.info(f"  モデルタイプ: {config['model']['type']}")
        
        # 2. データ読み込み
        print("\n2. データ読み込み...")
        logger.info("ティックデータ読み込み開始")
        
        df = load_tick_data(
            config['data']['input_file'], 
            config['data']['sample_rate']
        )
        
        if len(df) == 0:
            raise ValueError("データが空です")
        
        logger.info(f"データ読み込み完了: {len(df)} records")
        print(f"データ件数: {len(df):,} records")
        
        # データ基本統計
        logger.info(f"データ統計:")
        logger.info(f"  期間: {df['DATE'].min()} ～ {df['DATE'].max()}")
        logger.info(f"  BID範囲: {df['BID'].min():.5f} ～ {df['BID'].max():.5f}")
        logger.info(f"  ASK範囲: {df['ASK'].min():.5f} ～ {df['ASK'].max():.5f}")
        logger.info(f"  MID範囲: {df['MID'].min():.5f} ～ {df['MID'].max():.5f}")
        
        # 3. ラベル作成
        print("\n3. ラベル作成...")
        logger.info("ラベル作成開始")
        
        label_creator = LabelCreator(config)
        labels = label_creator.create_labels(df)
        
        # 有効データ範囲を取得
        valid_end = label_creator.get_valid_data_range(len(df))
        
        # データとラベルを有効範囲に制限
        df_valid = df.iloc[:valid_end].copy()
        labels_valid = labels[:valid_end]
        
        logger.info(f"有効データ: {len(df_valid)} records")
        
        # ラベル品質分析
        label_analysis = label_creator.analyze_label_quality(labels_valid, df_valid)
        
        # ラベル分布プロット
        distribution_df = plot_label_distribution(labels_valid, log_dir)
        print(f"ラベル分布:")
        print(distribution_df.to_string(index=False))
        
        # 4. 特徴量エンジニアリング
        print("\n4. 特徴量エンジニアリング...")
        logger.info("特徴量エンジニアリング開始")
        
        feature_engineer = FeatureEngineer(config)
        
        # テクニカル指標計算
        df_with_indicators = feature_engineer.calculate_indicators(df_valid)
        
        # 特徴量配列作成
        features, feature_names = feature_engineer.create_features(df_with_indicators)
        logger.info(f"特徴量: {len(feature_names)} 種類")
        logger.info(f"特徴量名: {feature_names}")
        
        # シーケンス作成
        feature_window = config['data']['feature_window']
        if len(features) < feature_window:
            raise ValueError(f"データが特徴量ウィンドウより短い: {len(features)} < {feature_window}")
        
        X, y = feature_engineer.create_sequences(features, labels_valid)
        logger.info(f"シーケンス作成完了: X{X.shape}, y{y.shape}")
        
        # 正規化
        X_normalized, _, norm_params = feature_engineer.normalize_features(X)
        logger.info("特徴量正規化完了")
        
        # メモリ使用量確認
        memory_usage_check()
        
        # 5. モデル訓練
        print("\n5. モデル訓練...")
        logger.info("モデル訓練開始")
        
        trainer = ModelTrainer(config, log_dir)
        training_results = trainer.train_model(X_normalized, y, feature_names)
        
        # 6. 結果保存
        print("\n6. 結果保存...")
        logger.info("結果保存開始")
        
        # モデルサマリー保存
        metrics = training_results['metrics']
        save_model_summary(config, metrics, log_dir)
        
        # 正規化パラメータ保存
        import json
        norm_params_serializable = {
            'mean': norm_params['mean'].tolist(),
            'std': norm_params['std'].tolist(),
            'feature_names': feature_names
        }
        
        with open(os.path.join(log_dir, 'normalization_params.json'), 'w') as f:
            json.dump(norm_params_serializable, f, indent=2)
        
        # 設定ファイルのコピー保存
        import shutil
        shutil.copy('config.json', os.path.join(log_dir, 'config.json'))
        
        # 最終結果表示
        print("\n" + "=" * 60)
        print("訓練完了!")
        print("=" * 60)
        print(f"ログディレクトリ: {log_dir}")
        print(f"モデルファイル: {training_results['model_path']}")
        print(f"検証精度: {metrics['val_accuracy']:.4f}")
        print(f"マクロF1スコア: {metrics['f1_macro']:.4f}")
        print(f"予測分布 - BUY: {metrics['buy_ratio']:.3f}, SELL: {metrics['sell_ratio']:.3f}, NO_TRADE: {metrics['no_trade_ratio']:.3f}")
        
        # 警告があれば表示
        if label_analysis.get('warnings'):
            print("\n警告:")
            for warning in label_analysis['warnings']:
                print(f"  - {warning}")
        
        logger.info("すべての処理が正常に完了しました")
        
        return training_results
        
    except Exception as e:
        error_msg = f"エラーが発生しました: {str(e)}"
        print(f"\n❌ {error_msg}")
        
        # ログが設定されている場合はログにも出力
        if 'logger' in locals():
            logger.error(error_msg)
            logger.error(f"トレースバック:\n{traceback.format_exc()}")
        else:
            print(f"トレースバック:\n{traceback.format_exc()}")
        
        # エラー情報をファイルに保存
        try:
            error_log_dir = f"logs/error_{datetime.now().strftime('%Y%m%d_%H%M')}"
            os.makedirs(error_log_dir, exist_ok=True)
            
            with open(os.path.join(error_log_dir, 'error.txt'), 'w', encoding='utf-8') as f:
                f.write(f"エラー発生時刻: {datetime.now()}\n")
                f.write(f"エラーメッセージ: {str(e)}\n\n")
                f.write(f"トレースバック:\n{traceback.format_exc()}")
            
            print(f"エラー情報を保存しました: {error_log_dir}")
            
        except Exception as save_error:
            print(f"エラー情報の保存に失敗: {save_error}")
        
        sys.exit(1)

def check_requirements():
    """必要なパッケージと環境をチェック"""
    
    print("環境チェック中...")
    
    # パッケージとインポート名のマッピング
    required_packages = {
        'tensorflow': 'tensorflow',
        'numpy': 'numpy', 
        'pandas': 'pandas',
        'scikit-learn': 'sklearn',  # パッケージ名とインポート名が異なる
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'psutil': 'psutil'
    }
    
    missing_packages = []
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✅ {package_name}: OK")
        except ImportError as e:
            missing_packages.append(package_name)
            print(f"❌ {package_name}: {e}")
    
    if missing_packages:
        print(f"\n❌ 以下のパッケージが見つかりません: {', '.join(missing_packages)}")
        print("pip install で必要なパッケージをインストールしてください")
        return False
    
    # TensorFlow GPU確認
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU利用可能: {len(gpus)} デバイス")
        else:
            print("⚠️  GPU不利用 (CPUモードで動作)")
    except Exception as e:
        print(f"⚠️  GPU確認エラー: {e}")
    
    # データファイル確認
    if not os.path.exists('config.json'):
        print("❌ config.json が見つかりません")
        return False
    
    config = load_config()
    data_file = config['data']['input_file']
    
    if not os.path.exists(data_file):
        print(f"❌ データファイルが見つかりません: {data_file}")
        return False
    
    print("✅ 環境チェック完了")
    return True

if __name__ == "__main__":
    print("USDJPYスキャルピング向けAI学習モデル 2025年版")
    print("開発環境: Python 3.8.20, TensorFlow-GPU 2.10.0")
    print()
    
    # 環境チェック
    if not check_requirements():
        print("環境に問題があります。修正してから再実行してください。")
        sys.exit(1)
    
    # メイン処理実行
    main()