#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
USDJPY スキャルピングEA用 AI学習モデル構築
メインスクリプト

作成日: 2025年
目的: 堅牢で破綻しない設計による高精度スキャルピングモデルの構築
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import json
import numpy as np
import pandas as pd

# モジュールのインポート
from modules.utils import (
    load_config, setup_logging, setup_gpu_memory_growth, 
    create_directories, memory_usage_mb
)
from modules.data_loader import TickDataLoader
from modules.feature_engineering import FeatureEngine
from modules.labeling import LabelEngine
from modules.train import ModelTrainer

def main():
    """メイン実行関数"""
    
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='USDJPYスキャルピングAIモデル構築')
    parser.add_argument('--config', type=str, default='config.json', 
                       help='設定ファイルパス')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'train_only', 'evaluate_only', 'predict'],
                       help='実行モード')
    parser.add_argument('--data_file', type=str, default=None,
                       help='入力データファイルパス（設定ファイルの値を上書き）')
    parser.add_argument('--sample_mode', action='store_true',
                       help='サンプルデータでの実行（開発・テスト用）')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='ログレベル')
    
    args = parser.parse_args()
    
    # ログ設定
    logger = setup_logging(args.log_level)
    logger.info("="*60)
    logger.info("USDJPYスキャルピングAIモデル構築開始")
    logger.info("="*60)
    
    try:
        # 設定読み込み
        config = load_config(args.config)
        logger.info(f"設定ファイル読み込み完了: {args.config}")
        
        # データファイルパス上書き
        if args.data_file:
            config['data']['input_file'] = args.data_file
            logger.info(f"データファイル指定: {args.data_file}")
        
        # ディレクトリ作成
        create_directories(config)
        
        # GPU設定
        setup_gpu_memory_growth()
        
        # 実行モードに応じて処理分岐
        if args.mode == 'full':
            run_full_pipeline(config, args.sample_mode)
        elif args.mode == 'train_only':
            run_training_only(config, args.sample_mode)
        elif args.mode == 'evaluate_only':
            run_evaluation_only(config)
        elif args.mode == 'predict':
            run_prediction_mode(config)
        
        logger.info("="*60)
        logger.info("処理完了")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"処理中にエラーが発生: {e}")
        logger.exception("詳細なエラー情報:")
        sys.exit(1)

def run_full_pipeline(config, sample_mode=False):
    """フルパイプライン実行"""
    logger = logging.getLogger(__name__)
    logger.info("フルパイプライン実行開始")
    
    # 1. データ読み込み
    logger.info("ステップ1: データ読み込み")
    data_loader = TickDataLoader(config)
    
    if sample_mode:
        logger.info("サンプルモードで実行")
        df = data_loader.load_tick_data()
        df = data_loader.create_sample_dataset(df, sample_ratio=0.1)
    else:
        df = data_loader.load_tick_data()
    
    # データ統計情報
    stats = data_loader.get_data_statistics(df)
    logger.info(f"データ統計: {json.dumps(stats, indent=2, ensure_ascii=False)}")
    
    # 2. 特徴量エンジニアリング
    logger.info("ステップ2: 特徴量エンジニアリング")
    feature_engine = FeatureEngine(config)
    df = feature_engine.calculate_technical_indicators(df)
    logger.info(f"特徴量計算完了 - メモリ使用量: {memory_usage_mb():.1f}MB")
    
    # 3. ラベリング
    logger.info("ステップ3: ラベリング")
    label_engine = LabelEngine(config)
    df = label_engine.generate_labels(df)
    
    # ラベル検証
    label_validation = label_engine.validate_labels(df)
    if not label_validation['is_balanced']:
        logger.warning("ラベル分布が不均衡です。クラス重み調整を推奨します。")
    
    # 4. シーケンスデータ作成（スケーリング一貫性確保）
    logger.info("ステップ4: シーケンスデータ作成")
    X, y, feature_names, timestamps = feature_engine.create_sequences(df, is_training=True)
    logger.info(f"シーケンス作成完了: X.shape={X.shape}, y.shape={y.shape}")
    
    # 5. モデル訓練
    logger.info("ステップ5: モデル訓練")
    trainer = ModelTrainer(config)
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(X, y)
    
    # スケーリングパラメータをトレーナーに渡す
    trainer.scaling_params = feature_engine.scaling_params
    
    # 訓練実行
    model_path = trainer.train_model(X_train, y_train, X_val, y_val, feature_names)
    
    # スケーリングパラメータを保存
    scaling_params_path = model_path.replace('.h5', '_scaling_params.json')
    feature_engine.save_scaling_params(scaling_params_path)
    
    # 6. モデル評価
    logger.info("ステップ6: モデル評価")
    evaluation_results = trainer.evaluate_model(X_test, y_test)
    
    # 7. 結果保存・可視化
    logger.info("ステップ7: 結果保存")
    save_results(trainer, evaluation_results, config, timestamps)
    
    logger.info("フルパイプライン完了")

def run_training_only(config, sample_mode=False):
    """訓練のみ実行（特徴量・ラベルが既に作成済みの場合）"""
    logger = logging.getLogger(__name__)
    logger.info("訓練のみ実行")
    
    # 前処理済みデータの読み込み
    processed_data_path = config['data']['input_file'].replace('.csv', '_processed.csv')
    
    if not os.path.exists(processed_data_path):
        logger.error("前処理済みデータが見つかりません。フルパイプラインを実行してください。")
        return
    
    # データ読み込み・処理（簡略版）
    data_loader = TickDataLoader(config)
    feature_engine = FeatureEngine(config)
    
    df = data_loader.load_tick_data(processed_data_path, validate=False)
    X, y, feature_names, timestamps = feature_engine.create_sequences(df)
    
    # 訓練実行
    trainer = ModelTrainer(config)
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(X, y)
    model_path = trainer.train_model(X_train, y_train, X_val, y_val, feature_names)
    
    # 評価
    evaluation_results = trainer.evaluate_model(X_test, y_test)
    save_results(trainer, evaluation_results, config, timestamps)

def run_evaluation_only(config):
    """評価のみ実行（訓練済みモデルがある場合）"""
    logger = logging.getLogger(__name__)
    logger.info("評価のみ実行")
    
    # 最新のモデルファイルを検索
    models_dir = config['data']['output_dir']
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
    
    if not model_files:
        logger.error("訓練済みモデルが見つかりません。")
        return
    
    # 最新のモデルを選択
    latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(models_dir, x)))
    model_path = os.path.join(models_dir, latest_model)
    
    logger.info(f"評価対象モデル: {model_path}")
    
    try:
        # メタデータ読み込み（エラーハンドリング）
        feature_names = None
        try:
            from modules.utils import load_model_metadata
            metadata = load_model_metadata(model_path)
            if metadata and 'feature_names' in metadata:
                feature_names = metadata['feature_names']
                logger.info(f"メタデータから特徴量名を取得: {len(feature_names)}個")
        except Exception as e:
            logger.warning(f"メタデータ読み込みエラー（無視して続行）: {e}")
        
        # データ読み込み・前処理（元の訓練データを使用）
        logger.info("評価用データ準備")
        data_loader = TickDataLoader(config)
        df = data_loader.load_tick_data()
        df = data_loader.create_sample_dataset(df, sample_ratio=0.1)  # サンプルモード相当
        
        # 特徴量エンジニアリング
        feature_engine = FeatureEngine(config)
        df = feature_engine.calculate_technical_indicators(df)
        
        # ラベリング
        label_engine = LabelEngine(config)
        df = label_engine.generate_labels(df)
        
        # シーケンス作成
        X, y, feature_names, timestamps = feature_engine.create_sequences(df)
        logger.info(f"評価データ準備完了: X.shape={X.shape}, y.shape={y.shape}")
        
        # データ分割（テストデータのみ使用）
        trainer = ModelTrainer(config)
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(X, y)
        
        # 訓練済みモデル読み込み
        import tensorflow as tf
        trained_model = tf.keras.models.load_model(model_path)
        trainer.model_wrapper.model = trained_model  # モデルを設定
        
        logger.info("モデル読み込み完了")
        
        # 評価実行
        logger.info("モデル評価開始")
        evaluation_results = trainer.evaluate_model(X_test, y_test)
        
        # 結果保存・可視化
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_results(trainer, evaluation_results, config, timestamps[-len(X_test):])
        
        # サマリー表示
        logger.info("="*60)
        logger.info("🎯 モデル評価結果サマリー")
        logger.info("="*60)
        logger.info(f"📊 基本性能:")
        logger.info(f"   テスト精度: {evaluation_results['test_accuracy']:.1%}")
        logger.info(f"   テスト損失: {evaluation_results['test_loss']:.4f}")
        
        logger.info(f"\n📈 クラス別F1スコア:")
        for class_name, f1 in evaluation_results['f1_scores'].items():
            logger.info(f"   {class_name:8}: {f1:.3f}")
        
        # 予測分布の確認
        y_pred = evaluation_results['predictions']['y_pred']
        pred_counts = np.bincount(y_pred)
        class_names = config['labels']['class_names']
        logger.info(f"\n🔍 予測分布（問題診断用）:")
        for i, count in enumerate(pred_counts):
            if i < len(class_names):
                logger.info(f"   {class_names[i]:8}: {count:,} ({count/len(y_pred)*100:.1f}%)")
        
        # 予測信頼度の統計
        y_proba = np.array(evaluation_results['predictions']['y_pred_proba'])
        max_confidences = np.max(y_proba, axis=1)
        logger.info(f"\n📊 予測信頼度統計:")
        logger.info(f"   最小信頼度: {np.min(max_confidences):.3f}")
        logger.info(f"   最大信頼度: {np.max(max_confidences):.3f}")
        logger.info(f"   平均信頼度: {np.mean(max_confidences):.3f}")
        logger.info(f"   信頼度0.6以上: {np.sum(max_confidences >= 0.6):,} ({np.sum(max_confidences >= 0.6)/len(max_confidences)*100:.1f}%)")
        
        # 閾値別性能表示（主要な閾値のみ）
        logger.info(f"\n📋 閾値別性能:")
        key_thresholds = ['0.1', '0.3', '0.5', '0.7', '0.9']
        for threshold in key_thresholds:
            if threshold in evaluation_results['threshold_evaluation']:
                metrics = evaluation_results['threshold_evaluation'][threshold]
                logger.info(f"   閾値{threshold}: 精度={metrics['accuracy']:.3f}, "
                           f"F1={metrics['f1_score']:.3f}, カバレッジ={metrics['coverage']:.3f}")
        
        # 最適閾値の推奨
        best_threshold = get_best_threshold(evaluation_results['threshold_evaluation'])
        logger.info(f"\n💡 推奨信頼度閾値: {best_threshold}")
        
        logger.info("="*60)
        logger.info("📁 出力ファイルが models/evaluation_report/ に保存されました")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"評価中にエラーが発生: {e}")
        logger.exception("詳細なエラー情報:")
    
    logger.info("評価のみ実行完了")

def run_prediction_mode(config):
    """予測モード（リアルタイム予測用）"""
    logger = logging.getLogger(__name__)
    logger.info("予測モード実行")
    
    # 最新のモデル読み込み
    models_dir = config['data']['output_dir']
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
    
    if not model_files:
        logger.error("訓練済みモデルが見つかりません。")
        return
    
    latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(models_dir, x)))
    model_path = os.path.join(models_dir, latest_model)
    
    logger.info(f"予測用モデル: {model_path}")
    
    # リアルタイム予測システムの構築
    # MT5連携用のCSV出力機能
    # ... 実装内容は省略（実際の使用時に詳細実装）
    
    logger.info("予測モード完了")

def save_results(trainer, evaluation_results, config, timestamps):
    """結果保存・可視化"""
    logger = logging.getLogger(__name__)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 出力ディレクトリを models/evaluation_report/ に設定
    output_dir = os.path.join(config['data']['output_dir'], 'evaluation_report')
    os.makedirs(output_dir, exist_ok=True)  # ディレクトリ作成
    
    logger.info(f"評価結果保存先: {output_dir}")
    
    # 評価レポート保存
    report_path = os.path.join(output_dir, f"evaluation_report_{timestamp}.json")
    trainer.save_evaluation_report(evaluation_results, report_path)
    
    # 可視化
    try:
        # 訓練履歴グラフ（もしあれば）
        if hasattr(trainer, 'training_history') and trainer.training_history:
            history_plot_path = os.path.join(output_dir, f"training_history_{timestamp}.png")
            trainer.plot_training_history(history_plot_path)
        
        # 混同行列
        cm_plot_path = os.path.join(output_dir, f"confusion_matrix_{timestamp}.png")
        cm = evaluation_results['confusion_matrix']
        trainer.plot_confusion_matrix(np.array(cm), cm_plot_path)
        
        # 閾値分析
        threshold_plot_path = os.path.join(output_dir, f"threshold_analysis_{timestamp}.png")
        trainer.plot_threshold_analysis(
            evaluation_results['threshold_evaluation'], 
            threshold_plot_path
        )
        
        # オーダー数分布グラフ（新機能）
        order_dist_plot_path = os.path.join(output_dir, f"order_distribution_{timestamp}.png")
        order_distribution_df = trainer.plot_order_distribution_by_class(
            evaluation_results,
            order_dist_plot_path
        )
        
        # オーダー分布データもCSVで保存
        order_dist_csv_path = os.path.join(output_dir, f"order_distribution_{timestamp}.csv")
        order_distribution_df.to_csv(order_dist_csv_path, index=False)
        logger.info(f"オーダー分布データ保存: {order_dist_csv_path}")
        
    except Exception as e:
        logger.warning(f"可視化でエラーが発生: {e}")
    
    # MT5連携用シグナルファイル出力
    if config['evaluation']['export_format'] == 'csv':
        try:
            # 予測結果をシグナル形式に変換
            if len(timestamps) > 0:
                # テストデータのシンプルな予測結果を出力
                signals_path = os.path.join(output_dir, f"trading_signals_{timestamp}.csv")
                
                # 簡単なシグナルファイル作成
                y_pred = evaluation_results['predictions']['y_pred']
                y_prob = evaluation_results['predictions']['y_pred_proba']
                class_names = config['labels']['class_names']
                
                signals_data = []
                for i, (pred, prob) in enumerate(zip(y_pred, y_prob)):
                    if i < len(timestamps):
                        timestamp_val = timestamps[i] if hasattr(timestamps[i], 'isoformat') else str(timestamps[i])
                        signals_data.append({
                            'timestamp': timestamp_val,
                            'signal': class_names[pred],
                            'confidence': max(prob),
                            'buy_probability': prob[0],
                            'sell_probability': prob[1],
                            'no_trade_probability': prob[2],
                            'trade_recommended': max(prob) >= config['evaluation']['min_confidence'] and pred != 2
                        })
                
                signals_df = pd.DataFrame(signals_data)
                signals_df.to_csv(signals_path, index=False)
                logger.info(f"取引シグナル出力: {signals_path}")
                
        except Exception as e:
            logger.warning(f"シグナル出力でエラーが発生: {e}")
    
    # パフォーマンスサマリー出力
    performance_summary = {
        'timestamp': timestamp,
        'test_accuracy': evaluation_results['test_accuracy'],
        'test_loss': evaluation_results['test_loss'],
        'best_threshold': get_best_threshold(evaluation_results['threshold_evaluation']),
        'f1_scores': evaluation_results['f1_scores'],
        'threshold_performance': evaluation_results['threshold_evaluation'],
        'model_path': trainer.model_path if hasattr(trainer, 'model_path') else 'unknown',
        'memory_usage_mb': memory_usage_mb()
    }
    
    summary_path = os.path.join(output_dir, f"performance_summary_{timestamp}.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(performance_summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"結果保存完了 - 出力ディレクトリ: {output_dir}")
    logger.info(f"生成ファイル:")
    logger.info(f"  - 評価レポート: evaluation_report_{timestamp}.json")
    logger.info(f"  - 混同行列: confusion_matrix_{timestamp}.png")
    logger.info(f"  - 閾値分析: threshold_analysis_{timestamp}.png")
    logger.info(f"  - オーダー分布グラフ: order_distribution_{timestamp}.png")
    logger.info(f"  - オーダー分布データ: order_distribution_{timestamp}.csv")
    logger.info(f"  - パフォーマンス要約: performance_summary_{timestamp}.json")
    if config['evaluation']['export_format'] == 'csv':
        logger.info(f"  - 取引シグナル: trading_signals_{timestamp}.csv")

def get_best_threshold(threshold_results):
    """最適な閾値を取得"""
    best_threshold = 0.5
    best_f1 = 0.0
    
    for threshold, metrics in threshold_results.items():
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_threshold = float(threshold)
    
    return best_threshold

if __name__ == "__main__":
    main()