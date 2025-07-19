"""
USDJPY スキャルピングEA パイプラインテスト
実際のティックデータを使用した動作検証
"""

import pandas as pd
import numpy as np
import logging
import os
import time
from datetime import datetime

# 自作モジュールのインポート
from utils import USDJPYUtils, setup_logging
from data_loader import TickDataLoader, load_sample_data
from feature_engineering import FeatureEngineer, create_sample_features
from labeling import ScalpingLabeler, create_sample_labels

# ログ設定
setup_logging(log_level=logging.INFO, log_file='pipeline_test.log')
logger = logging.getLogger(__name__)

class PipelineTester:
    """パイプライン統合テスト"""
    
    def __init__(self, data_path: str = "data/usdjpy_ticks.csv"):
        self.data_path = data_path
        self.utils = USDJPYUtils()
        self.loader = TickDataLoader()
        self.feature_engineer = FeatureEngineer()
        self.labeler = ScalpingLabeler()
        
    def check_data_file(self) -> bool:
        """データファイルの存在確認"""
        if not os.path.exists(self.data_path):
            logger.error(f"データファイルが見つかりません: {self.data_path}")
            return False
        
        file_size = os.path.getsize(self.data_path) / (1024 * 1024)  # MB
        logger.info(f"データファイル確認OK: {self.data_path} ({file_size:.1f} MB)")
        
        # CSVパターン判定
        pattern = self.utils.detect_csv_pattern(self.data_path)
        logger.info(f"検出されたCSVパターン: {pattern}")
        
        return True
    
    def test_small_sample(self, sample_size: int = 10000) -> dict:
        """小規模サンプルテスト"""
        logger.info(f"=== 小規模サンプルテスト開始 ({sample_size:,} 行) ===")
        
        results = {}
        start_time = time.time()
        
        try:
            # 1. データ読み込みテスト
            logger.info("1. データ読み込みテスト...")
            tick_df = self.loader.load_tick_data_auto(self.data_path, nrows=sample_size)
            results['tick_data_loaded'] = True
            results['tick_rows'] = len(tick_df)
            logger.info(f"ティックデータ読み込み完了: {len(tick_df):,} 行")
            
            # データサンプル表示
            logger.info("ティックデータサンプル:")
            logger.info(f"\n{tick_df.head()}")
            
            # 2. 1分足変換テスト
            logger.info("2. 1分足変換テスト...")
            ohlcv_df = self.loader.tick_to_ohlcv_1min(tick_df)
            results['ohlcv_converted'] = True
            results['ohlcv_rows'] = len(ohlcv_df)
            logger.info(f"1分足変換完了: {len(ohlcv_df):,} 本")
            
            # 1分足データサンプル表示
            logger.info("1分足データサンプル:")
            logger.info(f"\n{ohlcv_df.head()}")
            
            # 3. 特徴量生成テスト
            logger.info("3. 特徴量生成テスト...")
            features_df = self.feature_engineer.create_all_features(
                ohlcv_df, 
                include_advanced=True, 
                include_lags=False  # 小規模テストではラグなし
            )
            results['features_created'] = True
            results['feature_count'] = len(features_df.columns)
            logger.info(f"特徴量生成完了: {len(features_df.columns)} 列")
            
            # 特徴量サンプル表示
            key_features = ['close', 'rsi', 'macd', 'bb_percent_b', 'atr']
            available_features = [f for f in key_features if f in features_df.columns]
            logger.info("主要特徴量サンプル:")
            logger.info(f"\n{features_df[available_features].head().round(4)}")
            
            # 4. ラベル生成テスト
            logger.info("4. ラベル生成テスト...")
            labels = self.labeler.create_labels_vectorized(features_df)
            results['labels_created'] = True
            results['label_distribution'] = labels.value_counts().to_dict()
            logger.info(f"ラベル生成完了: {len(labels)} 行")
            
            # ラベル分布表示
            label_names = {0: 'NO_TRADE', 1: 'BUY', 2: 'SELL'}
            logger.info("ラベル分布:")
            for label_val, count in results['label_distribution'].items():
                percentage = count / len(labels) * 100
                logger.info(f"  {label_names[label_val]}: {count:,} ({percentage:.2f}%)")
            
            # 5. データ品質チェック
            logger.info("5. データ品質チェック...")
            
            # 欠損値チェック
            missing_counts = features_df.isna().sum()
            high_missing = missing_counts[missing_counts > len(features_df) * 0.1]  # 10%以上欠損
            
            results['high_missing_features'] = len(high_missing)
            if len(high_missing) > 0:
                logger.warning(f"高い欠損値を持つ特徴量 ({len(high_missing)} 個):")
                for feature, count in high_missing.head(5).items():
                    percentage = count / len(features_df) * 100
                    logger.warning(f"  {feature}: {count} ({percentage:.1f}%)")
            
            # 無限値チェック
            inf_counts = np.isinf(features_df.select_dtypes(include=[np.number])).sum()
            high_inf = inf_counts[inf_counts > 0]
            results['infinite_value_features'] = len(high_inf)
            
            if len(high_inf) > 0:
                logger.warning(f"無限値を含む特徴量: {list(high_inf.index)}")
            
            # 最終統合データフレーム作成
            final_df = features_df.copy()
            final_df['label'] = labels
            
            # 完全なデータのみ抽出（学習用）
            complete_data = final_df.dropna()
            results['complete_rows'] = len(complete_data)
            results['data_completeness'] = len(complete_data) / len(final_df) * 100
            
            logger.info(f"完全データ: {len(complete_data):,} 行 ({results['data_completeness']:.1f}%)")
            
            # 処理時間
            results['processing_time'] = time.time() - start_time
            results['success'] = True
            
            logger.info(f"小規模サンプルテスト完了 (処理時間: {results['processing_time']:.2f}秒)")
            
            return results
            
        except Exception as e:
            logger.error(f"小規模サンプルテストエラー: {e}")
            import traceback
            traceback.print_exc()
            results['success'] = False
            results['error'] = str(e)
            return results
    
    def test_medium_sample(self, sample_size: int = 100000) -> dict:
        """中規模サンプルテスト"""
        logger.info(f"=== 中規模サンプルテスト開始 ({sample_size:,} 行) ===")
        
        results = {}
        start_time = time.time()
        
        try:
            # メモリ使用量監視
            initial_memory = self.utils.memory_usage_mb()
            logger.info(f"初期メモリ使用量: {initial_memory:.1f} MB")
            
            # データ読み込み
            ohlcv_df = load_sample_data(self.data_path, sample_size)
            
            memory_after_load = self.utils.memory_usage_mb()
            logger.info(f"データ読み込み後メモリ: {memory_after_load:.1f} MB (+{memory_after_load - initial_memory:.1f} MB)")
            
            # 特徴量生成（高度な特徴量含む）
            features_df = self.feature_engineer.create_all_features(
                ohlcv_df, 
                include_advanced=True, 
                include_lags=True  # 中規模ではラグ特徴量も含む
            )
            
            memory_after_features = self.utils.memory_usage_mb()
            logger.info(f"特徴量生成後メモリ: {memory_after_features:.1f} MB (+{memory_after_features - memory_after_load:.1f} MB)")
            
            # ラベル生成
            labels = self.labeler.create_labels_vectorized(features_df)
            
            # 統計情報
            results['ohlcv_rows'] = len(ohlcv_df)
            results['feature_count'] = len(features_df.columns)
            results['label_distribution'] = labels.value_counts().to_dict()
            results['processing_time'] = time.time() - start_time
            results['peak_memory_mb'] = self.utils.memory_usage_mb()
            results['success'] = True
            
            logger.info(f"中規模サンプルテスト完了:")
            logger.info(f"  処理時間: {results['processing_time']:.2f}秒")
            logger.info(f"  ピークメモリ: {results['peak_memory_mb']:.1f} MB")
            logger.info(f"  特徴量数: {results['feature_count']}")
            
            return results
            
        except Exception as e:
            logger.error(f"中規模サンプルテストエラー: {e}")
            results['success'] = False
            results['error'] = str(e)
            return results
    
    def run_comprehensive_test(self):
        """包括的テスト実行"""
        logger.info("=== USDJPY スキャルピングEA パイプライン包括テスト ===")
        
        # 1. ファイル確認
        if not self.check_data_file():
            logger.error("テスト中止: データファイルが見つかりません")
            return
        
        # 2. 小規模テスト
        small_results = self.test_small_sample(10000)
        if not small_results.get('success', False):
            logger.error("小規模テスト失敗 - テスト中止")
            return
        
        # 3. 中規模テスト
        medium_results = self.test_medium_sample(100000)
        if not medium_results.get('success', False):
            logger.warning("中規模テスト失敗 - 大規模テストはスキップ")
        
        # 4. 結果サマリー
        logger.info("=== テスト結果サマリー ===")
        logger.info(f"小規模テスト: {'成功' if small_results['success'] else '失敗'}")
        logger.info(f"中規模テスト: {'成功' if medium_results.get('success', False) else '失敗'}")
        
        if small_results['success']:
            logger.info(f"データ処理性能:")
            logger.info(f"  小規模 ({small_results['tick_rows']:,} ティック → {small_results['ohlcv_rows']:,} 1分足)")
            logger.info(f"  処理時間: {small_results['processing_time']:.2f}秒")
            logger.info(f"  特徴量数: {small_results['feature_count']}")
            
            # ラベル分布
            label_names = {0: 'NO_TRADE', 1: 'BUY', 2: 'SELL'}
            logger.info(f"ラベル分布:")
            total_labels = sum(small_results['label_distribution'].values())
            for label_val, count in small_results['label_distribution'].items():
                percentage = count / total_labels * 100
                logger.info(f"  {label_names[label_val]}: {percentage:.1f}%")
        
        if medium_results.get('success', False):
            logger.info(f"中規模テスト:")
            logger.info(f"  処理時間: {medium_results['processing_time']:.2f}秒")
            logger.info(f"  ピークメモリ: {medium_results['peak_memory_mb']:.1f} MB")
        
        logger.info("=== パイプラインテスト完了 ===")


def main():
    """メイン実行関数"""
    print("USDJPY スキャルピングEA - パイプライン動作テスト")
    print("=" * 50)
    
    # データファイルパス確認
    data_path = "data/usdjpy_ticks.csv"
    if not os.path.exists(data_path):
        print(f"エラー: データファイルが見つかりません - {data_path}")
        print("ファイルが正しい場所にあることを確認してください。")
        return
    
    # テスト実行
    tester = PipelineTester(data_path)
    tester.run_comprehensive_test()
    
    print("\nテスト完了! 詳細はログファイル 'pipeline_test.log' を確認してください。")


if __name__ == "__main__":
    main()