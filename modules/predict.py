#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
USDJPYスキャルピング リアルタイム予測システム
MT5連携用予測スクリプト

機能:
- 訓練済みモデルによるリアルタイム予測
- MT5で読み取り可能なCSV/JSON出力
- 信頼度フィルタリング
- 連続予測モード対応
"""

import os
import sys
import json
import argparse
import logging
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Dict, Any, List, Optional

from modules.utils import load_config, setup_logging, load_model_metadata
from modules.data_loader import TickDataLoader
from modules.feature_engineering import FeatureEngine
from modules.model import ScalpingModel

class RealTimePredictorSystem:
    """
    リアルタイム予測システム
    MT5との連携を想定した設計
    """
    
    def __init__(self, config_path: str, model_path: str):
        self.config = load_config(config_path)
        self.model_path = model_path
        
        # ログ設定
        self.logger = setup_logging('INFO')
        
        # モデル・エンジン初期化
        self.model_wrapper = ScalpingModel(self.config)
        self.feature_engine = FeatureEngine(self.config)
        self.data_loader = TickDataLoader(self.config)
        
        # 設定値
        self.min_confidence = self.config['evaluation']['min_confidence']
        self.sequence_length = self.config['features']['sequence_length']
        self.export_format = self.config['evaluation']['export_format']
        
        # モデル読み込み
        self._load_trained_model()
        
        # 特徴量の履歴バッファ（シーケンス作成用）
        self.feature_buffer = []
        
        self.logger.info("リアルタイム予測システム初期化完了")
    
    def _load_trained_model(self):
        """訓練済みモデルの読み込み"""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            self.model_metadata = load_model_metadata(self.model_path)
            
            self.logger.info(f"モデル読み込み完了: {self.model_path}")
            self.logger.info(f"モデルパラメータ数: {self.model.count_params():,}")
            
            # モデル情報の表示
            if self.model_metadata:
                created_at = self.model_metadata.get('created_at', 'Unknown')
                self.logger.info(f"モデル作成日時: {created_at}")
                
        except Exception as e:
            self.logger.error(f"モデル読み込みエラー: {e}")
            raise
    
    def predict_single_tick(self, tick_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        単一ティックデータの予測
        
        Args:
            tick_data: {'datetime': datetime, 'BID': float, 'ASK': float}
        
        Returns:
            予測結果辞書
        """
        try:
            # ティックデータをDataFrameに変換
            df = pd.DataFrame([tick_data])
            
            # 特徴量計算
            df = self.feature_engine.calculate_technical_indicators(df)
            
            # 特徴量バッファに追加
            feature_row = self._extract_feature_vector(df.iloc[0])
            self.feature_buffer.append(feature_row)
            
            # シーケンス長に達していない場合
            if len(self.feature_buffer) < self.sequence_length:
                return {
                    'status': 'insufficient_data',
                    'buffer_size': len(self.feature_buffer),
                    'required_size': self.sequence_length,
                    'signal': 'NO_TRADE',
                    'confidence': 0.0
                }
            
            # バッファサイズ制限
            if len(self.feature_buffer) > self.sequence_length:
                self.feature_buffer = self.feature_buffer[-self.sequence_length:]
            
            # シーケンス作成
            sequence = np.array(self.feature_buffer)
            sequence = sequence.reshape(1, self.sequence_length, -1)
            
            # 予測実行
            prediction = self.model.predict(sequence, verbose=0)[0]
            
            # 結果解析
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)
            
            class_names = self.config['labels']['class_names']
            signal = class_names[predicted_class]
            
            # 信頼度フィルタリング
            if confidence < self.min_confidence:
                signal = 'NO_TRADE'
            
            result = {
                'status': 'success',
                'timestamp': tick_data['datetime'].isoformat(),
                'signal': signal,
                'confidence': float(confidence),
                'probabilities': {
                    'BUY': float(prediction[0]),
                    'SELL': float(prediction[1]),
                    'NO_TRADE': float(prediction[2])
                },
                'trade_recommended': confidence >= self.min_confidence and predicted_class != 2,
                'mid_price': (tick_data['BID'] + tick_data['ASK']) / 2
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"予測エラー: {e}")
            return {
                'status': 'error',
                'error_message': str(e),
                'signal': 'NO_TRADE',
                'confidence': 0.0
            }
    
    def _extract_feature_vector(self, row: pd.Series) -> np.ndarray:
        """行から特徴量ベクトルを抽出"""
        feature_columns = []
        
        # 基本特徴量
        basic_features = ['mid_price', 'spread', 'price_change', 'price_change_pct']
        feature_columns.extend(basic_features)
        
        # 移動平均
        for period in self.config['features']['sma_periods']:
            feature_columns.extend([f'sma_{period}', f'price_vs_sma_{period}'])
        
        for period in self.config['features']['ema_periods']:
            feature_columns.extend([f'ema_{period}', f'price_vs_ema_{period}'])
        
        # テクニカル指標
        technical_features = [
            'bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
            'macd_line', 'macd_signal', 'macd_histogram',
            'rsi', 'atr', 'cci'
        ]
        feature_columns.extend(technical_features)
        
        # 存在する特徴量のみ抽出
        available_features = [col for col in feature_columns if col in row.index]
        feature_vector = row[available_features].fillna(0).values
        
        return feature_vector.astype(np.float32)
    
    def process_csv_file(self, input_csv_path: str, output_path: str):
        """
        CSVファイル一括処理
        """
        self.logger.info(f"CSV一括処理開始: {input_csv_path}")
        
        # データ読み込み
        df = self.data_loader.load_tick_data(input_csv_path)
        
        # 予測結果を格納するリスト
        predictions = []
        
        # 各ティックを順次処理
        for i, row in df.iterrows():
            if i % 1000 == 0:
                self.logger.info(f"処理進捗: {i}/{len(df)}")
            
            tick_data = {
                'datetime': row['datetime'],
                'BID': row['BID'],
                'ASK': row['ASK']
            }
            
            result = self.predict_single_tick(tick_data)
            predictions.append(result)
        
        # 結果をDataFrameに変換
        results_df = pd.DataFrame(predictions)
        
        # 出力
        if self.export_format == 'csv':
            results_df.to_csv(output_path, index=False)
        else:
            results_df.to_json(output_path, orient='records', date_format='iso')
        
        self.logger.info(f"処理完了: {output_path}")
        
        # 統計情報
        total_signals = len(results_df)
        trade_signals = results_df['trade_recommended'].sum()
        avg_confidence = results_df['confidence'].mean()
        
        self.logger.info(f"予測統計:")
        self.logger.info(f"  総予測数: {total_signals:,}")
        self.logger.info(f"  取引推奨: {trade_signals:,} ({trade_signals/total_signals*100:.1f}%)")
        self.logger.info(f"  平均信頼度: {avg_confidence:.3f}")
        
        return results_df
    
    def start_continuous_prediction(self, data_source: str, output_dir: str, 
                                  update_interval: int = 1):
        """
        連続予測モード（リアルタイム監視）
        
        Args:
            data_source: データソースパス（CSV監視 or API接続）
            output_dir: 出力ディレクトリ
            update_interval: 更新間隔（秒）
        """
        self.logger.info("連続予測モード開始")
        
        while True:
            try:
                # 新しいデータの確認
                if self._check_new_data(data_source):
                    # 最新データ取得
                    latest_data = self._get_latest_data(data_source)
                    
                    if latest_data:
                        # 予測実行
                        result = self.predict_single_tick(latest_data)
                        
                        # 結果出力
                        self._output_prediction_result(result, output_dir)
                        
                        # ログ出力
                        if result['trade_recommended']:
                            self.logger.info(f"取引シグナル: {result['signal']} "
                                           f"(信頼度: {result['confidence']:.3f})")
                
                time.sleep(update_interval)
                
            except KeyboardInterrupt:
                self.logger.info("連続予測モード停止")
                break
            except Exception as e:
                self.logger.error(f"連続予測エラー: {e}")
                time.sleep(update_interval * 2)  # エラー時は少し長めに待機
    
    def _check_new_data(self, data_source: str) -> bool:
        """新しいデータの存在確認"""
        # 実装は省略（実際の運用時に詳細実装）
        return True
    
    def _get_latest_data(self, data_source: str) -> Optional[Dict[str, Any]]:
        """最新データ取得"""
        # 実装は省略（実際の運用時に詳細実装）
        return None
    
    def _output_prediction_result(self, result: Dict[str, Any], output_dir: str):
        """予測結果出力"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if result['trade_recommended']:
            # 取引推奨の場合のみファイル出力
            filename = f"signal_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description='USDJPYスキャルピング予測システム')
    parser.add_argument('--config', type=str, default='config.json',
                       help='設定ファイルパス')
    parser.add_argument('--model', type=str, required=True,
                       help='訓練済みモデルファイルパス')
    parser.add_argument('--mode', type=str, default='batch',
                       choices=['batch', 'continuous', 'single'],
                       help='実行モード')
    parser.add_argument('--input', type=str,
                       help='入力ファイルパス（batchモード用）')
    parser.add_argument('--output', type=str,
                       help='出力ファイルパス')
    parser.add_argument('--output_dir', type=str, default='predictions',
                       help='出力ディレクトリ（continuousモード用）')
    
    args = parser.parse_args()
    
    # 出力ディレクトリ作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # 予測システム初期化
        predictor = RealTimePredictorSystem(args.config, args.model)
        
        if args.mode == 'batch':
            if not args.input or not args.output:
                print("batchモードには--inputと--outputが必要です")
                return
            
            # 一括処理
            predictor.process_csv_file(args.input, args.output)
            
        elif args.mode == 'continuous':
            # 連続予測
            data_source = args.input if args.input else "default_source"
            predictor.start_continuous_prediction(data_source, args.output_dir)
            
        elif args.mode == 'single':
            # 単一予測テスト
            test_data = {
                'datetime': datetime.now(),
                'BID': 157.050,
                'ASK': 157.070
            }
            result = predictor.predict_single_tick(test_data)
            print(json.dumps(result, indent=2, ensure_ascii=False))
    
    except Exception as e:
        print(f"エラー: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()