#!/usr/bin/env python3
"""
USDJPYスキャルピング向けAI推論スクリプト
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import json
import argparse
from datetime import datetime
import tensorflow as tf

# 警告を抑制
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.utils import load_config, load_tick_data
from modules.feature_engineering import FeatureEngineer

class ForexPredictor:
    """FX予測クラス"""
    
    def __init__(self, model_path, config_path="config.json", normalization_params_path=None):
        """
        初期化
        
        Args:
            model_path: 学習済みモデルのパス
            config_path: 設定ファイルのパス
            normalization_params_path: 正規化パラメータファイルのパス
        """
        self.config = load_config(config_path)
        self.feature_engineer = FeatureEngineer(self.config)
        
        # モデル読み込み
        print(f"モデル読み込み中: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        print("モデル読み込み完了")
        
        # 正規化パラメータ読み込み
        if normalization_params_path and os.path.exists(normalization_params_path):
            with open(normalization_params_path, 'r') as f:
                self.norm_params = json.load(f)
            self.norm_mean = np.array(self.norm_params['mean'])
            self.norm_std = np.array(self.norm_params['std'])
            self.feature_names = self.norm_params.get('feature_names', [])
            print("正規化パラメータ読み込み完了")
        else:
            print("警告: 正規化パラメータが見つかりません。正規化をスキップします。")
            self.norm_params = None
        
        # ラベル名
        self.label_names = {0: 'BUY', 1: 'SELL', 2: 'NO_TRADE'}
        
        # 閾値設定
        self.trade_threshold = self.config['trading']['trade_threshold']
        
    def predict_single(self, tick_data):
        """
        単一データポイントの予測
        
        Args:
            tick_data: ティックデータ DataFrame（最新のfeature_window分を含む）
            
        Returns:
            dict: 予測結果
        """
        if len(tick_data) < self.config['data']['feature_window']:
            raise ValueError(f"データが不足しています。最低{self.config['data']['feature_window']}件必要です。")
        
        # 特徴量計算
        df_with_indicators = self.feature_engineer.calculate_indicators(tick_data)
        features, _ = self.feature_engineer.create_features(df_with_indicators)
        
        # 最新のfeature_window分を取得
        feature_window = self.config['data']['feature_window']
        latest_features = features[-feature_window:]
        
        # シーケンス形状に変換 (1, timesteps, features)
        X = np.expand_dims(latest_features, axis=0)
        
        # 正規化
        if self.norm_params is not None:
            X = (X - self.norm_mean) / self.norm_std
        
        # 予測実行
        predictions = self.model.predict(X, verbose=0)[0]
        
        # 結果解析
        predicted_class = np.argmax(predictions)
        confidence = float(predictions[predicted_class])
        
        # 閾値判定
        final_prediction = predicted_class if confidence >= self.trade_threshold else 2  # NO_TRADE
        
        result = {
            'prediction': int(final_prediction),
            'prediction_name': self.label_names[final_prediction],
            'confidence': confidence,
            'probabilities': {
                'BUY': float(predictions[0]),
                'SELL': float(predictions[1]),
                'NO_TRADE': float(predictions[2])
            },
            'raw_prediction': int(predicted_class),
            'raw_prediction_name': self.label_names[predicted_class],
            'threshold_applied': confidence < self.trade_threshold,
            'current_price': float(tick_data['MID'].iloc[-1]),
            'spread': float(tick_data['ASK'].iloc[-1] - tick_data['BID'].iloc[-1]),
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def predict_batch(self, tick_data, start_idx=None, end_idx=None):
        """
        バッチ予測
        
        Args:
            tick_data: ティックデータ DataFrame
            start_idx: 開始インデックス
            end_idx: 終了インデックス
            
        Returns:
            list: 予測結果のリスト
        """
        feature_window = self.config['data']['feature_window']
        
        if len(tick_data) < feature_window:
            raise ValueError(f"データが不足しています。最低{feature_window}件必要です。")
        
        # インデックス設定
        if start_idx is None:
            start_idx = feature_window
        if end_idx is None:
            end_idx = len(tick_data)
        
        # 特徴量計算
        df_with_indicators = self.feature_engineer.calculate_indicators(tick_data)
        features, _ = self.feature_engineer.create_features(df_with_indicators)
        
        # シーケンス作成
        X_sequences = []
        valid_indices = []
        
        for i in range(start_idx, min(end_idx, len(features))):
            if i >= feature_window:
                X_sequences.append(features[i-feature_window:i])
                valid_indices.append(i)
        
        if not X_sequences:
            return []
        
        X = np.array(X_sequences)
        
        # 正規化
        if self.norm_params is not None:
            X = (X - self.norm_mean) / self.norm_std
        
        # バッチ予測
        predictions = self.model.predict(X, verbose=0)
        
        results = []
        for i, (pred, data_idx) in enumerate(zip(predictions, valid_indices)):
            predicted_class = np.argmax(pred)
            confidence = float(pred[predicted_class])
            
            # 閾値判定
            final_prediction = predicted_class if confidence >= self.trade_threshold else 2
            
            result = {
                'index': data_idx,
                'prediction': int(final_prediction),
                'prediction_name': self.label_names[final_prediction],
                'confidence': confidence,
                'probabilities': {
                    'BUY': float(pred[0]),
                    'SELL': float(pred[1]),
                    'NO_TRADE': float(pred[2])
                },
                'raw_prediction': int(predicted_class),
                'raw_prediction_name': self.label_names[predicted_class],
                'threshold_applied': confidence < self.trade_threshold,
                'current_price': float(tick_data['MID'].iloc[data_idx]),
                'spread': float(tick_data['ASK'].iloc[data_idx] - tick_data['BID'].iloc[data_idx]),
                'date': tick_data['DATE'].iloc[data_idx] if 'DATE' in tick_data.columns else None,
                'time': tick_data['TIME'].iloc[data_idx] if 'TIME' in tick_data.columns else None
            }
            results.append(result)
        
        return results
    
    def get_trading_signal(self, tick_data):
        """
        トレーディングシグナル取得（実取引用）
        
        Args:
            tick_data: ティックデータ DataFrame
            
        Returns:
            dict: トレーディングシグナル
        """
        prediction_result = self.predict_single(tick_data)
        
        # トレーディングパラメータ
        current_price = prediction_result['current_price']
        pip_value = self.config['trading']['pip_value']
        tp_pips = self.config['trading']['tp_pips']
        sl_pips = self.config['trading']['sl_pips']
        spread_pips = self.config['trading']['spread_pips']
        
        signal = {
            'action': prediction_result['prediction_name'],
            'confidence': prediction_result['confidence'],
            'current_price': current_price,
            'entry_price': None,
            'tp_price': None,
            'sl_price': None,
            'spread_cost_pips': spread_pips,
            'expected_risk_reward': None,
            'timestamp': prediction_result['timestamp']
        }
        
        if prediction_result['prediction'] == 0:  # BUY
            signal['entry_price'] = current_price + (spread_pips * pip_value / 2)  # ASK価格相当
            signal['tp_price'] = signal['entry_price'] + (tp_pips * pip_value)
            signal['sl_price'] = signal['entry_price'] - (sl_pips * pip_value)
            signal['expected_risk_reward'] = tp_pips / sl_pips
            
        elif prediction_result['prediction'] == 1:  # SELL
            signal['entry_price'] = current_price - (spread_pips * pip_value / 2)  # BID価格相当
            signal['tp_price'] = signal['entry_price'] - (tp_pips * pip_value)
            signal['sl_price'] = signal['entry_price'] + (sl_pips * pip_value)
            signal['expected_risk_reward'] = tp_pips / sl_pips
        
        return signal

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='FX予測実行')
    parser.add_argument('--model', '-m', required=True, help='学習済みモデルファイルパス')
    parser.add_argument('--data', '-d', help='予測対象データファイル（省略時はconfig.jsonのinput_fileを使用）')
    parser.add_argument('--config', '-c', default='config.json', help='設定ファイルパス')
    parser.add_argument('--norm-params', '-n', help='正規化パラメータファイルパス')
    parser.add_argument('--output', '-o', help='結果出力ファイルパス')
    parser.add_argument('--start-idx', type=int, help='予測開始インデックス')
    parser.add_argument('--end-idx', type=int, help='予測終了インデックス')
    parser.add_argument('--single', action='store_true', help='最新データのみ予測')
    parser.add_argument('--signal', action='store_true', help='トレーディングシグナル出力')
    
    args = parser.parse_args()
    
    try:
        # 予測器初期化
        print("予測器初期化中...")
        predictor = ForexPredictor(
            model_path=args.model,
            config_path=args.config,
            normalization_params_path=args.norm_params
        )
        
        # データ読み込み
        data_file = args.data if args.data else predictor.config['data']['input_file']
        print(f"データ読み込み中: {data_file}")
        tick_data = load_tick_data(data_file, sample_rate=1)  # 予測時はサンプリングしない
        
        if args.single:
            # 単一予測
            print("最新データの予測実行中...")
            result = predictor.predict_single(tick_data)
            
            print("\n予測結果:")
            print(f"予測: {result['prediction_name']} (信頼度: {result['confidence']:.3f})")
            print(f"確率分布: BUY={result['probabilities']['BUY']:.3f}, "
                  f"SELL={result['probabilities']['SELL']:.3f}, "
                  f"NO_TRADE={result['probabilities']['NO_TRADE']:.3f}")
            print(f"現在価格: {result['current_price']:.5f}")
            print(f"スプレッド: {result['spread']:.5f}")
            
            if args.signal:
                signal = predictor.get_trading_signal(tick_data)
                print(f"\nトレーディングシグナル:")
                print(f"アクション: {signal['action']}")
                if signal['entry_price']:
                    print(f"エントリー価格: {signal['entry_price']:.5f}")
                    print(f"TP価格: {signal['tp_price']:.5f}")
                    print(f"SL価格: {signal['sl_price']:.5f}")
                    print(f"リスクリワード比: {signal['expected_risk_reward']:.2f}")
        
        else:
            # バッチ予測
            print("バッチ予測実行中...")
            results = predictor.predict_batch(
                tick_data,
                start_idx=args.start_idx,
                end_idx=args.end_idx
            )
            
            print(f"\n予測完了: {len(results)}件")
            
            # 統計情報
            predictions = [r['prediction'] for r in results]
            buy_count = sum(1 for p in predictions if p == 0)
            sell_count = sum(1 for p in predictions if p == 1)
            no_trade_count = sum(1 for p in predictions if p == 2)
            
            print(f"予測分布: BUY={buy_count}, SELL={sell_count}, NO_TRADE={no_trade_count}")
            
            # 結果保存
            if args.output:
                results_df = pd.DataFrame(results)
                results_df.to_csv(args.output, index=False)
                print(f"結果保存: {args.output}")
            
            # 最新5件の結果表示
            print("\n最新5件の予測結果:")
            for result in results[-5:]:
                print(f"Index {result['index']}: {result['prediction_name']} "
                      f"(信頼度: {result['confidence']:.3f}, 価格: {result['current_price']:.5f})")
    
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()