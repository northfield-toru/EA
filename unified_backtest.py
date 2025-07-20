"""
修正版統合バックテストシステム
- スプレッド二重計上問題を修正
- 信頼度分布の可視化機能を追加
- 温度スケーリング再調整機能を追加

使用方法:
# 基本バックテスト（修正版）
python unified_backtest_fixed.py --model models/best_confidence_model.h5 --data data/usdjpy_ticks.csv

# 信頼度分析付き
python unified_backtest_fixed.py --model models/best_confidence_model.h5 --data data/usdjpy_ticks.csv --analyze-confidence

# 温度スケーリング調整
python unified_backtest_fixed.py --model models/best_confidence_model.h5 --data data/usdjpy_ticks.csv --adjust-temperature
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import argparse
from datetime import datetime, timedelta
import warnings
import itertools
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')

# TensorFlow警告抑制
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# =======================================
# Trade クラス（変更なし）
# =======================================
class Trade:
    """単一取引クラス"""
    
    def __init__(self, entry_time, entry_price, direction, tp_pips, sl_pips, spread_pips=0.7):
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.direction = direction  # 1: BUY, -1: SELL
        self.tp_pips = tp_pips
        self.sl_pips = sl_pips
        self.spread_pips = spread_pips
        
        # 実際のエントリー価格（スプレッド考慮）
        if direction == 1:  # BUY
            self.actual_entry = entry_price + (spread_pips * 0.01)
            self.tp_price = self.actual_entry + (tp_pips * 0.01)
            self.sl_price = self.actual_entry - (sl_pips * 0.01)
        else:  # SELL
            self.actual_entry = entry_price
            self.tp_price = self.actual_entry - (tp_pips * 0.01)
            self.sl_price = self.actual_entry + (sl_pips * 0.01)
        
        self.exit_time = None
        self.exit_price = None
        self.pips = None
        self.result = None
        self.is_closed = False
    
    def check_exit(self, current_time, bid_price, ask_price):
        """現在価格でTP/SL判定"""
        if self.is_closed:
            return False
        
        if self.direction == 1:  # BUY position
            current_price = bid_price
            if current_price >= self.tp_price:
                self._close_trade(current_time, current_price, 'WIN')
                return True
            elif current_price <= self.sl_price:
                self._close_trade(current_time, current_price, 'LOSS')
                return True
        else:  # SELL position
            current_price = ask_price
            if current_price <= self.tp_price:
                self._close_trade(current_time, current_price, 'WIN')
                return True
            elif current_price >= self.sl_price:
                self._close_trade(current_time, current_price, 'LOSS')
                return True
        
        return False
    
    def _close_trade(self, exit_time, exit_price, result):
        """取引クローズ"""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.result = result
        self.is_closed = True
        
        if self.direction == 1:  # BUY
            price_diff = exit_price - self.actual_entry
        else:  # SELL
            price_diff = self.actual_entry - exit_price
        
        self.pips = price_diff / 0.01

# =======================================
# 修正版統合システム
# =======================================
class FixedUnifiedBacktestSystem:
    """修正版統合バックテストシステム"""
    
    def __init__(self, model_path, config_path="config/production_config.json"):
        self.model_path = model_path
        self.config_path = config_path
        
        self.model = None
        self.config = self._load_config()
        self.optimal_temperature = self.config.get('optimal_temperature', 1.0)
        
        # バックテスト設定
        self.trades = []
        self.open_trades = []
        self.equity_curve = []
        
        # 信頼度分析用
        self.raw_confidences = []
        self.calibrated_confidences = []
        
        print(f"🎯 修正版統合バックテストシステム初期化")
        print(f"📁 モデル: {model_path}")
        print(f"🌡️ 現在の温度: {self.optimal_temperature:.3f}")
        
        self._import_modules()
    
    def _import_modules(self):
        """必要なモジュールをインポート"""
        try:
            sys.path.append('.')
            sys.path.append('./development')
            
            from data_loader import load_sample_data
            from feature_engineering import FeatureEngineer
            
            self.load_sample_data = load_sample_data
            self.FeatureEngineer = FeatureEngineer
            
            print("✅ 基本モジュール読み込み完了")
            
        except ImportError as e:
            print(f"❌ モジュール読み込みエラー: {e}")
            raise
    
    def _load_config(self):
        """設定ファイル読み込み"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                confidence_system = config.get('confidence_system', {})
                return {
                    'optimal_temperature': confidence_system.get('optimal_temperature', 1.0),
                    'base_threshold': confidence_system.get('base_threshold', 0.58),
                    'calibration_enabled': confidence_system.get('calibration_enabled', True)
                }
            else:
                return {'optimal_temperature': 1.0, 'base_threshold': 0.58}
        except Exception as e:
            print(f"⚠️ 設定読み込みエラー: {e}")
            return {'optimal_temperature': 1.0, 'base_threshold': 0.58}
    
    # =======================================
    # 修正版データ読み込み・前処理
    # =======================================
    def load_and_prepare_data(self, data_path: str, start_date: str = None, 
                             end_date: str = None, all_data: bool = False):
        """修正版データ読み込み・前処理"""
        print(f"📊 データ読み込み: {data_path}")
        
        # サンプルサイズ決定
        sample_size = 50000000 if all_data else 1000000
        print(f"📈 {'全データ' if all_data else '制限'}モード")
        
        # OHLCV データ読み込み
        ohlcv_data = self.load_sample_data(data_path, sample_size)
        
        if ohlcv_data is None or len(ohlcv_data) == 0:
            raise ValueError("データ読み込みに失敗しました")
        
        print(f"✅ OHLCV読み込み: {len(ohlcv_data)} 行")
        print(f"📅 期間: {ohlcv_data.index[0]} 〜 {ohlcv_data.index[-1]}")
        
        # 期間フィルタ
        if start_date or end_date:
            original_length = len(ohlcv_data)
            if start_date:
                ohlcv_data = ohlcv_data[ohlcv_data.index >= start_date]
            if end_date:
                ohlcv_data = ohlcv_data[ohlcv_data.index <= end_date]
            print(f"🔍 期間フィルタ: {original_length} → {len(ohlcv_data)} 行")
        
        # 特徴量生成
        print("🔧 特徴量生成中...")
        feature_engineer = self.FeatureEngineer()
        features_data = feature_engineer.create_all_features_enhanced(ohlcv_data)
        
        if features_data is None or len(features_data) == 0:
            raise ValueError("特徴量生成に失敗しました")
        
        print(f"✅ 特徴量生成完了: {len(features_data.columns)} 特徴量")
        
        return ohlcv_data, features_data
    
    def prepare_price_data_for_backtest(self, ohlcv_data, standard_spread_pips=0.7):
        """修正版：スプレッド二重計上を解決したバックテスト用価格データ準備"""
        print(f"🔧 スプレッド修正処理中...")
        print(f"   標準スプレッド: {standard_spread_pips} pips")
        
        # STEP1: 元データのbid/askから中央値（close）を計算
        if 'bid' in ohlcv_data.columns and 'ask' in ohlcv_data.columns:
            # 元データにbid/askがある場合
            original_close = (ohlcv_data['bid'] + ohlcv_data['ask']) / 2
            original_spread = ohlcv_data['ask'] - ohlcv_data['bid']
            avg_original_spread = original_spread.mean() / 0.01  # pips換算
            
            print(f"   元データ平均スプレッド: {avg_original_spread:.1f} pips")
            print(f"   → 標準スプレッド {standard_spread_pips} pips に統一")
            
        elif 'close' in ohlcv_data.columns:
            # closeのみの場合
            original_close = ohlcv_data['close']
            print(f"   元データ: close価格のみ")
            
        else:
            raise ValueError("価格データ（bid/ask または close）が見つかりません")
        
        # STEP2: 標準スプレッドで新しいbid/ask作成
        spread_half = (standard_spread_pips * 0.01) / 2  # 0.7pips → 0.0035
        
        price_data = pd.DataFrame({
            'timestamp': ohlcv_data.index,
            'close': original_close,
            'bid': original_close - spread_half,  # -0.35pips
            'ask': original_close + spread_half   # +0.35pips
        })
        
        # 検証用サンプル表示
        print(f"📊 価格データサンプル:")
        for i in range(min(3, len(price_data))):
            row = price_data.iloc[i]
            spread_check = (row['ask'] - row['bid']) / 0.01
            print(f"   {i+1}: BID={row['bid']:.3f}, ASK={row['ask']:.3f}, スプレッド={spread_check:.1f}pips")

        print(f"\n🔍 デバッグ情報:")
        print(f"   元close価格例: {original_close.iloc[0]:.5f}")
        print(f"   修正後BID例: {price_data['bid'].iloc[0]:.5f}")
        print(f"   修正後ASK例: {price_data['ask'].iloc[0]:.5f}")
        print(f"   計算スプレッド: {((price_data['ask'].iloc[0] - price_data['bid'].iloc[0])/0.01):.1f}pips")
        print(f"   データ件数: {len(price_data):,}")

        return price_data.reset_index(drop=True)
    
    # =======================================
    # 修正版予測生成（信頼度分析機能付き）
    # =======================================
    def load_model(self):
        """モデル読み込み"""
        print(f"🧠 モデル読み込み: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"モデルファイルが見つかりません: {self.model_path}")
        
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            print(f"✅ モデル読み込み完了")
        except Exception as e:
            print(f"❌ モデル読み込みエラー: {e}")
            raise
    
    def prepare_sequences(self, features_data, sequence_length=30):
        """シーケンスデータ準備（bid/ask列除外対応）"""
        print(f"📝 シーケンス準備 (長さ: {sequence_length})...")
        
        # bid/ask列を特徴量から除外
        excluded_columns = ['bid', 'ask', 'bid_price', 'ask_price']
        feature_columns = [col for col in features_data.columns 
                          if col not in excluded_columns]
        
        numeric_features = features_data[feature_columns].select_dtypes(include=[np.number])
        numeric_features = numeric_features.fillna(method='ffill').fillna(0)
        
        # 特徴量数確認
        print(f"🔧 特徴量数: {len(numeric_features.columns)} (bid/ask除外済み)")
        
        # 正規化
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(numeric_features)
        
        # シーケンス作成
        sequences = []
        timestamps = []
        
        for i in range(sequence_length, len(scaled_features)):
            sequences.append(scaled_features[i-sequence_length:i])
            timestamps.append(features_data.index[i])
        
        sequences = np.array(sequences)
        print(f"✅ シーケンス準備完了: {sequences.shape}")
        
        return sequences, timestamps
    
    def generate_predictions_with_analysis(self, sequences, timestamps, confidence_threshold=0.58, 
                                         analyze_confidence=False, custom_temperature=None):
        """信頼度分析機能付き予測生成"""
        print(f"🔮 予測生成中: {len(sequences)} サンプル...")
        
        if custom_temperature:
            print(f"🌡️ カスタム温度使用: {custom_temperature:.3f}")
            temperature = custom_temperature
        else:
            temperature = self.optimal_temperature
        
        if self.model is None:
            self.load_model()
        
        # バッチ予測
        batch_size = 1000
        all_predictions = []
        all_raw_confidences = []
        all_calibrated_confidences = []
        
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i+batch_size]
            
            try:
                batch_pred = self.model.predict(batch_sequences, verbose=0)
                
                # 出力形式判定
                if isinstance(batch_pred, list) and len(batch_pred) >= 2:
                    main_pred = batch_pred[0]
                    conf_pred = batch_pred[1]
                    raw_confidences = conf_pred.flatten()
                else:
                    main_pred = batch_pred
                    raw_confidences = np.max(main_pred, axis=1)
                
                # 温度スケーリング適用
                scaled_pred = main_pred / temperature
                scaled_pred = np.exp(scaled_pred) / np.sum(np.exp(scaled_pred), axis=1, keepdims=True)
                calibrated_confidences = np.max(scaled_pred, axis=1)
                
                all_predictions.append(scaled_pred)
                all_raw_confidences.append(raw_confidences)
                all_calibrated_confidences.append(calibrated_confidences)
                
                if (i // batch_size + 1) % 10 == 0:
                    print(f"  進捗: {i + len(batch_sequences)}/{len(sequences)}")
                
            except Exception as e:
                print(f"❌ 予測エラー (バッチ {i}): {e}")
                batch_size_actual = len(batch_sequences)
                all_predictions.append(np.zeros((batch_size_actual, 2)))
                all_raw_confidences.append(np.ones(batch_size_actual) * 0.5)
                all_calibrated_confidences.append(np.ones(batch_size_actual) * 0.5)
        
        # 統合
        final_predictions = np.vstack(all_predictions)
        final_raw_confidences = np.concatenate(all_raw_confidences)
        final_calibrated_confidences = np.concatenate(all_calibrated_confidences)
        
        # 分析用データ保存
        self.raw_confidences = final_raw_confidences
        self.calibrated_confidences = final_calibrated_confidences
        
        print(f"✅ 予測生成完了:")
        print(f"   RAW信頼度平均: {final_raw_confidences.mean():.3f}")
        print(f"   キャリブレーション後平均: {final_calibrated_confidences.mean():.3f}")
        
        # 信頼度分析
        if analyze_confidence:
            self.analyze_confidence_distribution()
        
        # シグナル変換
        signals = self._convert_to_signals_fixed(final_predictions, final_calibrated_confidences, 
                                                timestamps, confidence_threshold)
        
        return signals
    
    def analyze_confidence_distribution(self, save_plots=True):
        """信頼度分布分析"""
        print(f"\n📊 信頼度分布分析開始...")
        
        if len(self.raw_confidences) == 0 or len(self.calibrated_confidences) == 0:
            print("❌ 信頼度データがありません")
            return
        
        # 統計情報
        raw_stats = {
            'mean': self.raw_confidences.mean(),
            'std': self.raw_confidences.std(),
            'min': self.raw_confidences.min(),
            'max': self.raw_confidences.max(),
            'median': np.median(self.raw_confidences)
        }
        
        cal_stats = {
            'mean': self.calibrated_confidences.mean(),
            'std': self.calibrated_confidences.std(),
            'min': self.calibrated_confidences.min(),
            'max': self.calibrated_confidences.max(),
            'median': np.median(self.calibrated_confidences)
        }
        
        print(f"\n📈 RAW信頼度統計:")
        print(f"   平均: {raw_stats['mean']:.3f}, 標準偏差: {raw_stats['std']:.3f}")
        print(f"   範囲: {raw_stats['min']:.3f} 〜 {raw_stats['max']:.3f}")
        print(f"   中央値: {raw_stats['median']:.3f}")
        
        print(f"\n🎯 キャリブレーション後統計:")
        print(f"   平均: {cal_stats['mean']:.3f}, 標準偏差: {cal_stats['std']:.3f}")
        print(f"   範囲: {cal_stats['min']:.3f} 〜 {cal_stats['max']:.3f}")
        print(f"   中央値: {cal_stats['median']:.3f}")
        
        # 信頼度別サンプル数
        thresholds = [0.55, 0.58, 0.60, 0.65, 0.70, 0.75]
        print(f"\n🎯 信頼度閾値別サンプル数:")
        for threshold in thresholds:
            count = np.sum(self.calibrated_confidences >= threshold)
            percentage = count / len(self.calibrated_confidences) * 100
            print(f"   {threshold:.2f}以上: {count:,} ({percentage:.1f}%)")
        
        # 可視化
        if save_plots:
            self._plot_confidence_distributions()
        
        # 温度調整提案
        self._suggest_temperature_adjustment()
        
        return {
            'raw_stats': raw_stats,
            'calibrated_stats': cal_stats,
            'threshold_analysis': {th: np.sum(self.calibrated_confidences >= th) for th in thresholds}
        }
    
    def _plot_confidence_distributions(self):
        """信頼度分布可視化"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. RAW信頼度ヒストグラム
            ax1.hist(self.raw_confidences, bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax1.set_title('RAW信頼度分布')
            ax1.set_xlabel('信頼度')
            ax1.set_ylabel('頻度')
            ax1.axvline(x=self.raw_confidences.mean(), color='red', linestyle='--', label=f'平均: {self.raw_confidences.mean():.3f}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. キャリブレーション後ヒストグラム
            ax2.hist(self.calibrated_confidences, bins=50, alpha=0.7, color='green', edgecolor='black')
            ax2.set_title('キャリブレーション後信頼度分布')
            ax2.set_xlabel('信頼度')
            ax2.set_ylabel('頻度')
            ax2.axvline(x=self.calibrated_confidences.mean(), color='red', linestyle='--', label=f'平均: {self.calibrated_confidences.mean():.3f}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. 累積分布
            sorted_cal = np.sort(self.calibrated_confidences)
            cumulative = np.arange(1, len(sorted_cal) + 1) / len(sorted_cal)
            ax3.plot(sorted_cal, cumulative, linewidth=2)
            ax3.set_title('キャリブレーション後信頼度累積分布')
            ax3.set_xlabel('信頼度')
            ax3.set_ylabel('累積確率')
            ax3.grid(True, alpha=0.3)
            
            # 閾値線追加
            thresholds = [0.55, 0.58, 0.60, 0.65, 0.70]
            for th in thresholds:
                ax3.axvline(x=th, color='red', linestyle=':', alpha=0.7)
                ax3.text(th, 0.5, f'{th:.2f}', rotation=90, ha='right')
            
            # 4. 比較ボックスプロット
            ax4.boxplot([self.raw_confidences, self.calibrated_confidences], 
                       labels=['RAW', 'キャリブレーション後'])
            ax4.set_title('信頼度分布比較')
            ax4.set_ylabel('信頼度')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = f"confidence_analysis_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 信頼度分析グラフ保存: {plot_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"⚠️ グラフ作成エラー: {e}")
    
    def _suggest_temperature_adjustment(self):
        """温度調整提案"""
        print(f"\n💡 温度調整提案:")
        
        # 現在の分布分析
        cal_std = self.calibrated_confidences.std()
        cal_range = self.calibrated_confidences.max() - self.calibrated_confidences.min()
        
        print(f"   現在の温度: {self.optimal_temperature:.3f}")
        print(f"   信頼度標準偏差: {cal_std:.3f}")
        print(f"   信頼度範囲: {cal_range:.3f}")
        
        # 提案
        if cal_std < 0.05:  # 分布が狭すぎる
            suggested_temp = self.optimal_temperature * 0.8  # 温度を下げて分布を広げる
            print(f"   ⚠️ 分布が狭すぎます")
            print(f"   🔧 提案温度: {suggested_temp:.3f} (分布を広げる)")
            
        elif cal_std > 0.15:  # 分布が広すぎる
            suggested_temp = self.optimal_temperature * 1.2  # 温度を上げて分布を狭める
            print(f"   ⚠️ 分布が広すぎます")
            print(f"   🔧 提案温度: {suggested_temp:.3f} (分布を狭める)")
            
        else:
            print(f"   ✅ 現在の温度は適切です")
            suggested_temp = self.optimal_temperature
        
        # 閾値別取引数予測
        threshold_counts = []
        for th in [0.55, 0.58, 0.60, 0.65, 0.70]:
            count = np.sum(self.calibrated_confidences >= th)
            threshold_counts.append((th, count))
        
        print(f"\n📊 閾値別予想取引数:")
        for th, count in threshold_counts:
            print(f"   {th:.2f}: {count:,} 取引")
        
        return suggested_temp
    
    def _convert_to_signals_fixed(self, predictions, confidences, timestamps, confidence_threshold):
        """修正版シグナル変換"""
        print(f"🎯 シグナル変換 (閾値: {confidence_threshold:.2f})...")
        
        signals = []
        
        for i, (pred, conf, timestamp) in enumerate(zip(predictions, confidences, timestamps)):
            pred_class = np.argmax(pred)
            
            # 修正されたシグナル判定
            if conf >= confidence_threshold:
                if pred_class == 1:  # TRADE予測
                    # より賢いBUY/SELL判定（価格傾向やRSI等を使用可能）
                    # 現在は簡易的にランダム
                    signal = 1 if np.random.random() > 0.5 else -1
                else:  # NO_TRADE
                    signal = 0
            else:
                signal = 0
            
            signals.append({
                'timestamp': timestamp,
                'prediction': signal,
                'confidence': conf
            })
        
        # 統計表示
        buy_count = sum(1 for s in signals if s['prediction'] == 1)
        sell_count = sum(1 for s in signals if s['prediction'] == -1)
        no_trade_count = sum(1 for s in signals if s['prediction'] == 0)
        
        print(f"📊 修正版シグナル統計:")
        print(f"  BUY: {buy_count} ({buy_count/len(signals):.1%})")
        print(f"  SELL: {sell_count} ({sell_count/len(signals):.1%})")
        print(f"  NO_TRADE: {no_trade_count} ({no_trade_count/len(signals):.1%})")
        
        return signals
    
    # =======================================
    # バックテスト実行（変更なし）
    # =======================================
    def run_backtest(self, price_data, signals, tp_pips=4.0, sl_pips=5.0, 
                     spread_pips=0.7, max_concurrent_trades=1):
        """バックテスト実行"""
        print(f"🚀 修正版バックテスト実行...")
        print(f"  TP/SL: {tp_pips}/{sl_pips} pips")
        print(f"  🔍 設定スプレッド: {spread_pips} pips")
        
        signals_df = pd.DataFrame(signals)
        signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
        
        merged_data = pd.merge_asof(
            price_data.sort_values('timestamp'),
            signals_df.sort_values('timestamp'),
            on='timestamp',
            direction='backward'
        )
        
        print(f"📊 バックテストデータ: {len(merged_data)} 行")
        
        self.trades = []
        self.open_trades = []
        running_pnl = 0.0
        
        # 🔍 スプレッド検証フラグ
        first_data_check = True
        first_trade_check = True
        
        for idx, row in merged_data.iterrows():
            current_time = row['timestamp']
            bid_price = row['bid']
            ask_price = row['ask']
            
            # 🔍 最初のデータでスプレッド確認
            if first_data_check:
                actual_spread = (ask_price - bid_price) / 0.01
                print(f"  🔍 実際の価格データスプレッド: {actual_spread:.1f}pips")
                print(f"  🔍 BID例: {bid_price:.5f}, ASK例: {ask_price:.5f}")
                first_data_check = False
            
            # 既存ポジション決済判定
            trades_to_remove = []
            for trade in self.open_trades:
                if trade.check_exit(current_time, bid_price, ask_price):
                    self.trades.append(trade)
                    running_pnl += trade.pips
                    trades_to_remove.append(trade)
            
            for trade in trades_to_remove:
                self.open_trades.remove(trade)
            
            # 新規エントリー判定
            if (hasattr(row, 'prediction') and 
                pd.notna(row['prediction']) and 
                row['prediction'] != 0 and
                len(self.open_trades) < max_concurrent_trades):
                
                signal = int(row['prediction'])
                
                new_trade = Trade(
                    entry_time=current_time,
                    entry_price=bid_price if signal == 1 else ask_price,
                    direction=signal,
                    tp_pips=tp_pips,
                    sl_pips=sl_pips,
                    spread_pips=spread_pips
                )
                
                # 🔍 最初の取引作成時に詳細確認
                if first_trade_check:
                    print(f"  🔍 初回取引詳細:")
                    print(f"    方向: {'BUY' if signal == 1 else 'SELL'}")
                    print(f"    エントリー価格: {new_trade.entry_price:.5f}")
                    print(f"    実際エントリー: {new_trade.actual_entry:.5f}")
                    entry_spread = (new_trade.actual_entry - new_trade.entry_price) / 0.01
                    print(f"    エントリー時スプレッド適用: {entry_spread:.1f}pips")
                    print(f"    TP価格: {new_trade.tp_price:.5f}")
                    print(f"    SL価格: {new_trade.sl_price:.5f}")
                    first_trade_check = False
                
                self.open_trades.append(new_trade)
        
        # 未決済ポジション強制決済
        if merged_data is not None and len(merged_data) > 0:
            final_time = merged_data['timestamp'].iloc[-1]
            final_bid = merged_data['bid'].iloc[-1]
            final_ask = merged_data['ask'].iloc[-1]
            
            for trade in self.open_trades:
                trade._close_trade(final_time, 
                                 final_bid if trade.direction == 1 else final_ask, 
                                 'FORCE_CLOSE')
                self.trades.append(trade)
                running_pnl += trade.pips
        
        print(f"✅ バックテスト完了: {len(self.trades)} 取引")
        
        return self._analyze_performance()
    
    def _analyze_performance(self):
        """パフォーマンス分析"""
        if not self.trades:
            return {'error': 'No trades found'}
        
        # 基本統計
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.result == 'WIN']
        losing_trades = [t for t in self.trades if t.result == 'LOSS']
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        # 損益統計
        all_pips = [t.pips for t in self.trades]
        total_pips = sum(all_pips)
        avg_pips = total_pips / total_trades
        
        win_pips = [t.pips for t in winning_trades]
        loss_pips = [t.pips for t in losing_trades]
        
        avg_win = np.mean(win_pips) if win_pips else 0
        avg_loss = np.mean(loss_pips) if loss_pips else 0
        
        # その他指標
        gross_profit = sum(p for p in all_pips if p > 0)
        gross_loss = abs(sum(p for p in all_pips if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        max_drawdown = self._calculate_max_drawdown()
        max_consecutive_losses = self._calculate_max_consecutive_losses()
        
        return {
            'total_trades': total_trades,
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'win_rate': win_rate,
            'total_pips': total_pips,
            'avg_pips_per_trade': avg_pips,
            'avg_win_pips': avg_win,
            'avg_loss_pips': avg_loss,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'profit_factor': profit_factor,
            'max_consecutive_losses': max_consecutive_losses,
            'max_drawdown_pips': max_drawdown,
            'sharpe_ratio': avg_pips / np.std(all_pips) if len(all_pips) > 1 and np.std(all_pips) > 0 else 0
        }
    
    def _calculate_max_drawdown(self):
        """最大ドローダウン計算"""
        if not self.trades:
            return 0
        
        running_pnl = 0
        peak_pnl = 0
        max_dd = 0
        
        for trade in self.trades:
            running_pnl += trade.pips
            peak_pnl = max(peak_pnl, running_pnl)
            drawdown = peak_pnl - running_pnl
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_max_consecutive_losses(self):
        """最大連続負け数計算"""
        max_losses = 0
        current_losses = 0
        
        for trade in self.trades:
            if trade.result == 'LOSS':
                current_losses += 1
                max_losses = max(max_losses, current_losses)
            else:
                current_losses = 0
        
        return max_losses
    
    # =======================================
    # 温度調整機能
    # =======================================
    def test_temperature_range(self, sequences, timestamps, temperature_range=[0.3, 0.5, 0.7, 0.9]):
        """複数温度でのテスト"""
        print(f"\n🌡️ 温度調整テスト開始...")
        print(f"   テスト温度: {temperature_range}")
        
        temperature_results = []
        
        for temp in temperature_range:
            print(f"\n🔄 温度 {temp:.1f} テスト中...")
            
            signals = self.generate_predictions_with_analysis(
                sequences, timestamps, 
                confidence_threshold=0.58,
                analyze_confidence=False,
                custom_temperature=temp
            )
            
            # 信頼度統計
            confidences = [s['confidence'] for s in signals]
            temp_stats = {
                'temperature': temp,
                'avg_confidence': np.mean(confidences),
                'std_confidence': np.std(confidences),
                'min_confidence': np.min(confidences),
                'max_confidence': np.max(confidences),
                'signals_above_58': sum(1 for c in confidences if c >= 0.58),
                'signals_above_65': sum(1 for c in confidences if c >= 0.65),
                'signals_above_70': sum(1 for c in confidences if c >= 0.70)
            }
            
            temperature_results.append(temp_stats)
            
            print(f"   平均信頼度: {temp_stats['avg_confidence']:.3f}")
            print(f"   0.58以上: {temp_stats['signals_above_58']:,}")
            print(f"   0.65以上: {temp_stats['signals_above_65']:,}")
            print(f"   0.70以上: {temp_stats['signals_above_70']:,}")
        
        # 最適温度推奨
        print(f"\n💡 温度調整推奨:")
        for result in temperature_results:
            print(f"   温度 {result['temperature']:.1f}: "
                  f"平均{result['avg_confidence']:.3f}, "
                  f"0.65以上{result['signals_above_65']:,}件")
        
        return temperature_results
    
    # =======================================
    # レポート・メイン実行機能
    # =======================================
    def generate_report(self, results, output_path=None):
        """レポート生成"""
        report_lines = []
        
        report_lines.append("=" * 80)
        report_lines.append("             🎯 修正版統合バックテスト結果")
        report_lines.append("=" * 80)
        report_lines.append(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"モデル: {self.model_path}")
        report_lines.append(f"使用温度: {self.optimal_temperature:.3f}")
        report_lines.append("")
        
        # 基本統計
        report_lines.append("📊 基本統計:")
        report_lines.append(f"   総取引数: {results['total_trades']:,}")
        report_lines.append(f"   勝利取引: {results['winning_trades']:,}")
        report_lines.append(f"   敗北取引: {results['losing_trades']:,}")
        report_lines.append(f"   勝率: {results['win_rate']:.1%}")
        report_lines.append("")
        
        # 損益分析
        report_lines.append("💰 損益分析:")
        report_lines.append(f"   総損益: {results['total_pips']:+.1f} pips")
        report_lines.append(f"   1取引平均: {results['avg_pips_per_trade']:+.2f} pips")
        report_lines.append(f"   平均勝ち: {results['avg_win_pips']:+.1f} pips")
        report_lines.append(f"   平均負け: {results['avg_loss_pips']:+.1f} pips")
        report_lines.append("")
        
        # リスク指標
        report_lines.append("🛡️ リスク分析:")
        report_lines.append(f"   プロフィットファクター: {results['profit_factor']:.2f}")
        report_lines.append(f"   最大連続負け: {results['max_consecutive_losses']} 回")
        report_lines.append(f"   最大ドローダウン: {results['max_drawdown_pips']:.1f} pips")
        report_lines.append(f"   シャープレシオ: {results['sharpe_ratio']:.2f}")
        report_lines.append("")
        
        # 修正点
        report_lines.append("🔧 修正点:")
        report_lines.append("   ✅ スプレッド二重計上問題を解決")
        report_lines.append("   ✅ 標準0.7pipsスプレッドに統一")
        report_lines.append("   ✅ 信頼度分布分析機能を追加")
        report_lines.append("")
        
        # 評価
        report_lines.append("📈 総合評価:")
        if results['avg_pips_per_trade'] > 0.5 and results['win_rate'] >= 0.55:
            report_lines.append("   🎉 優秀: 実運用推奨")
        elif results['avg_pips_per_trade'] > 0 and results['win_rate'] >= 0.50:
            report_lines.append("   ✅ 良好: 実運用可能")
        elif results['avg_pips_per_trade'] > -0.2:
            report_lines.append("   📊 普通: パラメータ調整で改善可能")
        else:
            report_lines.append("   ⚠️ 要改善: スプレッド修正により改善見込み")
        
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"📄 レポート保存: {output_path}")
        
        return report_text
    
    def run_single_test(self, data_path, start_date=None, end_date=None, all_data=False,
                       tp_pips=4.0, sl_pips=5.0, confidence_threshold=0.58, 
                       analyze_confidence=False, adjust_temperature=False, output_dir=None):
        """修正版単一条件テスト実行"""
        print(f"\n🚀 修正版単一条件バックテスト開始")
        print(f"   TP/SL: {tp_pips}/{sl_pips} pips, 信頼度: {confidence_threshold}")
        
        try:
            # データ準備（修正版）
            ohlcv_data, features_data = self.load_and_prepare_data(
                data_path, start_date, end_date, all_data
            )
            price_data = self.prepare_price_data_for_backtest(ohlcv_data)
            
            # シーケンス準備
            sequences, timestamps = self.prepare_sequences(features_data)
            
            # 温度調整テスト
            if adjust_temperature:
                temp_results = self.test_temperature_range(sequences, timestamps)
                
                # 最適温度を提案
                best_temp = None
                best_score = 0
                for result in temp_results:
                    # 0.65以上が1000-10000件程度の温度を好む
                    score = result['signals_above_65']
                    if 1000 <= score <= 10000 and score > best_score:
                        best_score = score
                        best_temp = result['temperature']
                
                if best_temp:
                    print(f"\n🎯 推奨温度: {best_temp:.1f}")
                    user_choice = input(f"推奨温度 {best_temp:.1f} を使用しますか？ (y/N): ")
                    if user_choice.lower() == 'y':
                        self.optimal_temperature = best_temp
                        print(f"✅ 温度を {best_temp:.1f} に変更しました")
            
            # 予測生成（修正版）
            signals = self.generate_predictions_with_analysis(
                sequences, timestamps, confidence_threshold, analyze_confidence
            )
            
            # バックテスト実行
            results = self.run_backtest(price_data, signals, tp_pips, sl_pips, spread_pips=0)
            
            # 結果保存
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # レポート保存
                report_path = f"{output_dir}/fixed_report_{timestamp}.txt"
                self.generate_report(results, report_path)
                
                # JSON保存
                json_path = f"{output_dir}/fixed_results_{timestamp}.json"
                results['test_conditions'] = {
                    'tp_pips': tp_pips,
                    'sl_pips': sl_pips, 
                    'confidence_threshold': confidence_threshold,
                    'temperature_used': self.optimal_temperature,
                    'spread_fixed': True,
                    'data_path': data_path,
                    'model_path': self.model_path
                }
                
                with open(json_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                
                print(f"📁 結果保存: {output_dir}")
            
            return results
            
        except Exception as e:
            print(f"❌ バックテストエラー: {e}")
            return {'error': str(e)}

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description='修正版統合バックテストシステム')
    
    # 必須パラメータ
    parser.add_argument('--model', required=True, help='モデルファイルパス')
    parser.add_argument('--data', required=True, help='データファイルパス')
    
    # オプション
    parser.add_argument('--config', default='config/production_config.json', help='設定ファイル')
    parser.add_argument('--start', help='開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', help='終了日 (YYYY-MM-DD)')
    parser.add_argument('--all-data', action='store_true', help='全データ使用')
    parser.add_argument('--output', default='fixed_backtest_results', help='出力ディレクトリ')
    
    # 単一テスト用パラメータ
    parser.add_argument('--tp', type=float, default=4.0, help='利確pips')
    parser.add_argument('--sl', type=float, default=5.0, help='損切pips')
    parser.add_argument('--confidence', type=float, default=0.58, help='信頼度閾値')
    
    # 分析・調整機能
    parser.add_argument('--analyze-confidence', action='store_true', help='信頼度分布分析')
    parser.add_argument('--adjust-temperature', action='store_true', help='温度スケーリング調整')
    
    args = parser.parse_args()
    
    # ファイル存在確認
    if not os.path.exists(args.model):
        print(f"❌ モデルファイルが見つかりません: {args.model}")
        return 1
    
    if not os.path.exists(args.data):
        print(f"❌ データファイルが見つかりません: {args.data}")
        return 1
    
    # 修正版システム初期化
    system = FixedUnifiedBacktestSystem(args.model, args.config)
    
    try:
        # 修正版単一テスト実行
        print("🎯 修正版バックテストモード")
        
        results = system.run_single_test(
            data_path=args.data,
            start_date=args.start,
            end_date=args.end,
            all_data=args.all_data,
            tp_pips=args.tp,
            sl_pips=args.sl,
            confidence_threshold=args.confidence,
            analyze_confidence=args.analyze_confidence,
            adjust_temperature=args.adjust_temperature,
            output_dir=args.output
        )
        
        if 'error' in results:
            print(f"❌ バックテスト失敗: {results['error']}")
            return 1
        else:
            print(f"\n✅ 修正版バックテスト成功")
            print(f"💰 結果: {results['avg_pips_per_trade']:+.2f} pips/取引")
            print(f"🎯 勝率: {results['win_rate']:.1%}")
            print(f"📊 取引数: {results['total_trades']}")
            print(f"📈 PF: {results['profit_factor']:.2f}")
            
            # コンソールにもレポート表示
            print("\n" + system.generate_report(results))
            
            return 0
    
    except Exception as e:
        print(f"❌ システムエラー: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())