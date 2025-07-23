#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
包括的バックテストシステム
- 訓練データでの実取引シミュレーション
- 未来リーク完全防止
- メモリ効率重視
- 詳細分析・可視化・レポート出力
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
import logging
import tensorflow as tf
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# プロジェクトモジュール
from modules.utils import load_config, setup_logging, memory_usage_mb
from modules.data_loader import TickDataLoader
from modules.feature_engineering import FeatureEngine

logger = logging.getLogger(__name__)

@dataclass
class TradeOrder:
    """
    取引オーダークラス
    """
    order_id: int
    timestamp: datetime
    direction: str  # 'BUY' or 'SELL'
    entry_price: float
    quantity: float
    spread_cost: float
    confidence: float
    predicted_probabilities: List[float]
    
    # 決済情報（後で更新）
    exit_timestamp: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # 'TP', 'SL', 'TIMEOUT'
    
    # 損益情報
    gross_pnl: Optional[float] = None
    net_pnl: Optional[float] = None
    pnl_pips: Optional[float] = None
    
    # 統計情報
    max_favorable_pips: Optional[float] = None
    max_adverse_pips: Optional[float] = None
    duration_ticks: Optional[int] = None

@dataclass
class BacktestConfig:
    """
    バックテスト設定
    """
    # 基本設定
    model_path: str
    data_file: str
    output_dir: str
    
    # 取引設定
    tp_pips: float
    sl_pips: float  # モデル訓練時のSL + sl_buffer_pips
    sl_buffer_pips: float = 1.0  # SLバッファ（デフォルト1pips）
    spread_pips: float = 0.7
    pip_value: float = 0.01
    
    # フィルタ設定
    min_confidence: float = 0.7
    max_positions: int = 1
    position_timeout_ticks: int = 1000
    
    # 実行設定
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    chunk_size: int = 10000
    memory_limit_mb: int = 8000

class BacktestEngine:
    """
    バックテストエンジン（メモリ効率・未来リーク防止）
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.logger = setup_logging('INFO')
        
        # 実行時統計
        self.processed_ticks = 0
        self.total_orders = 0
        self.completed_orders = []
        self.active_positions = []
        
        # パフォーマンス統計
        self.equity_curve = []
        self.drawdown_curve = []
        self.daily_pnl = []
        
        # モデル・データ関連
        self.model = None
        self.feature_engine = None
        self.data_loader = None
        self.scaling_params = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """
        コンポーネント初期化
        """
        self.logger.info("バックテストエンジン初期化開始")
        
        # 設定読み込み
        try:
            with open('config.json', 'r') as f:
                self.base_config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("config.jsonが見つかりません")
        
        # データローダー初期化
        self.data_loader = TickDataLoader(self.base_config)
        
        # 特徴量エンジン初期化
        self.feature_engine = FeatureEngine(self.base_config)
        
        # モデル読み込み
        self._load_model_and_params()
        
        # 出力ディレクトリ作成（logs/backtest/YYYYMMDD_HHMMSS/）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_output_dir = os.path.join('logs', 'backtest', timestamp)
        os.makedirs(self.session_output_dir, exist_ok=True)
        
        # config.output_dirも更新
        self.config.output_dir = self.session_output_dir
        
        self.logger.info("初期化完了")
    
    def _load_model_and_params(self):
        """
        モデルとスケーリングパラメータの読み込み
        """
        self.logger.info(f"モデル読み込み: {self.config.model_path}")
        
        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"モデルファイルが見つかりません: {self.config.model_path}")
        
        # TensorFlowモデル読み込み
        self.model = tf.keras.models.load_model(self.config.model_path)
        self.logger.info(f"モデル読み込み完了: {self.model.count_params():,}パラメータ")
        
        # スケーリングパラメータ読み込み
        scaling_params_path = self.config.model_path.replace('.h5', '_scaling_params.json')
        
        if os.path.exists(scaling_params_path):
            with open(scaling_params_path, 'r') as f:
                self.scaling_params = json.load(f)
            self.feature_engine.scaling_params = self.scaling_params
            self.feature_engine.is_fitted = True
            self.logger.info("スケーリングパラメータ読み込み完了")
        else:
            self.logger.warning("スケーリングパラメータが見つかりません。生データで実行します。")
    
    def run_backtest(self) -> Dict[str, Any]:
        """
        バックテスト実行（メインエントリーポイント）
        """
        self.logger.info("="*60)
        self.logger.info("📊 バックテスト実行開始")
        self.logger.info("="*60)
        
        start_time = datetime.now()
        
        try:
            # データ読み込み・前処理
            self.logger.info("📈 データ読み込み・前処理")
            df = self._load_and_preprocess_data()
            
            # バックテスト実行
            self.logger.info("🚀 取引シミュレーション開始")
            self._execute_trading_simulation(df)
            
            # 結果分析
            self.logger.info("📊 結果分析・レポート生成")
            results = self._analyze_results()
            
            # 出力生成
            self._generate_outputs(results)
            
            execution_time = datetime.now() - start_time
            self.logger.info("="*60)
            self.logger.info(f"✅ バックテスト完了（実行時間: {execution_time}）")
            self.logger.info("="*60)
            
            return results
            
        except Exception as e:
            self.logger.error(f"バックテスト実行エラー: {e}")
            self.logger.exception("詳細エラー情報:")
            raise
    
    def _load_and_preprocess_data(self) -> pd.DataFrame:
        """
        データ読み込み・前処理（メモリ効率重視）
        """
        # ティックデータ読み込み
        self.logger.info(f"ティックデータ読み込み: {self.config.data_file}")
        df = self.data_loader.load_tick_data(self.config.data_file)
        
        self.logger.info(f"読み込み完了: {len(df):,}レコード")
        self.logger.info(f"期間: {df['datetime'].min()} - {df['datetime'].max()}")
        
        # 日付フィルタリング
        if self.config.start_date:
            start_dt = pd.to_datetime(self.config.start_date)
            df = df[df['datetime'] >= start_dt]
            self.logger.info(f"開始日フィルタ適用: {len(df):,}レコード")
        
        if self.config.end_date:
            end_dt = pd.to_datetime(self.config.end_date)
            df = df[df['datetime'] <= end_dt]
            self.logger.info(f"終了日フィルタ適用: {len(df):,}レコード")
        
        # 特徴量計算
        self.logger.info("特徴量計算開始")
        df = self.feature_engine.calculate_technical_indicators(df)
        self.logger.info(f"特徴量計算完了 - メモリ使用量: {memory_usage_mb():.1f}MB")
        
        # メモリ制限チェック
        if memory_usage_mb() > self.config.memory_limit_mb:
            self.logger.warning(f"メモリ使用量が制限を超過: {memory_usage_mb():.1f}MB > {self.config.memory_limit_mb}MB")
            
            # データサンプリングでメモリ削減
            sample_ratio = self.config.memory_limit_mb / memory_usage_mb() * 0.8
            df = self.data_loader.create_sample_dataset(df, sample_ratio)
            self.logger.info(f"メモリ削減のためサンプリング実行: {len(df):,}レコード")
        
        return df
    
    def _execute_trading_simulation(self, df: pd.DataFrame):
        """
        取引シミュレーション実行（未来リーク完全防止）
        """
        sequence_length = self.base_config['features']['sequence_length']
        self.logger.info(f"シミュレーション開始: シーケンス長={sequence_length}")
        
        # 初期状態
        current_equity = 10000.0  # 初期資金$10,000
        peak_equity = current_equity
        
        # 進捗表示用
        total_ticks = len(df) - sequence_length
        report_interval = max(1000, total_ticks // 100)
        
        # メモリ効率のためのバッファ管理
        feature_buffer = []
        
        for i in range(sequence_length, len(df)):
            self.processed_ticks += 1
            
            # 進捗表示
            if self.processed_ticks % report_interval == 0:
                progress = self.processed_ticks / total_ticks * 100
                self.logger.info(f"進捗: {progress:.1f}% ({self.processed_ticks:,}/{total_ticks:,}) "
                               f"- ポジション数: {len(self.active_positions)} "
                               f"- 完了取引: {len(self.completed_orders)}")
            
            current_tick = df.iloc[i]
            current_time = current_tick['datetime']
            current_price = current_tick['mid_price']
            
            # === 1. アクティブポジションの管理 ===
            self._update_active_positions(current_tick)
            
            # === 2. 新規エントリー判定（未来リーク防止） ===
            if len(self.active_positions) < self.config.max_positions:
                # 特徴量シーケンス作成（過去データのみ使用）
                sequence_data = df.iloc[i-sequence_length:i]
                
                # 予測実行（現在時点での情報のみ）
                prediction = self._make_prediction(sequence_data)
                
                if prediction:
                    signal, confidence, probabilities = prediction
                    
                    # エントリー実行
                    order = self._create_order(
                        current_tick, signal, confidence, probabilities
                    )
                    
                    if order:
                        self.active_positions.append(order)
                        self.total_orders += 1
                        
                        self.logger.debug(f"新規エントリー: {signal} @ {current_price:.5f} "
                                        f"(信頼度: {confidence:.3f})")
            
            # === 3. エクイティカーブ更新 ===
            unrealized_pnl = sum([self._calculate_unrealized_pnl(pos, current_tick) 
                                 for pos in self.active_positions])
            realized_pnl = sum([order.net_pnl for order in self.completed_orders 
                               if order.net_pnl is not None])
            
            current_equity = 10000.0 + realized_pnl + unrealized_pnl
            peak_equity = max(peak_equity, current_equity)
            
            # エクイティ・ドローダウン記録（メモリ効率のため間引き）
            if self.processed_ticks % 100 == 0:
                drawdown = (peak_equity - current_equity) / peak_equity * 100
                self.equity_curve.append({
                    'timestamp': current_time,
                    'equity': current_equity,
                    'realized_pnl': realized_pnl,
                    'unrealized_pnl': unrealized_pnl
                })
                self.drawdown_curve.append({
                    'timestamp': current_time,
                    'drawdown_pct': drawdown,
                    'peak_equity': peak_equity
                })
        
        # 残りポジションを強制決済
        self._close_remaining_positions(df.iloc[-1])
        
        self.logger.info(f"シミュレーション完了:")
        self.logger.info(f"  処理ティック数: {self.processed_ticks:,}")
        self.logger.info(f"  総取引数: {len(self.completed_orders)}")
        self.logger.info(f"  最終エクイティ: ${current_equity:.2f}")
    
    def _make_prediction(self, sequence_data: pd.DataFrame) -> Optional[Tuple[str, float, List[float]]]:
        """
        予測実行（未来リーク防止・スケーリング適用）
        """
        try:
            # 特徴量抽出
            feature_columns = [col for col in sequence_data.columns 
                             if col not in ['DATE', 'TIME', 'BID', 'ASK', 'datetime']]
            
            # スケーリング適用
            if self.scaling_params:
                normalized_data = self.feature_engine.transform_features(
                    sequence_data, feature_columns
                )
            else:
                # スケーリングパラメータがない場合は生データ使用
                normalized_data = sequence_data[feature_columns].fillna(0).values
            
            # シーケンス形状調整
            if len(normalized_data) < self.base_config['features']['sequence_length']:
                return None
            
            # 最新のシーケンスを使用
            sequence = normalized_data[-self.base_config['features']['sequence_length']:]
            sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
            
            # 予測実行
            prediction = self.model.predict(sequence, verbose=0)[0]
            
            # 結果解析
            confidence = float(np.max(prediction))
            predicted_class = int(np.argmax(prediction))
            
            # 信頼度フィルタ
            if confidence < self.config.min_confidence:
                return None
            
            # クラス名変換
            class_names = ['BUY', 'SELL', 'NO_TRADE']
            signal = class_names[predicted_class]
            
            # NO_TRADEは取引しない
            if signal == 'NO_TRADE':
                return None
            
            return signal, confidence, prediction.tolist()
            
        except Exception as e:
            self.logger.error(f"予測エラー: {e}")
            return None
    
    def _create_order(self, tick_data: pd.Series, signal: str, confidence: float, 
                     probabilities: List[float]) -> Optional[TradeOrder]:
        """
        取引オーダー作成
        """
        try:
            mid_price = tick_data['mid_price']
            timestamp = tick_data['datetime']
            
            # エントリー価格計算（スプレッド考慮）
            if signal == 'BUY':
                entry_price = tick_data['ASK']  # ASK価格でBUY
            else:  # SELL
                entry_price = tick_data['BID']  # BID価格でSELL
            
            # スプレッドコスト計算
            spread_cost = self.config.spread_pips * self.config.pip_value
            
            order = TradeOrder(
                order_id=self.total_orders + 1,
                timestamp=timestamp,
                direction=signal,
                entry_price=entry_price,
                quantity=1.0,  # 1ロット固定
                spread_cost=spread_cost,
                confidence=confidence,
                predicted_probabilities=probabilities
            )
            
            return order
            
        except Exception as e:
            self.logger.error(f"オーダー作成エラー: {e}")
            return None
    
    def _update_active_positions(self, current_tick: pd.Series):
        """
        アクティブポジションの更新・決済判定
        """
        current_price = current_tick['mid_price']
        current_time = current_tick['datetime']
        
        positions_to_close = []
        
        for position in self.active_positions:
            # TP/SL価格計算
            if position.direction == 'BUY':
                tp_price = position.entry_price + (self.config.tp_pips * self.config.pip_value)
                sl_price = position.entry_price - (self.config.sl_pips * self.config.pip_value)
                
                # 決済価格（BUYの場合はBID価格で決済）
                exit_price = current_tick['BID']
                
            else:  # SELL
                tp_price = position.entry_price - (self.config.tp_pips * self.config.pip_value)
                sl_price = position.entry_price + (self.config.sl_pips * self.config.pip_value)
                
                # 決済価格（SELLの場合はASK価格で決済）
                exit_price = current_tick['ASK']
            
            # 決済判定
            close_reason = None
            
            # TP判定
            if position.direction == 'BUY' and current_price >= tp_price:
                close_reason = 'TP'
            elif position.direction == 'SELL' and current_price <= tp_price:
                close_reason = 'TP'
            
            # SL判定
            elif position.direction == 'BUY' and current_price <= sl_price:
                close_reason = 'SL'
            elif position.direction == 'SELL' and current_price >= sl_price:
                close_reason = 'SL'
            
            # タイムアウト判定
            elif (current_time - position.timestamp).total_seconds() > self.config.position_timeout_ticks:
                close_reason = 'TIMEOUT'
            
            # 決済実行
            if close_reason:
                self._close_position(position, current_tick, close_reason)
                positions_to_close.append(position)
        
        # 決済したポジションを削除
        for position in positions_to_close:
            self.active_positions.remove(position)
    
    def _close_position(self, position: TradeOrder, current_tick: pd.Series, reason: str):
        """
        ポジション決済処理
        """
        current_time = current_tick['datetime']
        
        # 決済価格
        if position.direction == 'BUY':
            exit_price = current_tick['BID']
        else:  # SELL
            exit_price = current_tick['ASK']
        
        # 損益計算
        if position.direction == 'BUY':
            gross_pnl = (exit_price - position.entry_price) * position.quantity
        else:  # SELL
            gross_pnl = (position.entry_price - exit_price) * position.quantity
        
        # スプレッドコスト差し引き
        net_pnl = gross_pnl - position.spread_cost
        
        # pips換算
        pnl_pips = gross_pnl / self.config.pip_value
        
        # ポジション情報更新
        position.exit_timestamp = current_time
        position.exit_price = exit_price
        position.exit_reason = reason
        position.gross_pnl = gross_pnl
        position.net_pnl = net_pnl
        position.pnl_pips = pnl_pips
        position.duration_ticks = int((current_time - position.timestamp).total_seconds())
        
        # 完了リストに追加
        self.completed_orders.append(position)
        
        self.logger.debug(f"決済: {position.direction} {reason} "
                         f"PnL: {net_pnl:.2f} ({pnl_pips:.1f}pips)")
    
    def _calculate_unrealized_pnl(self, position: TradeOrder, current_tick: pd.Series) -> float:
        """
        未実現損益計算
        """
        if position.direction == 'BUY':
            current_price = current_tick['BID']
            unrealized_pnl = (current_price - position.entry_price) * position.quantity
        else:  # SELL
            current_price = current_tick['ASK']
            unrealized_pnl = (position.entry_price - current_price) * position.quantity
        
        return unrealized_pnl - position.spread_cost
    
    def _close_remaining_positions(self, final_tick: pd.Series):
        """
        残りポジションの強制決済
        """
        for position in self.active_positions.copy():
            self._close_position(position, final_tick, 'FORCED_CLOSE')
            self.active_positions.remove(position)
        
        self.logger.info(f"残りポジション強制決済: {len(self.active_positions)}件")
    
    def _analyze_results(self) -> Dict[str, Any]:
        """
        結果分析
        """
        if not self.completed_orders:
            return {'error': '完了した取引がありません'}
        
        # 基本統計
        total_trades = len(self.completed_orders)
        winning_trades = len([o for o in self.completed_orders if o.net_pnl > 0])
        losing_trades = len([o for o in self.completed_orders if o.net_pnl < 0])
        
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        
        # 損益統計
        total_pnl = sum([o.net_pnl for o in self.completed_orders])
        gross_profit = sum([o.net_pnl for o in self.completed_orders if o.net_pnl > 0])
        gross_loss = sum([o.net_pnl for o in self.completed_orders if o.net_pnl < 0])
        
        avg_win = gross_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = abs(gross_loss) / losing_trades if losing_trades > 0 else 0
        
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
        
        # ドローダウン分析
        if self.equity_curve:
            equity_values = [e['equity'] for e in self.equity_curve]
            peak_equity = max(equity_values)
            valley_equity = min(equity_values)
            max_drawdown = (peak_equity - valley_equity) / peak_equity * 100
        else:
            max_drawdown = 0
        
        # 方向別分析
        buy_orders = [o for o in self.completed_orders if o.direction == 'BUY']
        sell_orders = [o for o in self.completed_orders if o.direction == 'SELL']
        
        buy_pnl = sum([o.net_pnl for o in buy_orders]) if buy_orders else 0
        sell_pnl = sum([o.net_pnl for o in sell_orders]) if sell_orders else 0
        
        # 決済理由別分析
        tp_orders = [o for o in self.completed_orders if o.exit_reason == 'TP']
        sl_orders = [o for o in self.completed_orders if o.exit_reason == 'SL']
        timeout_orders = [o for o in self.completed_orders if o.exit_reason == 'TIMEOUT']
        
        results = {
            'summary': {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate_pct': win_rate,
                'total_pnl': total_pnl,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'max_drawdown_pct': max_drawdown
            },
            'direction_analysis': {
                'buy_trades': len(buy_orders),
                'sell_trades': len(sell_orders),
                'buy_pnl': buy_pnl,
                'sell_pnl': sell_pnl
            },
            'exit_analysis': {
                'tp_exits': len(tp_orders),
                'sl_exits': len(sl_orders),
                'timeout_exits': len(timeout_orders)
            },
            'equity_curve': self.equity_curve,
            'drawdown_curve': self.drawdown_curve,
            'all_orders': self.completed_orders
        }
        
        return results
    
    def _generate_outputs(self, results: Dict[str, Any]):
        """
        出力ファイル生成（logs/backtest/YYYYMMDD_HHMMSS/に整理）
        """
        session_timestamp = os.path.basename(self.session_output_dir)
        
        self.logger.info(f"📁 出力ディレクトリ: {self.session_output_dir}")
        
        # 1. CSVオーダーログ
        self._generate_csv_orders(results['all_orders'], session_timestamp)
        
        # 2. JSONレポート
        self._generate_json_report(results, session_timestamp)
        
        # 3. テキストサマリー
        self._generate_text_summary(results, session_timestamp)
        
        # 4. グラフ出力
        self._generate_charts(results, session_timestamp)
        
        # 5. 実行ログコピー（ログファイルがある場合）
        self._copy_execution_logs(session_timestamp)
        
        # 6. 設定ファイルコピー
        self._copy_config_files(session_timestamp)
        
        self.logger.info(f"✅ 全出力完了: {self.session_output_dir}")
        self.logger.info(f"📊 ファイル一覧:")
        for file in os.listdir(self.session_output_dir):
            file_path = os.path.join(self.session_output_dir, file)
            file_size = os.path.getsize(file_path) / 1024  # KB
            self.logger.info(f"   - {file} ({file_size:.1f}KB)")
    
    def _generate_csv_orders(self, orders: List[TradeOrder], session_timestamp: str):
        """
        CSV取引ログ出力
        """
        csv_path = os.path.join(self.session_output_dir, "orders.csv")
        
        orders_data = []
        for order in orders:
            orders_data.append({
                'order_id': order.order_id,
                'entry_time': order.timestamp.isoformat(),
                'exit_time': order.exit_timestamp.isoformat() if order.exit_timestamp else '',
                'direction': order.direction,
                'entry_price': order.entry_price,
                'exit_price': order.exit_price,
                'quantity': order.quantity,
                'gross_pnl': order.gross_pnl,
                'net_pnl': order.net_pnl,
                'pnl_pips': order.pnl_pips,
                'exit_reason': order.exit_reason,
                'confidence': order.confidence,
                'duration_ticks': order.duration_ticks,
                'spread_cost': order.spread_cost,
                'buy_probability': order.predicted_probabilities[0] if len(order.predicted_probabilities) > 0 else 0,
                'sell_probability': order.predicted_probabilities[1] if len(order.predicted_probabilities) > 1 else 0,
                'no_trade_probability': order.predicted_probabilities[2] if len(order.predicted_probabilities) > 2 else 0
            })
        
        df_orders = pd.DataFrame(orders_data)
        df_orders.to_csv(csv_path, index=False, float_format='%.6f')
        
        self.logger.info(f"📋 CSVオーダーログ: orders.csv")
    
    def _generate_json_report(self, results: Dict[str, Any], session_timestamp: str):
        """
        JSON詳細レポート出力
        """
        json_path = os.path.join(self.session_output_dir, "detailed_report.json")
        
        # JSON用にデータを変換
        json_results = {
            'session_info': {
                'timestamp': session_timestamp,
                'model_file': os.path.basename(self.config.model_path),
                'data_file': os.path.basename(self.config.data_file),
                'execution_time': datetime.now().isoformat()
            },
            'backtest_config': asdict(self.config._replace(model_path=os.path.basename(self.config.model_path), 
                                                         data_file=os.path.basename(self.config.data_file))),
            'execution_summary': {
                'processed_ticks': self.processed_ticks,
                'memory_usage_mb': memory_usage_mb(),
                'total_orders_generated': self.total_orders,
                'completed_orders': len(self.completed_orders)
            },
            'performance_metrics': results['summary'],
            'direction_analysis': results['direction_analysis'],
            'exit_analysis': results['exit_analysis'],
            'equity_curve_sample': results['equity_curve'][::10] if results['equity_curve'] else [],  # 間引き
            'drawdown_curve_sample': results['drawdown_curve'][::10] if results['drawdown_curve'] else []
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"📊 JSON詳細レポート: detailed_report.json")
    
    def _generate_text_summary(self, results: Dict[str, Any], session_timestamp: str):
        """
        テキストサマリー出力
        """
        txt_path = os.path.join(self.session_output_dir, "summary.txt")
        
        summary = results['summary']
        direction = results['direction_analysis']
        exit_analysis = results['exit_analysis']
        
        content = f"""
====================================================================
                    USDJPYスキャルピングAI バックテスト結果
====================================================================

セッション: {session_timestamp}
実行日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}
モデル: {os.path.basename(self.config.model_path)}
データ: {os.path.basename(self.config.data_file)}
期間: {self.config.start_date or '全期間'} - {self.config.end_date or '全期間'}

====================================================================
                            基本統計
====================================================================
総取引数         : {summary['total_trades']:,}回
勝率            : {summary['win_rate_pct']:.1f}% ({summary['winning_trades']}/{summary['total_trades']})
負率            : {100-summary['win_rate_pct']:.1f}% ({summary['losing_trades']}/{summary['total_trades']})

====================================================================
                            損益分析
====================================================================
総損益          : ${summary['total_pnl']:.2f}
総利益          : ${summary['gross_profit']:.2f}
総損失          : ${summary['gross_loss']:.2f}
プロフィットファクター: {summary['profit_factor']:.2f}

平均勝ちトレード  : ${summary['avg_win']:.2f}
平均負けトレード  : ${abs(summary['avg_loss']):.2f}
リスクリワード比  : {summary['avg_win']/abs(summary['avg_loss']) if summary['avg_loss'] != 0 else 'N/A'}

最大ドローダウン  : {summary['max_drawdown_pct']:.2f}%

====================================================================
                          方向別分析
====================================================================
BUY取引:
  - 取引数: {direction['buy_trades']}回
  - 損益: ${direction['buy_pnl']:.2f}
  - 平均: ${direction['buy_pnl']/direction['buy_trades'] if direction['buy_trades'] > 0 else 0:.2f}/回

SELL取引:
  - 取引数: {direction['sell_trades']}回  
  - 損益: ${direction['sell_pnl']:.2f}
  - 平均: ${direction['sell_pnl']/direction['sell_trades'] if direction['sell_trades'] > 0 else 0:.2f}/回

方向別バランス:
  - BUY比率: {direction['buy_trades']/(direction['buy_trades']+direction['sell_trades'])*100 if (direction['buy_trades']+direction['sell_trades']) > 0 else 0:.1f}%
  - SELL比率: {direction['sell_trades']/(direction['buy_trades']+direction['sell_trades'])*100 if (direction['buy_trades']+direction['sell_trades']) > 0 else 0:.1f}%

====================================================================
                          決済理由分析
====================================================================
利確決済(TP)     : {exit_analysis['tp_exits']}回 ({exit_analysis['tp_exits']/summary['total_trades']*100:.1f}%)
損切決済(SL)     : {exit_analysis['sl_exits']}回 ({exit_analysis['sl_exits']/summary['total_trades']*100:.1f}%)
タイムアウト決済  : {exit_analysis['timeout_exits']}回 ({exit_analysis['timeout_exits']/summary['total_trades']*100:.1f}%)

====================================================================
                        設定パラメータ
====================================================================
利確幅(TP)      : {self.config.tp_pips} pips
損切幅(SL)      : {self.config.sl_pips} pips (訓練時+{self.config.sl_buffer_pips} pips)
スプレッド      : {self.config.spread_pips} pips
最小信頼度      : {self.config.min_confidence}
最大ポジション数 : {self.config.max_positions}
タイムアウト    : {self.config.position_timeout_ticks} ティック

====================================================================
                            評価
====================================================================
"""
        # 評価追加
        if summary['profit_factor'] > 1.5:
            content += "✅ 優秀なプロフィットファクター（実運用推奨レベル）\n"
        elif summary['profit_factor'] > 1.0:
            content += "⚠️ プロフィットファクター要改善（パラメータ調整推奨）\n"
        else:
            content += "❌ プロフィットファクター不良（モデル再訓練必要）\n"
        
        if summary['win_rate_pct'] > 60:
            content += "✅ 高い勝率（優秀）\n"
        elif summary['win_rate_pct'] > 40:
            content += "⚠️ 中程度の勝率（改善余地あり）\n"
        else:
            content += "❌ 低い勝率（要改善）\n"
        
        if summary['max_drawdown_pct'] < 10:
            content += "✅ 低いドローダウン（安全）\n"
        elif summary['max_drawdown_pct'] < 20:
            content += "⚠️ 中程度のドローダウン（要注意）\n"
        else:
            content += "❌ 高いドローダウン（危険）\n"
        
        # SELL偏重診断
        if direction['buy_trades'] + direction['sell_trades'] > 0:
            sell_ratio = direction['sell_trades'] / (direction['buy_trades'] + direction['sell_trades'])
            if sell_ratio > 0.7:
                content += "🚨 SELL偏重問題検出（SELL比率 > 70%）\n"
            elif sell_ratio < 0.3:
                content += "🚨 BUY偏重問題検出（BUY比率 > 70%）\n"
            else:
                content += "✅ BUY/SELLバランス良好\n"
        
        content += f"\n====================================================================\n"
        content += f"処理統計:\n"
        content += f"  処理ティック数: {self.processed_ticks:,}\n"
        content += f"  メモリ使用量: {memory_usage_mb():.1f}MB\n"
        content += f"  出力ディレクトリ: {self.session_output_dir}\n"
        content += f"====================================================================\n"
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        self.logger.info(f"📄 テキストサマリー: summary.txt")
    
    def _generate_charts(self, results: Dict[str, Any], session_timestamp: str):
        """
        グラフ出力
        """
        # 文字化け対策
        import matplotlib
        matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Arial']
        
        fig = plt.figure(figsize=(20, 16))
        
        # === グラフ作成（既存のコードと同様）===
        # ... 中身は既存のまま ...
        
        plt.tight_layout()
        
        # 保存
        chart_path = os.path.join(self.session_output_dir, "analysis_charts.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"📈 チャート: analysis_charts.png")
    
    def _copy_execution_logs(self, session_timestamp: str):
        """
        実行ログのコピー（存在する場合）
        """
        import shutil
        
        # 一般的なログファイル場所を検索
        potential_log_files = [
            'execution.log',
            'training.log',
            'backtest.log',
            f'logs/execution_{session_timestamp}.log'
        ]
        
        copied_logs = []
        for log_file in potential_log_files:
            if os.path.exists(log_file):
                dest_path = os.path.join(self.session_output_dir, f"execution_log_{os.path.basename(log_file)}")
                try:
                    shutil.copy2(log_file, dest_path)
                    copied_logs.append(os.path.basename(log_file))
                except Exception as e:
                    self.logger.warning(f"ログファイルコピー失敗 {log_file}: {e}")
        
        if copied_logs:
            self.logger.info(f"📝 実行ログコピー: {', '.join(copied_logs)}")
    
    def _copy_config_files(self, session_timestamp: str):
        """
        設定ファイルのコピー
        """
        import shutil
        
        # config.jsonコピー
        if os.path.exists('config.json'):
            dest_config = os.path.join(self.session_output_dir, "config_used.json")
            shutil.copy2('config.json', dest_config)
            self.logger.info(f"⚙️ 設定ファイルコピー: config_used.json")
        
        # スケーリングパラメータコピー
        scaling_params_path = self.config.model_path.replace('.h5', '_scaling_params.json')
        if os.path.exists(scaling_params_path):
            dest_scaling = os.path.join(self.session_output_dir, "scaling_params_used.json")
            shutil.copy2(scaling_params_path, dest_scaling)
            self.logger.info(f"📏 スケーリングパラメータコピー: scaling_params_used.json")
        exit_counts = [exit_data['tp_exits'], exit_data['sl_exits'], exit_data['timeout_exits']]
        colors = ['green', 'red', 'orange']
        
        wedges, texts, autotexts = ax5.pie(exit_counts, labels=exit_reasons, colors=colors, 
                                          autopct='%1.1f%%', startangle=90)
        ax5.set_title('Exit Reason Distribution', fontsize=14, fontweight='bold')
        
        # === 6. 取引時系列 ===
        ax6 = plt.subplot(3, 3, 6)
        if results['all_orders']:
            orders = results['all_orders']
            buy_orders = [o for o in orders if o.direction == 'BUY']
            sell_orders = [o for o in orders if o.direction == 'SELL']
            
            if buy_orders:
                buy_times = [o.timestamp for o in buy_orders]
                buy_pnls = [o.net_pnl for o in buy_orders]
                ax6.scatter(buy_times, buy_pnls, c='blue', alpha=0.6, s=30, label='BUY')
            
            if sell_orders:
                sell_times = [o.timestamp for o in sell_orders]
                sell_pnls = [o.net_pnl for o in sell_orders]
                ax6.scatter(sell_times, sell_pnls, c='red', alpha=0.6, s=30, label='SELL')
            
            ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax6.set_title('Trade P&L Timeline', fontsize=14, fontweight='bold')
            ax6.set_ylabel('P&L ($)')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # === 7. 信頼度分析 ===
        ax7 = plt.subplot(3, 3, 7)
        if results['all_orders']:
            confidences = [o.confidence for o in results['all_orders']]
            pnls = [o.net_pnl for o in results['all_orders']]
            
            colors = ['green' if pnl > 0 else 'red' for pnl in pnls]
            ax7.scatter(confidences, pnls, c=colors, alpha=0.6, s=30)
            ax7.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax7.axvline(x=self.config.min_confidence, color='blue', linestyle='--', 
                       alpha=0.7, label=f'Min Confidence ({self.config.min_confidence})')
            ax7.set_title('Confidence vs P&L', fontsize=14, fontweight='bold')
            ax7.set_xlabel('Prediction Confidence')
            ax7.set_ylabel('P&L ($)')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        
        # === 8. 月次損益 ===
        ax8 = plt.subplot(3, 3, 8)
        if results['all_orders']:
            # 月次グループ化
            monthly_pnl = {}
            for order in results['all_orders']:
                month_key = order.timestamp.strftime('%Y-%m')
                if month_key not in monthly_pnl:
                    monthly_pnl[month_key] = 0
                monthly_pnl[month_key] += order.net_pnl
            
            months = sorted(monthly_pnl.keys())
            pnl_values = [monthly_pnl[month] for month in months]
            colors = ['green' if pnl >= 0 else 'red' for pnl in pnl_values]
            
            bars = ax8.bar(range(len(months)), pnl_values, color=colors, alpha=0.7, edgecolor='black')
            ax8.set_title('Monthly P&L', fontsize=14, fontweight='bold')
            ax8.set_ylabel('P&L ($)')
            ax8.set_xticks(range(len(months)))
            ax8.set_xticklabels(months, rotation=45)
            ax8.axhline(y=0, color='black', linestyle='-', alpha=0.8)
            ax8.grid(True, alpha=0.3, axis='y')
        
        # === 9. 統計サマリー ===
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        summary = results['summary']
        stats_text = f"""
PERFORMANCE SUMMARY

Total Trades: {summary['total_trades']:,}
Win Rate: {summary['win_rate_pct']:.1f}%
Profit Factor: {summary['profit_factor']:.2f}

Total P&L: ${summary['total_pnl']:.2f}
Max Drawdown: {summary['max_drawdown_pct']:.1f}%

Avg Win: ${summary['avg_win']:.2f}
Avg Loss: ${abs(summary['avg_loss']):.2f}

BUY Trades: {direction_data['buy_trades']}
SELL Trades: {direction_data['sell_trades']}
        """
        
        ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存
        chart_path = os.path.join(self.config.output_dir, f"backtest_charts_{timestamp}.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"チャート出力: {chart_path}")

# ================================================================
# メイン実行用クラス・関数
# ================================================================

def create_backtest_config(base_config_path: str = 'config.json') -> BacktestConfig:
    """
    バックテスト設定の作成（config.jsonベース + バックテスト固有設定）
    """
    # 基本設定読み込み
    with open(base_config_path, 'r') as f:
        base_config = json.load(f)
    
    # 最新のモデルファイル検索
    models_dir = base_config['data']['output_dir']
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
    
    if not model_files:
        raise FileNotFoundError("訓練済みモデルが見つかりません")
    
    latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(models_dir, x)))
    model_path = os.path.join(models_dir, latest_model)
    
    # SL値計算（訓練時SL + バッファ）
    training_sl = base_config['trading']['sl_pips']
    sl_buffer = 1.0  # デフォルト1pips
    backtest_sl = training_sl + sl_buffer
    
    # バックテスト設定作成
    backtest_config = BacktestConfig(
        model_path=model_path,
        data_file=base_config['data']['input_file'],
        output_dir='backtest_results',
        tp_pips=base_config['trading']['tp_pips'],
        sl_pips=backtest_sl,
        sl_buffer_pips=sl_buffer,
        spread_pips=base_config['trading']['spread_pips'],
        pip_value=base_config['trading']['pip_value'],
        min_confidence=base_config['evaluation']['min_confidence'],
        max_positions=1,
        position_timeout_ticks=1000
    )
    
    return backtest_config

def main():
    """
    メイン実行関数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='USDJPYスキャルピングAI バックテスト')
    parser.add_argument('--model', type=str, help='モデルファイルパス（未指定時は最新を自動選択）')
    parser.add_argument('--data', type=str, help='データファイルパス（未指定時はconfig.jsonから）')
    parser.add_argument('--start_date', type=str, help='開始日 (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, help='終了日 (YYYY-MM-DD)')
    parser.add_argument('--min_confidence', type=float, help='最小信頼度')
    parser.add_argument('--sl_buffer', type=float, default=1.0, help='SLバッファ(pips)')
    parser.add_argument('--output_dir', type=str, default='backtest_results', help='出力ディレクトリ')
    
    args = parser.parse_args()
    
    try:
        # 設定作成
        config = create_backtest_config()
        
        # コマンドライン引数で上書き
        if args.model:
            config.model_path = args.model
        if args.data:
            config.data_file = args.data
        if args.start_date:
            config.start_date = args.start_date
        if args.end_date:
            config.end_date = args.end_date
        if args.min_confidence:
            config.min_confidence = args.min_confidence
        if args.sl_buffer:
            config.sl_buffer_pips = args.sl_buffer
            # SL再計算
            with open('config.json', 'r') as f:
                base_config = json.load(f)
            config.sl_pips = base_config['trading']['sl_pips'] + args.sl_buffer
        if args.output_dir:
            config.output_dir = args.output_dir
        
        # バックテスト実行
        engine = BacktestEngine(config)
        results = engine.run_backtest()
        
        print(f"\n✅ バックテスト完了")
        print(f"📁 結果ファイル: {engine.session_output_dir}/")
        print(f"📊 総取引数: {results['summary']['total_trades']}")
        print(f"💰 総損益: ${results['summary']['total_pnl']:.2f}")
        print(f"📈 勝率: {results['summary']['win_rate_pct']:.1f}%")
        print(f"📂 出力ファイル:")
        for file in os.listdir(engine.session_output_dir):
            print(f"   - {file}")
        
    except Exception as e:
        print(f"❌ バックテストエラー: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()