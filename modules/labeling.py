import numpy as np
import pandas as pd
import logging
from typing import Tuple, List

class LabelCreator:
    """ラベル作成クラス"""
    
    def __init__(self, config):
        self.config = config
        self.tp_pips = config['trading']['tp_pips']
        self.sl_pips = config['trading']['sl_pips']
        self.pip_value = config['trading']['pip_value']
        self.future_window = config['data']['future_window']
        self.sample_rate = config['data']['sample_rate']
        
        # サンプリングに応じてfuture_windowを調整
        self.adjusted_future_window = max(1, self.future_window // self.sample_rate)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Label parameters - TP: {self.tp_pips}pips, SL: {self.sl_pips}pips")
        self.logger.info(f"Future window: {self.future_window} -> {self.adjusted_future_window} (adjusted for sample_rate: {self.sample_rate})")
    
    def create_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        ラベルを作成（3クラス分類）
        
        Args:
            df: ティックデータ（MID価格含む）
            
        Returns:
            np.ndarray: ラベル配列 (0: BUY, 1: SELL, 2: NO_TRADE)
        """
        labels = []
        mid_prices = df['MID'].values
        
        self.logger.info(f"Creating labels for {len(df)} samples...")
        
        for i in range(len(df)):
            # 未来データの範囲を計算
            future_end = min(i + self.adjusted_future_window, len(df))
            
            if future_end <= i:
                # 未来データが不足している場合はNO_TRADE
                labels.append(2)
                continue
            
            # 現在価格
            current_price = mid_prices[i]
            
            # 未来の価格範囲を取得
            future_prices = mid_prices[i+1:future_end]
            
            if len(future_prices) == 0:
                labels.append(2)
                continue
            
            # BUYシナリオ判定
            buy_label = self._check_buy_scenario(current_price, future_prices)
            
            # SELLシナリオ判定
            sell_label = self._check_sell_scenario(current_price, future_prices)
            
            # ラベル決定ロジック
            if buy_label and not sell_label:
                labels.append(0)  # BUY
            elif sell_label and not buy_label:
                labels.append(1)  # SELL
            else:
                labels.append(2)  # NO_TRADE
        
        labels = np.array(labels)
        
        # ラベル分布をログ出力
        unique, counts = np.unique(labels, return_counts=True)
        label_names = {0: 'BUY', 1: 'SELL', 2: 'NO_TRADE'}
        
        self.logger.info("Label distribution:")
        for label, count in zip(unique, counts):
            percentage = count / len(labels) * 100
            self.logger.info(f"  {label_names[label]}: {count} ({percentage:.2f}%)")
        
        return labels
    
    def _check_buy_scenario(self, current_price: float, future_prices: np.ndarray) -> bool:
        """
        BUYシナリオをチェック
        
        条件:
        - 将来価格が+4.0pips上昇する
        - かつ、-5.0pips下落しない
        
        Args:
            current_price: 現在価格
            future_prices: 将来価格配列
            
        Returns:
            bool: BUYシナリオに該当するかどうか
        """
        tp_price = current_price + (self.tp_pips * self.pip_value)
        sl_price = current_price - (self.sl_pips * self.pip_value)
        
        # 各時点での利益/損失チェック
        tp_hit = False
        sl_hit = False
        
        for price in future_prices:
            if price >= tp_price:
                tp_hit = True
                break  # TP達成時点で判定終了
            elif price <= sl_price:
                sl_hit = True
                break  # SL到達時点で判定終了
        
        # BUY条件: TP到達 かつ SL非到達
        return tp_hit and not sl_hit
    
    def _check_sell_scenario(self, current_price: float, future_prices: np.ndarray) -> bool:
        """
        SELLシナリオをチェック
        
        条件:
        - 将来価格が-4.0pips下落する
        - かつ、+5.0pips上昇しない
        
        Args:
            current_price: 現在価格
            future_prices: 将来価格配列
            
        Returns:
            bool: SELLシナリオに該当するかどうか
        """
        tp_price = current_price - (self.tp_pips * self.pip_value)
        sl_price = current_price + (self.sl_pips * self.pip_value)
        
        # 各時点での利益/損失チェック
        tp_hit = False
        sl_hit = False
        
        for price in future_prices:
            if price <= tp_price:
                tp_hit = True
                break  # TP達成時点で判定終了
            elif price >= sl_price:
                sl_hit = True
                break  # SL到達時点で判定終了
        
        # SELL条件: TP到達 かつ SL非到達
        return tp_hit and not sl_hit
    
    def get_valid_data_range(self, total_length: int) -> int:
        """
        有効なデータ範囲を取得（未来リーク防止）
        
        Args:
            total_length: 全データ長
            
        Returns:
            int: 有効なデータの終了インデックス
        """
        # 最後のadjusted_future_window分は未来データが不足するため除外
        valid_end = total_length - self.adjusted_future_window
        
        if valid_end <= 0:
            raise ValueError(f"Data too short for future_window: {self.adjusted_future_window}")
        
        self.logger.info(f"Valid data range: 0 to {valid_end} (excluded last {self.adjusted_future_window} samples)")
        
        return valid_end
    
    def analyze_label_quality(self, labels: np.ndarray, df: pd.DataFrame) -> dict:
        """
        ラベル品質分析
        
        Args:
            labels: ラベル配列
            df: 元データ
            
        Returns:
            dict: 分析結果
        """
        analysis = {}
        
        # 基本統計
        unique, counts = np.unique(labels, return_counts=True)
        label_names = {0: 'BUY', 1: 'SELL', 2: 'NO_TRADE'}
        
        for label, count in zip(unique, counts):
            analysis[f'{label_names[label]}_count'] = int(count)
            analysis[f'{label_names[label]}_ratio'] = float(count / len(labels))
        
        # クラスバランス分析
        buy_ratio = analysis.get('BUY_ratio', 0)
        sell_ratio = analysis.get('SELL_ratio', 0)
        no_trade_ratio = analysis.get('NO_TRADE_ratio', 0)
        
        # バランス指標
        analysis['class_balance_score'] = 1 - max(buy_ratio, sell_ratio, no_trade_ratio)
        analysis['trade_ratio'] = buy_ratio + sell_ratio
        
        # 警告判定
        warnings = []
        
        if buy_ratio < 0.05 or sell_ratio < 0.05:
            warnings.append("極端にトレード信号が少ない")
        
        if no_trade_ratio > 0.95:
            warnings.append("NO_TRADEが支配的（95%以上）")
        
        if abs(buy_ratio - sell_ratio) > 0.3:
            warnings.append("BUY/SELL比率が大きく偏っている")
        
        analysis['warnings'] = warnings
        
        # ログ出力
        self.logger.info("Label quality analysis:")
        for key, value in analysis.items():
            if key != 'warnings':
                self.logger.info(f"  {key}: {value}")
        
        if warnings:
            self.logger.warning("Label quality warnings:")
            for warning in warnings:
                self.logger.warning(f"  - {warning}")
        
        return analysis
    
    def create_backtest_labels(self, df: pd.DataFrame, start_idx: int = 0) -> Tuple[np.ndarray, List[dict]]:
        """
        バックテスト用詳細ラベル作成
        
        Args:
            df: ティックデータ
            start_idx: 開始インデックス
            
        Returns:
            Tuple: (ラベル配列, 詳細情報リスト)
        """
        labels = []
        details = []
        mid_prices = df['MID'].values
        
        for i in range(start_idx, len(df)):
            future_end = min(i + self.adjusted_future_window, len(df))
            
            if future_end <= i:
                labels.append(2)
                details.append({
                    'reason': 'insufficient_future_data',
                    'tp_hit_time': None,
                    'sl_hit_time': None,
                    'max_profit_pips': 0,
                    'max_loss_pips': 0
                })
                continue
            
            current_price = mid_prices[i]
            future_prices = mid_prices[i+1:future_end]
            
            # 詳細分析
            detail = self._analyze_price_movement(current_price, future_prices)
            
            # ラベル決定
            if detail['buy_valid'] and not detail['sell_valid']:
                labels.append(0)  # BUY
            elif detail['sell_valid'] and not detail['buy_valid']:
                labels.append(1)  # SELL
            else:
                labels.append(2)  # NO_TRADE
            
            details.append(detail)
        
        return np.array(labels), details
    
    def _analyze_price_movement(self, current_price: float, future_prices: np.ndarray) -> dict:
        """価格変動の詳細分析"""
        tp_buy = current_price + (self.tp_pips * self.pip_value)
        sl_buy = current_price - (self.sl_pips * self.pip_value)
        tp_sell = current_price - (self.tp_pips * self.pip_value)
        sl_sell = current_price + (self.sl_pips * self.pip_value)
        
        # 変動追跡
        max_profit_buy = 0
        max_loss_buy = 0
        max_profit_sell = 0
        max_loss_sell = 0
        
        buy_tp_time = None
        buy_sl_time = None
        sell_tp_time = None
        sell_sl_time = None
        
        for t, price in enumerate(future_prices):
            # BUY分析
            profit_buy = (price - current_price) / self.pip_value
            max_profit_buy = max(max_profit_buy, profit_buy)
            max_loss_buy = min(max_loss_buy, profit_buy)
            
            if price >= tp_buy and buy_tp_time is None:
                buy_tp_time = t
            if price <= sl_buy and buy_sl_time is None:
                buy_sl_time = t
            
            # SELL分析
            profit_sell = (current_price - price) / self.pip_value
            max_profit_sell = max(max_profit_sell, profit_sell)
            max_loss_sell = min(max_loss_sell, profit_sell)
            
            if price <= tp_sell and sell_tp_time is None:
                sell_tp_time = t
            if price >= sl_sell and sell_sl_time is None:
                sell_sl_time = t
        
        return {
            'buy_valid': buy_tp_time is not None and buy_sl_time is None,
            'sell_valid': sell_tp_time is not None and sell_sl_time is None,
            'buy_tp_time': buy_tp_time,
            'buy_sl_time': buy_sl_time,
            'sell_tp_time': sell_tp_time,
            'sell_sl_time': sell_sl_time,
            'max_profit_buy_pips': max_profit_buy,
            'max_loss_buy_pips': max_loss_buy,
            'max_profit_sell_pips': max_profit_sell,
            'max_loss_sell_pips': max_loss_sell,
            'reason': 'analyzed'
        }