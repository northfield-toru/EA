"""
USDJPY スキャルピングEA用 ラベリングモジュール - 修正版
重複定義削除・バグ修正済み
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Optional
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from utils import USDJPYUtils

logger = logging.getLogger(__name__)

class ScalpingLabeler:
    """スキャルピング用ラベラー（修正版）"""
    
    def __init__(self, 
                 profit_pips: float = 6.0,
                 loss_pips: float = 6.0,
                 lookforward_ticks: int = 100,
                 spread_pips: float = 0.7,
                 use_or_conditions: bool = True):
        """
        Args:
            profit_pips: 利確目標pips
            loss_pips: 損切りpips
            lookforward_ticks: 未来参照ティック数
            spread_pips: スプレッド（pips）
            use_or_conditions: OR条件を使用するか（True=緩和, False=厳格）
        """
        self.profit_pips = profit_pips
        self.loss_pips = loss_pips
        self.lookforward_ticks = lookforward_ticks
        self.spread_pips = spread_pips
        self.use_or_conditions = use_or_conditions
        self.use_flexible_conditions = use_or_conditions  # 修正: 同じ値を設定
        self.utils = USDJPYUtils()
        
        condition_type = "OR条件(緩和)" if use_or_conditions else "AND条件(厳格)"
        logger.info(f"ラベル設定 - 利確:{profit_pips}pips, 損切:{loss_pips}pips, 前方参照:{lookforward_ticks}ティック, {condition_type}")
    
    def _calculate_future_extremes(self, prices: np.array, start_idx: int) -> Tuple[float, float]:
        """
        指定位置から未来方向の最高値・最安値を計算
        """
        end_idx = min(start_idx + self.lookforward_ticks, len(prices))
        
        if start_idx >= len(prices) - 1:
            return prices[start_idx], prices[start_idx]
        
        future_prices = prices[start_idx + 1:end_idx + 1]
        
        if len(future_prices) == 0:
            return prices[start_idx], prices[start_idx]
        
        return np.max(future_prices), np.min(future_prices)
    
    def _check_buy_condition_strict(self, current_price: float, future_max: float, future_min: float) -> bool:
        """
        厳格なBUY条件（AND条件必須）
        """
        entry_price = current_price + self.utils.pips_to_price(self.spread_pips / 2)
        
        # 利確条件：必須
        profit_target = entry_price + self.utils.pips_to_price(self.profit_pips)
        profit_achieved = future_max >= profit_target
        
        # 損切り条件：必須
        loss_threshold = entry_price - self.utils.pips_to_price(self.loss_pips)
        no_excessive_loss = future_min >= loss_threshold
        
        # AND条件のみ
        return profit_achieved and no_excessive_loss
    
    def _check_sell_condition_strict(self, current_price: float, future_max: float, future_min: float) -> bool:
        """
        厳格なSELL条件（AND条件必須）
        """
        entry_price = current_price - self.utils.pips_to_price(self.spread_pips / 2)
        
        # 利確条件：必須
        profit_target = entry_price - self.utils.pips_to_price(self.profit_pips)
        profit_achieved = future_min <= profit_target
        
        # 損切り条件：必須
        loss_threshold = entry_price + self.utils.pips_to_price(self.loss_pips)
        no_excessive_loss = future_max <= loss_threshold
        
        # AND条件のみ
        return profit_achieved and no_excessive_loss
    
    def create_binary_labels_strict(self, df: pd.DataFrame, price_col: str = 'close') -> pd.Series:
        """
        厳格な2値分類ラベル生成（AND条件必須）
        目標：TRADE 10-30%, NO_TRADE 70-90%
        """
        logger.info(f"厳格2値分類ラベル生成開始: {len(df)} 行")
        
        if len(df) == 0:
            return pd.Series([], dtype=int, name='strict_binary_label')
        
        prices = df[price_col].values
        labels = np.zeros(len(prices), dtype=int)  # 0=NO_TRADE
        
        trade_count = 0
        no_trade_count = 0
        
        for i in range(len(prices) - 1):
            try:
                future_max, future_min = self._calculate_future_extremes(prices, i)
                
                # 厳格な条件でBUY/SELLをチェック
                buy_viable = self._check_buy_condition_strict(prices[i], future_max, future_min)
                sell_viable = self._check_sell_condition_strict(prices[i], future_max, future_min)
                
                # いずれかの方向でTRADE条件を満たせばTRADE
                if buy_viable or sell_viable:
                    labels[i] = 1  # TRADE
                    trade_count += 1
                else:
                    labels[i] = 0  # NO_TRADE
                    no_trade_count += 1
                    
            except Exception as e:
                if trade_count + no_trade_count < 10:
                    logger.warning(f"行 {i} の処理でエラー: {e}")
                labels[i] = 0  # エラー時はNO_TRADE
                no_trade_count += 1
        
        # 最後の行
        labels[-1] = 0
        no_trade_count += 1
        
        # 統計表示
        total = len(labels)
        trade_ratio = trade_count / total
        no_trade_ratio = no_trade_count / total
        
        logger.info(f"厳格ラベル統計:")
        logger.info(f"  TRADE: {trade_count:,} ({trade_ratio:.1%})")
        logger.info(f"  NO_TRADE: {no_trade_count:,} ({no_trade_ratio:.1%})")
        
        return pd.Series(labels, index=df.index, name='strict_binary_label')
    
    def _check_buy_condition_profit_focused(self, current_price: float, future_max: float, future_min: float) -> bool:
        """
        勝率重視のBUY条件（利確9pips, 損切り3pips）
        """
        entry_price = current_price + self.utils.pips_to_price(self.spread_pips / 2)
        
        # より高い利確目標（9pips）
        profit_target = entry_price + self.utils.pips_to_price(9.0)
        profit_achieved = future_max >= profit_target
        
        # より狭い損切り許容（3pips）
        loss_threshold = entry_price - self.utils.pips_to_price(3.0)
        no_excessive_loss = future_min >= loss_threshold
        
        # 厳格なAND条件
        return profit_achieved and no_excessive_loss
    
    def _check_sell_condition_profit_focused(self, current_price: float, future_max: float, future_min: float) -> bool:
        """
        勝率重視のSELL条件（利確9pips, 損切り3pips）
        """
        entry_price = current_price - self.utils.pips_to_price(self.spread_pips / 2)
        
        # より高い利確目標（9pips）
        profit_target = entry_price - self.utils.pips_to_price(9.0)
        profit_achieved = future_min <= profit_target
        
        # より狭い損切り許容（3pips）
        loss_threshold = entry_price + self.utils.pips_to_price(3.0)
        no_excessive_loss = future_max <= loss_threshold
        
        # 厳格なAND条件
        return profit_achieved and no_excessive_loss
    
    def create_profit_focused_labels(self, df: pd.DataFrame, price_col: str = 'close') -> pd.Series:
        """
        勝率重視の2値分類ラベル生成（利確9pips, 損切り3pips）
        """
        logger.info(f"勝率重視ラベル生成開始: {len(df)} 行")
        logger.info("条件: 利確9pips, 損切り3pips, AND条件")
        
        if len(df) == 0:
            return pd.Series([], dtype=int, name='profit_focused_label')
        
        prices = df[price_col].values
        labels = np.zeros(len(prices), dtype=int)
        
        high_confidence_trade = 0
        conservative_no_trade = 0
        
        for i in range(len(prices) - 1):
            try:
                future_max, future_min = self._calculate_future_extremes(prices, i)
                
                # 勝率重視の厳格条件でチェック
                buy_viable = self._check_buy_condition_profit_focused(prices[i], future_max, future_min)
                sell_viable = self._check_sell_condition_profit_focused(prices[i], future_max, future_min)
                
                # 高確信度シグナルのみTRADE
                if buy_viable or sell_viable:
                    labels[i] = 1  # 高確信度TRADE
                    high_confidence_trade += 1
                else:
                    labels[i] = 0  # 保守的NO_TRADE
                    conservative_no_trade += 1
                    
            except Exception as e:
                if high_confidence_trade + conservative_no_trade < 10:
                    logger.warning(f"行 {i} の処理でエラー: {e}")
                labels[i] = 0  # エラー時は保守的にNO_TRADE
                conservative_no_trade += 1
        
        # 最後の行
        labels[-1] = 0
        conservative_no_trade += 1
        
        # 統計分析
        total = len(labels)
        trade_ratio = high_confidence_trade / total
        
        logger.info(f"勝率重視ラベル統計:")
        logger.info(f"  高確信TRADE: {high_confidence_trade:,} ({trade_ratio:.1%})")
        logger.info(f"  保守的NO_TRADE: {conservative_no_trade:,} ({(1-trade_ratio):.1%})")
        
        return pd.Series(labels, index=df.index, name='profit_focused_label')
    
    def create_realistic_profit_labels(self, df: pd.DataFrame, tp_pips: float = 4.0, sl_pips: float = 5.0, price_col: str = 'close') -> pd.Series:
        """
        現実的利益ラベル生成（バックテストと同じTP/SL設定）
        """
        logger.info(f"現実的利益ラベル生成開始: {len(df)} 行")
        logger.info(f"設定: 利確{tp_pips}pips, 損切り{sl_pips}pips")
        
        if len(df) == 0:
            return pd.Series([], dtype=int, name='realistic_profit_label')
        
        prices = df[price_col].values
        labels = np.zeros(len(prices), dtype=int)
        
        realistic_trade_count = 0
        no_trade_count = 0
        
        for i in range(len(prices) - 1):
            try:
                future_max, future_min = self._calculate_future_extremes(prices, i)
                
                # BUY方向の判定
                buy_entry = prices[i] + self.utils.pips_to_price(self.spread_pips / 2)
                buy_tp_target = buy_entry + self.utils.pips_to_price(tp_pips)
                buy_sl_threshold = buy_entry - self.utils.pips_to_price(sl_pips)
                
                buy_hits_tp = future_max >= buy_tp_target
                buy_avoids_sl = future_min >= buy_sl_threshold
                
                # SELL方向の判定
                sell_entry = prices[i] - self.utils.pips_to_price(self.spread_pips / 2)
                sell_tp_target = sell_entry - self.utils.pips_to_price(tp_pips)
                sell_sl_threshold = sell_entry + self.utils.pips_to_price(sl_pips)
                
                sell_hits_tp = future_min <= sell_tp_target
                sell_avoids_sl = future_max <= sell_sl_threshold
                
                # 現実的条件: TP到達 AND SL回避
                buy_realistic = buy_hits_tp and buy_avoids_sl
                sell_realistic = sell_hits_tp and sell_avoids_sl
                
                if buy_realistic or sell_realistic:
                    labels[i] = 1  # TRADE
                    realistic_trade_count += 1
                else:
                    labels[i] = 0  # NO_TRADE
                    no_trade_count += 1
                    
            except Exception as e:
                if realistic_trade_count + no_trade_count < 10:
                    logger.warning(f"行 {i} の処理でエラー: {e}")
                labels[i] = 0
                no_trade_count += 1
        
        # 最後の行
        labels[-1] = 0
        no_trade_count += 1
        
        # 統計表示
        total = len(labels)
        trade_ratio = realistic_trade_count / total
        
        logger.info(f"現実的利益ラベル統計:")
        logger.info(f"  TRADE: {realistic_trade_count:,} ({trade_ratio:.1%})")
        logger.info(f"  NO_TRADE: {no_trade_count:,} ({(1-trade_ratio):.1%})")
        
        return pd.Series(labels, index=df.index, name='realistic_profit_label')
    
    def create_ultra_conservative_labels(self, df: pd.DataFrame, price_col: str = 'close') -> pd.Series:
        """
        超保守的なラベル生成（利確12pips, 損切り2pips）
        """
        logger.info(f"超保守的ラベル生成開始: {len(df)} 行")
        logger.info("条件: 利確12pips, 損切り2pips, 超厳格")
        
        if len(df) == 0:
            return pd.Series([], dtype=int, name='ultra_conservative_label')
        
        prices = df[price_col].values
        labels = np.zeros(len(prices), dtype=int)
        
        ultra_trade = 0
        
        for i in range(len(prices) - 1):
            try:
                future_max, future_min = self._calculate_future_extremes(prices, i)
                
                # 超厳格BUY条件
                buy_entry = prices[i] + self.utils.pips_to_price(self.spread_pips / 2)
                buy_big_profit = buy_entry + self.utils.pips_to_price(12.0)
                buy_tiny_loss = buy_entry - self.utils.pips_to_price(2.0)
                buy_ultra = (future_max >= buy_big_profit) and (future_min >= buy_tiny_loss)
                
                # 超厳格SELL条件
                sell_entry = prices[i] - self.utils.pips_to_price(self.spread_pips / 2)
                sell_big_profit = sell_entry - self.utils.pips_to_price(12.0)
                sell_tiny_loss = sell_entry + self.utils.pips_to_price(2.0)
                sell_ultra = (future_min <= sell_big_profit) and (future_max <= sell_tiny_loss)
                
                if buy_ultra or sell_ultra:
                    labels[i] = 1  # 超高確信TRADE
                    ultra_trade += 1
            except Exception as e:
                if ultra_trade < 10:
                    logger.warning(f"行 {i} の処理でエラー: {e}")
                continue
        
        logger.info(f"超保守統計: 超高確信TRADE {ultra_trade} ({ultra_trade/len(labels):.1%})")
        
        return pd.Series(labels, index=df.index, name='ultra_conservative_label')

    # その他の既存メソッドは簡略化版のみ残す
    def create_binary_labels_vectorized(self, df: pd.DataFrame, price_col: str = 'close') -> pd.Series:
        """基本的な2値分類ラベル生成"""
        logger.info(f"基本2値分類ラベル生成開始: {len(df)} 行")
        
        if len(df) == 0:
            return pd.Series([], dtype=int, name='binary_label')
        
        prices = df[price_col].values
        labels = np.zeros(len(prices), dtype=int)
        
        for i in range(len(prices) - 1):
            try:
                future_max, future_min = self._calculate_future_extremes(prices, i)
                
                # 基本的なBUY/SELL条件
                buy_viable = self._check_buy_condition_strict(prices[i], future_max, future_min)
                sell_viable = self._check_sell_condition_strict(prices[i], future_max, future_min)
                
                if buy_viable or sell_viable:
                    labels[i] = 1  # TRADE
                    
            except Exception as e:
                continue
        
        return pd.Series(labels, index=df.index, name='binary_label')

if __name__ == "__main__":
    print("修正版ラベリングモジュール - 重複削除・バグ修正済み")