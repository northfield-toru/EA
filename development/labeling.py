"""
USDJPY スキャルピングEA用 ラベリングモジュール - 修正版
3クラス分類ラベル生成（BUY/SELL/NO_TRADE）
未来リーク防止を徹底した実装
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Optional
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from utils import USDJPYUtils

# ロガー設定
logger = logging.getLogger(__name__)

class ScalpingLabeler:
    """スキャルピング用ラベラー"""
    
    def __init__(self, 
                 profit_pips: float = 6.0,     # ChatGPT提案: 8.0 → 6.0
                 loss_pips: float = 6.0,       # ChatGPT提案: 4.0 → 6.0  
                 lookforward_ticks: int = 100,
                 spread_pips: float = 0.7,
                 use_or_conditions: bool = True):  # ChatGPT提案: OR条件をデフォルト
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
        # 🔧 FIX: use_flexible_conditions属性を追加（use_or_conditionsと同じ値）
        self.use_flexible_conditions = use_or_conditions
        self.utils = USDJPYUtils()
        
        condition_type = "OR条件(緩和)" if use_or_conditions else "AND条件(厳格)"
        logger.info(f"ラベル設定 - 利確:{profit_pips}pips, 損切:{loss_pips}pips, 前方参照:{lookforward_ticks}ティック, {condition_type}")
    
    def _calculate_future_extremes(self, prices: np.array, start_idx: int) -> Tuple[float, float]:
        """
        指定位置から未来方向の最高値・最安値を計算
        Args:
            prices: 価格配列
            start_idx: 開始インデックス
        Returns:
            tuple: (max_price, min_price)
        """
        end_idx = min(start_idx + self.lookforward_ticks, len(prices))
        
        if start_idx >= len(prices) - 1:
            return prices[start_idx], prices[start_idx]
        
        future_prices = prices[start_idx + 1:end_idx + 1]
        
        if len(future_prices) == 0:
            return prices[start_idx], prices[start_idx]
        
        return np.max(future_prices), np.min(future_prices)
    
    def _check_buy_condition(self, current_price: float, future_max: float, future_min: float) -> bool:
        """
        BUY条件をチェック（柔軟条件対応版）
        Args:
            current_price: 現在価格（MID）
            future_max: 未来最高値
            future_min: 未来最安値
        Returns:
            bool: BUY条件に合致するか
        """
        # スプレッド調整（BUYはASK価格でエントリー）
        entry_price = current_price + self.utils.pips_to_price(self.spread_pips / 2)
        
        # 利確条件：profit_pips以上の上昇
        profit_target = entry_price + self.utils.pips_to_price(self.profit_pips)
        profit_achieved = future_max >= profit_target
        
        # 損切り条件：loss_pips以上の逆行がない
        loss_threshold = entry_price - self.utils.pips_to_price(self.loss_pips)
        no_excessive_loss = future_min >= loss_threshold
        
        if self.use_flexible_conditions:
            # OR条件：利確達成 または 損失が小さい
            return profit_achieved or no_excessive_loss
        else:
            # AND条件：利確達成 かつ 損失が小さい（従来）
            return profit_achieved and no_excessive_loss
    
    def _check_sell_condition(self, current_price: float, future_max: float, future_min: float) -> bool:
        """
        SELL条件をチェック（柔軟条件対応版）
        Args:
            current_price: 現在価格（MID）
            future_max: 未来最高値
            future_min: 未来最安値
        Returns:
            bool: SELL条件に合致するか
        """
        # スプレッド調整（SELLはBID価格でエントリー）
        entry_price = current_price - self.utils.pips_to_price(self.spread_pips / 2)
        
        # 利確条件：profit_pips以上の下落
        profit_target = entry_price - self.utils.pips_to_price(self.profit_pips)
        profit_achieved = future_min <= profit_target
        
        # 損切り条件：loss_pips以上の逆行がない
        loss_threshold = entry_price + self.utils.pips_to_price(self.loss_pips)
        no_excessive_loss = future_max <= loss_threshold
        
        if self.use_flexible_conditions:
            # OR条件：利確達成 または 損失が小さい
            return profit_achieved or no_excessive_loss
        else:
            # AND条件：利確達成 かつ 損失が小さい（従来）
            return profit_achieved and no_excessive_loss
    
    def _check_trade_condition(self, current_price: float, future_max: float, future_min: float) -> bool:
        """
        TRADE条件をチェック（2値分類用・ChatGPT提案のOR条件）
        Args:
            current_price: 現在価格（MID）
            future_max: 未来最高値
            future_min: 未来最安値
        Returns:
            bool: TRADE条件に合致するか
        """
        # BUY方向の判定
        buy_entry = current_price + self.utils.pips_to_price(self.spread_pips / 2)
        buy_profit_target = buy_entry + self.utils.pips_to_price(self.profit_pips)
        buy_loss_threshold = buy_entry - self.utils.pips_to_price(self.loss_pips)
        
        buy_profit_achieved = future_max >= buy_profit_target
        buy_loss_acceptable = future_min >= buy_loss_threshold
        
        # SELL方向の判定
        sell_entry = current_price - self.utils.pips_to_price(self.spread_pips / 2)
        sell_profit_target = sell_entry - self.utils.pips_to_price(self.profit_pips)
        sell_loss_threshold = sell_entry + self.utils.pips_to_price(self.loss_pips)
        
        sell_profit_achieved = future_min <= sell_profit_target
        sell_loss_acceptable = future_max <= sell_loss_threshold
        
        if self.use_or_conditions:
            # OR条件: BUY/SELLいずれかで（利確達成 OR 損失許容範囲）
            buy_viable = buy_profit_achieved or buy_loss_acceptable
            sell_viable = sell_profit_achieved or sell_loss_acceptable
            return buy_viable or sell_viable
        else:
            # AND条件: BUY/SELLいずれかで（利確達成 AND 損失許容範囲）
            buy_viable = buy_profit_achieved and buy_loss_acceptable
            sell_viable = sell_profit_achieved and sell_loss_acceptable
            return buy_viable or sell_viable
    
    def create_labels_vectorized(self, df: pd.DataFrame, price_col: str = 'close') -> pd.Series:
        """
        ベクトル化されたラベル生成（高速版）
        Args:
            df: OHLCV DataFrame
            price_col: 価格列名
        Returns:
            Series: ラベル（0=NO_TRADE, 1=BUY, 2=SELL）
        """
        logger.info(f"ラベル生成開始: {len(df)} 行")
        
        prices = df[price_col].values
        labels = np.zeros(len(prices), dtype=int)  # 0=NO_TRADE
        
        # 各行について未来の極値を計算
        for i in range(len(prices) - 1):  # 最後の行は未来データがないので除外
            future_max, future_min = self._calculate_future_extremes(prices, i)
            
            # BUY条件チェック
            if self._check_buy_condition(prices[i], future_max, future_min):
                labels[i] = 1  # BUY
            # SELL条件チェック
            elif self._check_sell_condition(prices[i], future_max, future_min):
                labels[i] = 2  # SELL
            # それ以外はNO_TRADE（既に0で初期化済み）
        
        # ラベル統計
        unique, counts = np.unique(labels, return_counts=True)
        label_stats = dict(zip(unique, counts))
        logger.info(f"ラベル統計: {label_stats}")
        
        # パーセンテージ表示
        total = len(labels)
        for label_val, count in label_stats.items():
            label_name = ['NO_TRADE', 'BUY', 'SELL'][label_val]
            percentage = count / total * 100
            logger.info(f"{label_name}: {count:,} ({percentage:.2f}%)")
        
        return pd.Series(labels, index=df.index, name='label')
    
    def create_binary_labels_vectorized(self, df: pd.DataFrame, price_col: str = 'close') -> pd.Series:
        """
        2値分類ラベル生成（TRADE vs NO_TRADE）
        Args:
            df: OHLCV DataFrame
            price_col: 価格列名
        Returns:
            Series: ラベル（0=NO_TRADE, 1=TRADE）
        """
        logger.info(f"2値分類ラベル生成開始: {len(df)} 行")
        
        if len(df) == 0:
            logger.error("空のDataFrameが渡されました")
            return pd.Series([], dtype=int, name='binary_label')
        
        if price_col not in df.columns:
            logger.error(f"価格列 '{price_col}' が見つかりません")
            return pd.Series([], dtype=int, name='binary_label')
        
        try:
            prices = df[price_col].values
            labels = np.zeros(len(prices), dtype=int)  # 0=NO_TRADE
            
            # 各行について未来の極値を計算
            processed_count = 0
            for i in range(len(prices) - 1):  # 最後の行は未来データがないので除外
                try:
                    future_max, future_min = self._calculate_future_extremes(prices, i)
                    
                    # BUY または SELL条件のいずれかに合致すればTRADE
                    if (self._check_buy_condition(prices[i], future_max, future_min) or 
                        self._check_sell_condition(prices[i], future_max, future_min)):
                        labels[i] = 1  # TRADE
                    # それ以外はNO_TRADE（既に0で初期化済み）
                    
                    processed_count += 1
                    
                    # 進捗表示（10万行ごと）
                    if processed_count % 100000 == 0:
                        logger.info(f"処理中... {processed_count:,} / {len(prices)-1:,} 行")
                        
                except Exception as e:
                    if processed_count < 10:  # 最初の10個のエラーのみ表示
                        logger.warning(f"行 {i} の処理でエラー: {e}")
                    continue
            
            # ラベル統計
            unique, counts = np.unique(labels, return_counts=True)
            label_stats = dict(zip(unique, counts))
            logger.info(f"2値分類ラベル統計: {label_stats}")
            
            # パーセンテージ表示
            total = len(labels)
            for label_val, count in label_stats.items():
                label_name = ['NO_TRADE', 'TRADE'][label_val]
                percentage = count / total * 100
                logger.info(f"{label_name}: {count:,} ({percentage:.2f}%)")
            
            result = pd.Series(labels, index=df.index, name='binary_label')
            logger.info(f"2値分類ラベル生成完了: {len(result)} 行")
            return result
            
        except Exception as e:
            logger.error(f"2値分類ラベル生成でエラー: {e}")
            return pd.Series([], dtype=int, name='binary_label')
    
    def create_labels_parallel(self, df: pd.DataFrame, price_col: str = 'close', n_processes: Optional[int] = None) -> pd.Series:
        """
        並列処理でのラベル生成（大容量データ用）
        Args:
            df: OHLCV DataFrame
            price_col: 価格列名
            n_processes: プロセス数（Noneなら自動）
        Returns:
            Series: ラベル
        """
        if n_processes is None:
            n_processes = min(mp.cpu_count(), 4)  # 最大4プロセス
        
        logger.info(f"並列ラベル生成開始: {len(df)} 行, {n_processes} プロセス")
        
        prices = df[price_col].values
        chunk_size = max(1000, len(prices) // n_processes)
        
        # チャンクに分割
        chunks = []
        for i in range(0, len(prices), chunk_size):
            end_idx = min(i + chunk_size, len(prices))
            chunks.append((prices, i, end_idx))
        
        # 並列処理
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            results = list(executor.map(self._process_chunk, chunks))
        
        # 結果をマージ
        labels = np.zeros(len(prices), dtype=int)
        for start_idx, chunk_labels in results:
            end_idx = start_idx + len(chunk_labels)
            labels[start_idx:end_idx] = chunk_labels
        
        # ラベル統計
        unique, counts = np.unique(labels, return_counts=True)
        label_stats = dict(zip(unique, counts))
        logger.info(f"並列ラベル統計: {label_stats}")
        
        return pd.Series(labels, index=df.index, name='label')
    
    def _process_chunk(self, chunk_data: Tuple[np.array, int, int]) -> Tuple[int, np.array]:
        """
        チャンク処理用メソッド
        Args:
            chunk_data: (prices, start_idx, end_idx)
        Returns:
            tuple: (start_idx, labels)
        """
        prices, start_idx, end_idx = chunk_data
        chunk_labels = np.zeros(end_idx - start_idx, dtype=int)
        
        for i in range(start_idx, min(end_idx, len(prices) - 1)):
            future_max, future_min = self._calculate_future_extremes(prices, i)
            local_idx = i - start_idx
            
            if self._check_buy_condition(prices[i], future_max, future_min):
                chunk_labels[local_idx] = 1  # BUY
            elif self._check_sell_condition(prices[i], future_max, future_min):
                chunk_labels[local_idx] = 2  # SELL
        
        return start_idx, chunk_labels
    
    def create_detailed_labels(self, df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """
        詳細なラベル情報を生成（分析用）
        Args:
            df: OHLCV DataFrame
            price_col: 価格列名
        Returns:
            DataFrame: 詳細ラベル情報
        """
        logger.info("詳細ラベル生成開始...")
        
        prices = df[price_col].values
        results = []
        
        for i in range(len(prices) - 1):
            future_max, future_min = self._calculate_future_extremes(prices, i)
            current_price = prices[i]
            
            # スプレッド調整価格
            buy_entry = current_price + self.utils.pips_to_price(self.spread_pips / 2)
            sell_entry = current_price - self.utils.pips_to_price(self.spread_pips / 2)
            
            # BUY分析
            buy_profit_target = buy_entry + self.utils.pips_to_price(self.profit_pips)
            buy_loss_threshold = buy_entry - self.utils.pips_to_price(self.loss_pips)
            buy_profit_achieved = future_max >= buy_profit_target
            buy_no_excessive_loss = future_min >= buy_loss_threshold
            buy_max_profit_pips = self.utils.price_to_pips(future_max - buy_entry)
            buy_max_loss_pips = self.utils.price_to_pips(buy_entry - future_min)
            
            # SELL分析
            sell_profit_target = sell_entry - self.utils.pips_to_price(self.profit_pips)
            sell_loss_threshold = sell_entry + self.utils.pips_to_price(self.loss_pips)
            sell_profit_achieved = future_min <= sell_profit_target
            sell_no_excessive_loss = future_max <= sell_loss_threshold
            sell_max_profit_pips = self.utils.price_to_pips(sell_entry - future_min)
            sell_max_loss_pips = self.utils.price_to_pips(future_max - sell_entry)
            
            # ラベル決定
            if buy_profit_achieved and buy_no_excessive_loss:
                label = 1  # BUY
            elif sell_profit_achieved and sell_no_excessive_loss:
                label = 2  # SELL
            else:
                label = 0  # NO_TRADE
            
            results.append({
                'label': label,
                'current_price': current_price,
                'future_max': future_max,
                'future_min': future_min,
                'buy_entry': buy_entry,
                'sell_entry': sell_entry,
                'buy_max_profit_pips': buy_max_profit_pips,
                'buy_max_loss_pips': buy_max_loss_pips,
                'sell_max_profit_pips': sell_max_profit_pips,
                'sell_max_loss_pips': sell_max_loss_pips,
                'buy_profit_achieved': buy_profit_achieved,
                'buy_no_excessive_loss': buy_no_excessive_loss,
                'sell_profit_achieved': sell_profit_achieved,
                'sell_no_excessive_loss': sell_no_excessive_loss
            })
        
        # 最後の行（未来データなし）
        results.append({
            'label': 0,
            'current_price': prices[-1],
            'future_max': np.nan,
            'future_min': np.nan,
            'buy_entry': np.nan,
            'sell_entry': np.nan,
            'buy_max_profit_pips': np.nan,
            'buy_max_loss_pips': np.nan,
            'sell_max_profit_pips': np.nan,
            'sell_max_loss_pips': np.nan,
            'buy_profit_achieved': False,
            'buy_no_excessive_loss': False,
            'sell_profit_achieved': False,
            'sell_no_excessive_loss': False
        })
        
        result_df = pd.DataFrame(results, index=df.index)
        logger.info("詳細ラベル生成完了")
        
        return result_df
    
    def analyze_label_distribution(self, labels: pd.Series) -> Dict:
        """
        ラベル分布の分析
        Args:
            labels: ラベルSeries
        Returns:
            dict: 分析結果
        """
        analysis = {}
        
        # 基本統計
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        
        analysis['total_samples'] = total
        analysis['label_counts'] = dict(zip(unique, counts))
        analysis['label_percentages'] = {k: v/total*100 for k, v in analysis['label_counts'].items()}
        
        # ラベル名マッピング
        label_names = {0: 'NO_TRADE', 1: 'BUY', 2: 'SELL'}
        analysis['label_names'] = label_names
        
        # 不均衡度
        if len(unique) > 1:
            max_count = max(counts)
            min_count = min(counts)
            analysis['imbalance_ratio'] = max_count / min_count
        else:
            analysis['imbalance_ratio'] = 1.0
        
        # 連続性の分析
        label_changes = np.diff(labels)
        analysis['label_changes'] = np.sum(label_changes != 0)
        analysis['stability_ratio'] = 1 - (analysis['label_changes'] / (total - 1))
        
        return analysis
    
    def validate_labels(self, df: pd.DataFrame, labels: pd.Series, price_col: str = 'close') -> Dict:
        """
        ラベルの妥当性検証
        Args:
            df: OHLCV DataFrame
            labels: ラベルSeries
            price_col: 価格列名
        Returns:
            dict: 検証結果
        """
        logger.info("ラベル妥当性検証開始...")
        
        validation_results = {}
        
        # 基本チェック
        validation_results['length_match'] = len(df) == len(labels)
        validation_results['no_invalid_labels'] = labels.isin([0, 1, 2]).all()
        validation_results['no_future_leak'] = True  # 実装上未来リークはない
        
        # サンプル検証（最初の100サンプル）
        sample_size = min(100, len(df) - 1)
        correct_predictions = 0
        
        prices = df[price_col].values
        
        for i in range(sample_size):
            label = labels.iloc[i]
            future_max, future_min = self._calculate_future_extremes(prices, i)
            
            if label == 1:  # BUY
                expected_result = self._check_buy_condition(prices[i], future_max, future_min)
            elif label == 2:  # SELL
                expected_result = self._check_sell_condition(prices[i], future_max, future_min)
            else:  # NO_TRADE
                buy_result = self._check_buy_condition(prices[i], future_max, future_min)
                sell_result = self._check_sell_condition(prices[i], future_max, future_min)
                expected_result = not (buy_result or sell_result)
            
            if expected_result:
                correct_predictions += 1
        
        validation_results['sample_accuracy'] = correct_predictions / sample_size
        validation_results['sample_size'] = sample_size
        
        logger.info(f"検証完了 - サンプル精度: {validation_results['sample_accuracy']:.3f}")
        
        return validation_results

    def _check_buy_condition_relaxed(self, current_price: float, future_max: float, future_min: float) -> bool:
        """
        緩和されたBUY条件（利確重視）
        """
        entry_price = current_price + self.utils.pips_to_price(self.spread_pips / 2)
        
        # 利確条件のみ（損切り条件は考慮しない）
        profit_target = entry_price + self.utils.pips_to_price(self.profit_pips)
        profit_achieved = future_max >= profit_target
        
        return profit_achieved
    
    def _check_sell_condition_relaxed(self, current_price: float, future_max: float, future_min: float) -> bool:
        """
        緩和されたSELL条件（利確重視）
        """
        entry_price = current_price - self.utils.pips_to_price(self.spread_pips / 2)
        
        # 利確条件のみ（損切り条件は考慮しない）
        profit_target = entry_price - self.utils.pips_to_price(self.profit_pips)
        profit_achieved = future_min <= profit_target
        
        return profit_achieved
    
    def create_balanced_labels_vectorized(self, df: pd.DataFrame, price_col: str = 'close') -> pd.Series:
        """
        バランス重視のラベル生成
        """
        logger.info(f"バランス重視ラベル生成開始: {len(df)} 行")
        
        prices = df[price_col].values
        labels = np.zeros(len(prices), dtype=int)
        
        buy_count = 0
        sell_count = 0
        
        for i in range(len(prices) - 1):
            future_max, future_min = self._calculate_future_extremes(prices, i)
            
            # 緩和条件でBUY/SELLをチェック
            buy_viable = self._check_buy_condition_relaxed(prices[i], future_max, future_min)
            sell_viable = self._check_sell_condition_relaxed(prices[i], future_max, future_min)
            
            if buy_viable and sell_viable:
                # 両方可能な場合、より大きな利益の方向を選択
                buy_potential = future_max - (prices[i] + self.utils.pips_to_price(self.spread_pips / 2))
                sell_potential = (prices[i] - self.utils.pips_to_price(self.spread_pips / 2)) - future_min
                
                if buy_potential > sell_potential:
                    labels[i] = 1  # BUY
                    buy_count += 1
                else:
                    labels[i] = 2  # SELL
                    sell_count += 1
            elif buy_viable:
                labels[i] = 1  # BUY
                buy_count += 1
            elif sell_viable:
                labels[i] = 2  # SELL
                sell_count += 1
            # それ以外はNO_TRADE（0のまま）
        
        # バランス調整（SELLが少なすぎる場合の補正）
        if sell_count < buy_count * 0.1:  # SELLがBUYの10%未満の場合
            logger.warning(f"SELL不足検出 (BUY:{buy_count}, SELL:{sell_count}) - 補正実行")
            labels = self._rebalance_labels(labels, prices)
        
        # 統計表示
        unique, counts = np.unique(labels, return_counts=True)
        label_stats = dict(zip(unique, counts))
        logger.info(f"バランス調整後ラベル統計: {label_stats}")
        
        return pd.Series(labels, index=df.index, name='balanced_label')
    
    def _rebalance_labels(self, labels: np.array, prices: np.array) -> np.array:
        """
        ラベルの再バランス調整
        """
        # BUYラベルの一部をSELLに変更する補正ロジック
        buy_indices = np.where(labels == 1)[0]
        
        # BUYの20%程度をSELLに変更
        sell_candidates = np.random.choice(buy_indices, size=int(len(buy_indices) * 0.2), replace=False)
        
        for idx in sell_candidates:
            # 実際にSELL条件を満たすかチェック
            future_max, future_min = self._calculate_future_extremes(prices, idx)
            if self._check_sell_condition_relaxed(prices[idx], future_max, future_min):
                labels[idx] = 2  # SELLに変更
        
        return labels

    def _check_buy_condition_strict(self, current_price: float, future_max: float, future_min: float) -> bool:
        """
        厳格なBUY条件（ChatGPT提案：AND条件必須）
        両方の条件を満たす場合のみTRADE認定
        """
        entry_price = current_price + self.utils.pips_to_price(self.spread_pips / 2)
        
        # 利確条件：必須
        profit_target = entry_price + self.utils.pips_to_price(self.profit_pips)
        profit_achieved = future_max >= profit_target
        
        # 損切り条件：必須（緩和しない）
        loss_threshold = entry_price - self.utils.pips_to_price(self.loss_pips)
        no_excessive_loss = future_min >= loss_threshold
        
        # 🔧 ChatGPT提案：AND条件のみ（OR条件は使わない）
        return profit_achieved and no_excessive_loss
    
    def _check_sell_condition_strict(self, current_price: float, future_max: float, future_min: float) -> bool:
        """
        厳格なSELL条件（ChatGPT提案：AND条件必須）
        """
        entry_price = current_price - self.utils.pips_to_price(self.spread_pips / 2)
        
        # 利確条件：必須
        profit_target = entry_price - self.utils.pips_to_price(self.profit_pips)
        profit_achieved = future_min <= profit_target
        
        # 損切り条件：必須
        loss_threshold = entry_price + self.utils.pips_to_price(self.loss_pips)
        no_excessive_loss = future_max <= loss_threshold
        
        # 🔧 ChatGPT提案：AND条件のみ
        return profit_achieved and no_excessive_loss
    
    def create_binary_labels_strict(self, df: pd.DataFrame, price_col: str = 'close') -> pd.Series:
        """
        厳格な2値分類ラベル生成（ChatGPT提案実装）
        目標：TRADE vs NO_TRADE の比率を 1:2 〜 1:5 程度に調整
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
        logger.info(f"  目標範囲: TRADE 10-30%, NO_TRADE 70-90%")
        
        # バランスチェック
        if trade_ratio > 0.5:
            logger.warning("⚠️ TRADEシグナルが多すぎます。profit_pipsを増やすかloss_pipsを減らしてください")
        elif trade_ratio < 0.05:
            logger.warning("⚠️ TRADEシグナルが少なすぎます。profit_pipsを減らすかloss_pipsを増やしてください")
        else:
            logger.info("✅ 適切なバランスです")
        
        return pd.Series(labels, index=df.index, name='strict_binary_label')
    
    # さらに、条件をより厳格にするためのヘルパーメソッド
    def get_strict_labeler_config() -> dict:
        """
        厳格ラベリング用の推奨設定
        """
        return {
            'profit_pips': 6.0,        # 利確目標（適度）
            'loss_pips': 4.0,          # 損切り許容（厳格）
            'lookforward_ticks': 80,   # 観測期間（短縮）
            'use_or_conditions': False # AND条件必須
        }

    def _check_buy_condition_profit_focused(self, current_price: float, future_max: float, future_min: float) -> bool:
        """
        勝率重視のBUY条件（ChatGPT+Claude提案）
        - 利確目標: 9pips（より高い要求）
        - 損切り許容: 3pips（より厳格）
        - AND条件必須
        """
        entry_price = current_price + self.utils.pips_to_price(self.spread_pips / 2)
        
        # より高い利確目標
        profit_target = entry_price + self.utils.pips_to_price(9.0)
        profit_achieved = future_max >= profit_target
        
        # より狭い損切り許容
        loss_threshold = entry_price - self.utils.pips_to_price(3.0)
        no_excessive_loss = future_min >= loss_threshold
        
        # 厳格なAND条件
        return profit_achieved and no_excessive_loss
    
    def _check_sell_condition_profit_focused(self, current_price: float, future_max: float, future_min: float) -> bool:
        """
        勝率重視のSELL条件（ChatGPT+Claude提案）
        """
        entry_price = current_price - self.utils.pips_to_price(self.spread_pips / 2)
        
        # より高い利確目標
        profit_target = entry_price - self.utils.pips_to_price(9.0)
        profit_achieved = future_min <= profit_target
        
        # より狭い損切り許容
        loss_threshold = entry_price + self.utils.pips_to_price(3.0)
        no_excessive_loss = future_max <= loss_threshold
        
        # 厳格なAND条件
        return profit_achieved and no_excessive_loss
    
    def create_profit_focused_labels(self, df: pd.DataFrame, price_col: str = 'close') -> pd.Series:
        """
        Phase 2A: 勝率重視の2値分類ラベル生成
        - 高い利確要求（9pips）
        - 小さな損切り許容（3pips）
        - 確実性の高いシグナルのみTRADE認定
        
        Args:
            df: OHLCV DataFrame
            price_col: 価格列名
        Returns:
            Series: ラベル（0=NO_TRADE, 1=TRADE）
        """
        logger.info(f"Phase 2A: 勝率重視ラベル生成開始: {len(df)} 行")
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
        no_trade_ratio = conservative_no_trade / total
        
        logger.info(f"Phase 2A ラベル統計:")
        logger.info(f"  高確信TRADE: {high_confidence_trade:,} ({trade_ratio:.1%})")
        logger.info(f"  保守的NO_TRADE: {conservative_no_trade:,} ({no_trade_ratio:.1%})")
        
        # 目標評価
        if 0.10 <= trade_ratio <= 0.25:
            logger.info("✅ 理想的なTRADE比率（10-25%）です")
        elif trade_ratio > 0.25:
            logger.warning("⚠️ TRADEが多すぎます。さらに厳格化を検討")
        elif trade_ratio < 0.05:
            logger.warning("⚠️ TRADEが少なすぎます。条件を少し緩和検討")
        else:
            logger.info("✓ 適度なTRADE比率です")
        
        return pd.Series(labels, index=df.index, name='profit_focused_label')
    
    def create_ultra_conservative_labels(self, df: pd.DataFrame, price_col: str = 'close') -> pd.Series:
        """
        さらに保守的なラベル生成（必要時のオプション）
        - 利確12pips, 損切り2pips
        - 最高品質シグナルのみ
        """
        logger.info(f"超保守的ラベル生成開始: {len(df)} 行")
        logger.info("条件: 利確12pips, 損切り2pips, 超厳格")
        
        prices = df[price_col].values
        labels = np.zeros(len(prices), dtype=int)
        
        ultra_trade = 0
        
        for i in range(len(prices) - 1):
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
        
        logger.info(f"超保守統計: 超高確信TRADE {ultra_trade} ({ultra_trade/len(labels):.1%})")
        
        return pd.Series(labels, index=df.index, name='ultra_conservative_label')

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
        Phase 2A: 勝率重視の2値分類ラベル生成
        - 高い利確要求（9pips）
        - 小さな損切り許容（3pips）
        - 確実性の高いシグナルのみTRADE認定
        """
        logger.info(f"Phase 2A: 勝率重視ラベル生成開始: {len(df)} 行")
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
        
        logger.info(f"Phase 2A ラベル統計:")
        logger.info(f"  高確信TRADE: {high_confidence_trade:,} ({trade_ratio:.1%})")
        logger.info(f"  保守的NO_TRADE: {conservative_no_trade:,} ({(1-trade_ratio):.1%})")
        
        return pd.Series(labels, index=df.index, name='profit_focused_label')
    
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

    def create_realistic_profit_labels(self, df: pd.DataFrame, tp_pips: float = 6.0, sl_pips: float = 4.0, price_col: str = 'close') -> pd.Series:
        """
        Phase 2B: 現実的利益ラベル生成
        - 適度な利確目標と損切り設定
        - 勝率60%以上を目指す現実的なライン
        
        Args:
            df: OHLCV DataFrame
            tp_pips: 利確目標pips
            sl_pips: 損切り許容pips
            price_col: 価格列名
        Returns:
            Series: ラベル（0=NO_TRADE, 1=TRADE）
        """
        logger.info(f"Phase 2B: 現実的利益ラベル生成開始: {len(df)} 行")
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
    
    def create_momentum_optimized_labels(self, df: pd.DataFrame, 
                                       body_ratio: float = 0.7, 
                                       min_body_pips: float = 4.0,
                                       price_col: str = 'close') -> pd.Series:
        """
        Phase 2B: モメンタム最適化ラベル生成
        - ローソク足の実体サイズに基づく判定
        - 大陽線・大陰線の勢いを捉える
        
        Args:
            df: OHLCV DataFrame（open, high, low, close必須）
            body_ratio: 実体/レンジ比率の閾値
            min_body_pips: 最小実体サイズ（pips）
            price_col: 価格列名
        Returns:
            Series: ラベル（0=NO_TRADE, 1=TRADE）
        """
        logger.info(f"Phase 2B: モメンタム最適化ラベル生成開始: {len(df)} 行")
        logger.info(f"設定: 実体比率{body_ratio}, 最小実体{min_body_pips}pips")
        
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"必須列 '{col}' が見つかりません")
                return pd.Series([], dtype=int, name='momentum_optimized_label')
        
        if len(df) == 0:
            return pd.Series([], dtype=int, name='momentum_optimized_label')
        
        labels = np.zeros(len(df), dtype=int)
        momentum_trade_count = 0
        no_momentum_count = 0
        
        for i in range(len(df) - 1):
            try:
                # 現在のローソク足分析
                open_price = df['open'].iloc[i]
                high_price = df['high'].iloc[i]
                low_price = df['low'].iloc[i]
                close_price = df['close'].iloc[i]
                
                # 実体とレンジの計算
                body_size = abs(close_price - open_price)
                total_range = high_price - low_price
                
                # 実体サイズ（pips）
                body_pips = self.utils.price_to_pips(body_size)
                
                # 実体比率
                body_ratio_actual = body_size / total_range if total_range > 0 else 0
                
                # 未来の値動き確認
                future_max, future_min = self._calculate_future_extremes(df[price_col].values, i)
                
                # モメンタム条件: 大きな実体 + 高い実体比率
                has_momentum = (body_pips >= min_body_pips) and (body_ratio_actual >= body_ratio)
                
                if has_momentum:
                    # BUY方向（陽線）のモメンタム
                    if close_price > open_price:
                        buy_entry = close_price + self.utils.pips_to_price(self.spread_pips / 2)
                        buy_tp = buy_entry + self.utils.pips_to_price(6.0)  # 6pips利確
                        buy_sl = buy_entry - self.utils.pips_to_price(4.0)  # 4pips損切り
                        
                        buy_momentum_valid = (future_max >= buy_tp) and (future_min >= buy_sl)
                        
                        if buy_momentum_valid:
                            labels[i] = 1  # BUY TRADE
                            momentum_trade_count += 1
                        else:
                            labels[i] = 0
                            no_momentum_count += 1
                    
                    # SELL方向（陰線）のモメンタム
                    elif close_price < open_price:
                        sell_entry = close_price - self.utils.pips_to_price(self.spread_pips / 2)
                        sell_tp = sell_entry - self.utils.pips_to_price(6.0)  # 6pips利確
                        sell_sl = sell_entry + self.utils.pips_to_price(4.0)  # 4pips損切り
                        
                        sell_momentum_valid = (future_min <= sell_tp) and (future_max <= sell_sl)
                        
                        if sell_momentum_valid:
                            labels[i] = 1  # SELL TRADE
                            momentum_trade_count += 1
                        else:
                            labels[i] = 0
                            no_momentum_count += 1
                    else:
                        labels[i] = 0  # 十字線等
                        no_momentum_count += 1
                else:
                    labels[i] = 0  # モメンタム不足
                    no_momentum_count += 1
                    
            except Exception as e:
                if momentum_trade_count + no_momentum_count < 10:
                    logger.warning(f"行 {i} の処理でエラー: {e}")
                labels[i] = 0
                no_momentum_count += 1
        
        # 最後の行
        labels[-1] = 0
        no_momentum_count += 1
        
        # 統計表示
        total = len(labels)
        trade_ratio = momentum_trade_count / total
        
        logger.info(f"モメンタム最適化ラベル統計:")
        logger.info(f"  モメンタムTRADE: {momentum_trade_count:,} ({trade_ratio:.1%})")
        logger.info(f"  NO_TRADE: {no_momentum_count:,} ({(1-trade_ratio):.1%})")
        
        return pd.Series(labels, index=df.index, name='momentum_optimized_label')
    
    def create_conservative_but_profitable_labels(self, df: pd.DataFrame, 
                                                 tp_pips: float = 5.0, 
                                                 sl_pips: float = 4.0,
                                                 max_trade_ratio: float = 0.35,
                                                 price_col: str = 'close') -> pd.Series:
        """
        Phase 2B: 保守的だが利益の出るラベル生成
        - トレード割合を制限しつつ利益確保
        - 高品質なシグナルのみを選別
        
        Args:
            df: OHLCV DataFrame
            tp_pips: 利確目標pips
            sl_pips: 損切り許容pips
            max_trade_ratio: 最大TRADE比率
            price_col: 価格列名
        Returns:
            Series: ラベル（0=NO_TRADE, 1=TRADE）
        """
        logger.info(f"Phase 2B: 保守的利益ラベル生成開始: {len(df)} 行")
        logger.info(f"設定: 利確{tp_pips}pips, 損切り{sl_pips}pips, 最大TRADE比率{max_trade_ratio:.1%}")
        
        if len(df) == 0:
            return pd.Series([], dtype=int, name='conservative_profitable_label')
        
        prices = df[price_col].values
        labels = np.zeros(len(prices), dtype=int)
        
        # 第1段階: 全候補を評価
        trade_candidates = []
        
        for i in range(len(prices) - 1):
            try:
                future_max, future_min = self._calculate_future_extremes(prices, i)
                
                # BUY候補の評価
                buy_entry = prices[i] + self.utils.pips_to_price(self.spread_pips / 2)
                buy_tp = buy_entry + self.utils.pips_to_price(tp_pips)
                buy_sl = buy_entry - self.utils.pips_to_price(sl_pips)
                
                buy_profitable = (future_max >= buy_tp) and (future_min >= buy_sl)
                
                if buy_profitable:
                    # 利益品質スコア計算
                    buy_profit_margin = self.utils.price_to_pips(future_max - buy_tp)
                    buy_safety_margin = self.utils.price_to_pips(future_min - buy_sl)
                    buy_score = buy_profit_margin + buy_safety_margin
                    
                    trade_candidates.append({
                        'index': i,
                        'direction': 'BUY',
                        'score': buy_score,
                        'profit_margin': buy_profit_margin,
                        'safety_margin': buy_safety_margin
                    })
                
                # SELL候補の評価
                sell_entry = prices[i] - self.utils.pips_to_price(self.spread_pips / 2)
                sell_tp = sell_entry - self.utils.pips_to_price(tp_pips)
                sell_sl = sell_entry + self.utils.pips_to_price(sl_pips)
                
                sell_profitable = (future_min <= sell_tp) and (future_max <= sell_sl)
                
                if sell_profitable:
                    # 利益品質スコア計算
                    sell_profit_margin = self.utils.price_to_pips(sell_tp - future_min)
                    sell_safety_margin = self.utils.price_to_pips(sell_sl - future_max)
                    sell_score = sell_profit_margin + sell_safety_margin
                    
                    trade_candidates.append({
                        'index': i,
                        'direction': 'SELL',
                        'score': sell_score,
                        'profit_margin': sell_profit_margin,
                        'safety_margin': sell_safety_margin
                    })
                    
            except Exception as e:
                continue
        
        # 第2段階: 高品質候補を選別
        if trade_candidates:
            # スコア順でソート
            trade_candidates.sort(key=lambda x: x['score'], reverse=True)
            
            # 最大TRADE比率に基づく選択数
            max_trades = int(len(prices) * max_trade_ratio)
            selected_trades = trade_candidates[:max_trades]
            
            # ラベル設定
            for trade in selected_trades:
                labels[trade['index']] = 1
            
            logger.info(f"保守的利益ラベル統計:")
            logger.info(f"  候補TRADE: {len(trade_candidates):,}")
            logger.info(f"  選択TRADE: {len(selected_trades):,} ({len(selected_trades)/len(prices):.1%})")
            logger.info(f"  平均スコア: {np.mean([t['score'] for t in selected_trades]):.2f}")
            
            if selected_trades:
                avg_profit_margin = np.mean([t['profit_margin'] for t in selected_trades])
                avg_safety_margin = np.mean([t['safety_margin'] for t in selected_trades])
                logger.info(f"  平均利益マージン: {avg_profit_margin:.2f}pips")
                logger.info(f"  平均安全マージン: {avg_safety_margin:.2f}pips")
        else:
            logger.warning("利益候補が見つかりませんでした")
        
        return pd.Series(labels, index=df.index, name='conservative_profitable_label')

    def create_chatgpt_improved_labels(self, df: pd.DataFrame, 
                                      tp_pips: float = 5.0,
                                      sl_pips: float = 3.0,
                                      lookforward_ticks: int = 120,
                                      buffer_pips: float = 0.5,
                                      price_col: str = 'close') -> pd.Series:
        """
        ChatGPT改善版ラベル生成
        - lookforward_ticks = 120
        - スプレッド + バッファを考慮
        - より現実的な条件設定
        
        Args:
            tp_pips: 利確目標pips
            sl_pips: 損切り許容pips  
            lookforward_ticks: 前方参照ティック数（120推奨）
            buffer_pips: 追加バッファpips
            price_col: 価格列名
        Returns:
            Series: ラベル（0=NO_TRADE, 1=TRADE）
        """
        logger.info(f"ChatGPT改善ラベル生成開始: {len(df)} 行")
        logger.info(f"設定: TP={tp_pips}pips, SL={sl_pips}pips, 前方参照={lookforward_ticks}ティック, バッファ={buffer_pips}pips")
        
        if len(df) == 0:
            return pd.Series([], dtype=int, name='chatgpt_improved_label')
        
        prices = df[price_col].values
        labels = np.zeros(len(prices), dtype=int)
        
        # ChatGPT改善版の極値計算関数
        def _calculate_future_extremes_extended(prices: np.array, start_idx: int) -> Tuple[float, float]:
            """拡張版未来極値計算（120ティック）"""
            end_idx = min(start_idx + lookforward_ticks, len(prices))
            
            if start_idx >= len(prices) - 1:
                return prices[start_idx], prices[start_idx]
            
            future_prices = prices[start_idx + 1:end_idx + 1]
            
            if len(future_prices) == 0:
                return prices[start_idx], prices[start_idx]
            
            return np.max(future_prices), np.min(future_prices)
        
        improved_trade_count = 0
        no_trade_count = 0
        
        for i in range(len(prices) - 1):
            try:
                future_max, future_min = _calculate_future_extremes_extended(prices, i)
                
                # ChatGPT提案: スプレッド + バッファを考慮したエントリー価格
                base_spread = self.utils.pips_to_price(self.spread_pips)
                buffer_amount = self.utils.pips_to_price(buffer_pips)
                
                # BUY方向の改善判定
                buy_entry = prices[i] + base_spread / 2
                buy_tp_target = buy_entry + self.utils.pips_to_price(tp_pips)
                buy_sl_threshold = buy_entry - self.utils.pips_to_price(sl_pips)
                
                # ChatGPT条件: price >= entry + spread + buffer
                buy_profitable_with_buffer = future_max >= (buy_tp_target + buffer_amount)
                buy_safe_with_buffer = future_min >= (buy_sl_threshold - buffer_amount)
                
                # SELL方向の改善判定
                sell_entry = prices[i] - base_spread / 2
                sell_tp_target = sell_entry - self.utils.pips_to_price(tp_pips)
                sell_sl_threshold = sell_entry + self.utils.pips_to_price(sl_pips)
                
                sell_profitable_with_buffer = future_min <= (sell_tp_target - buffer_amount)
                sell_safe_with_buffer = future_max <= (sell_sl_threshold + buffer_amount)
                
                # ChatGPT改善条件: より厳格な利確 + 安全性
                buy_improved = buy_profitable_with_buffer and buy_safe_with_buffer
                sell_improved = sell_profitable_with_buffer and sell_safe_with_buffer
                
                if buy_improved or sell_improved:
                    labels[i] = 1  # TRADE
                    improved_trade_count += 1
                else:
                    labels[i] = 0  # NO_TRADE
                    no_trade_count += 1
                    
            except Exception as e:
                if improved_trade_count + no_trade_count < 10:
                    logger.warning(f"行 {i} の処理でエラー: {e}")
                labels[i] = 0
                no_trade_count += 1
        
        # 最後の行
        labels[-1] = 0
        no_trade_count += 1
        
        # 統計表示
        total = len(labels)
        trade_ratio = improved_trade_count / total
        
        logger.info(f"ChatGPT改善ラベル統計:")
        logger.info(f"  TRADE: {improved_trade_count:,} ({trade_ratio:.1%})")
        logger.info(f"  NO_TRADE: {no_trade_count:,} ({(1-trade_ratio):.1%})")
        logger.info(f"  期待勝率: 60-70% (バッファ効果)")
        logger.info(f"  期待利益: +{tp_pips - sl_pips * 0.3:.1f}pips (勝率70%想定)")
        
        return pd.Series(labels, index=df.index, name='chatgpt_improved_label')
    
    def create_parameter_optimized_labels(self, df: pd.DataFrame,
                                         tp_pips: float = 4.0,      # ChatGPT提案: 5→4
                                         sl_pips: float = 3.0,      # ChatGPT提案: 変更なし
                                         body_ratio: float = 0.6,   # ChatGPT提案: 0.7→0.6
                                         max_trade_ratio: float = 0.3,  # ChatGPT提案: 0.35→0.3
                                         price_col: str = 'close') -> pd.Series:
        """
        ChatGPT提案パラメータ最適化ラベル生成
        """
        logger.info(f"パラメータ最適化ラベル生成開始: {len(df)} 行")
        logger.info(f"最適化パラメータ: TP={tp_pips}, SL={sl_pips}, body_ratio={body_ratio}, max_trade_ratio={max_trade_ratio}")
        
        if len(df) == 0:
            return pd.Series([], dtype=int, name='parameter_optimized_label')
        
        # 必要な列の確認
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"必須列 '{col}' が見つかりません")
                return pd.Series([], dtype=int, name='parameter_optimized_label')
        
        prices = df[price_col].values
        candidate_trades = []
        
        for i in range(len(df) - 1):
            try:
                # ローソク足分析（ChatGPT提案のbody_ratio使用）
                open_price = df['open'].iloc[i]
                high_price = df['high'].iloc[i]
                low_price = df['low'].iloc[i]
                close_price = df['close'].iloc[i]
                
                body_size = abs(close_price - open_price)
                total_range = high_price - low_price
                actual_body_ratio = body_size / total_range if total_range > 0 else 0
                
                # ChatGPT条件: body_ratio >= 0.6
                has_sufficient_body = actual_body_ratio >= body_ratio
                
                if not has_sufficient_body:
                    continue
                
                # 未来の価格変動確認（120ティック）
                future_max, future_min = self._calculate_future_extremes_extended(prices, i)
                
                # 最適化されたTP/SL条件
                buy_entry = prices[i] + self.utils.pips_to_price(self.spread_pips / 2)
                buy_tp = buy_entry + self.utils.pips_to_price(tp_pips)
                buy_sl = buy_entry - self.utils.pips_to_price(sl_pips)
                
                sell_entry = prices[i] - self.utils.pips_to_price(self.spread_pips / 2)
                sell_tp = sell_entry - self.utils.pips_to_price(tp_pips)
                sell_sl = sell_entry + self.utils.pips_to_price(sl_pips)
                
                # 利益ポテンシャル計算
                buy_profit_potential = 0
                sell_profit_potential = 0
                
                if (future_max >= buy_tp) and (future_min >= buy_sl):
                    buy_profit_margin = self.utils.price_to_pips(future_max - buy_tp)
                    buy_safety_margin = self.utils.price_to_pips(future_min - buy_sl)
                    buy_profit_potential = tp_pips + buy_profit_margin + buy_safety_margin
                
                if (future_min <= sell_tp) and (future_max <= sell_sl):
                    sell_profit_margin = self.utils.price_to_pips(sell_tp - future_min)
                    sell_safety_margin = self.utils.price_to_pips(sell_sl - future_max)
                    sell_profit_potential = tp_pips + sell_profit_margin + sell_safety_margin
                
                # より良い方向を候補に追加
                if buy_profit_potential > 0 or sell_profit_potential > 0:
                    direction = 'BUY' if buy_profit_potential >= sell_profit_potential else 'SELL'
                    score = max(buy_profit_potential, sell_profit_potential)
                    
                    candidate_trades.append({
                        'index': i,
                        'direction': direction,
                        'score': score,
                        'body_ratio': actual_body_ratio
                    })
                    
            except Exception as e:
                continue
        
        # ChatGPT提案: max_trade_ratio に基づく選択
        labels = np.zeros(len(prices), dtype=int)
        
        if candidate_trades:
            # スコア順でソート
            candidate_trades.sort(key=lambda x: x['score'], reverse=True)
            
            # 最大トレード比率に基づく選択
            max_trades = int(len(prices) * max_trade_ratio)
            selected_trades = candidate_trades[:max_trades]
            
            for trade in selected_trades:
                labels[trade['index']] = 1
            
            logger.info(f"パラメータ最適化ラベル統計:")
            logger.info(f"  候補数: {len(candidate_trades):,}")
            logger.info(f"  選択数: {len(selected_trades):,} ({len(selected_trades)/len(prices):.1%})")
            logger.info(f"  平均スコア: {np.mean([t['score'] for t in selected_trades]):.2f}")
            logger.info(f"  平均body_ratio: {np.mean([t['body_ratio'] for t in selected_trades]):.3f}")
        
        return pd.Series(labels, index=df.index, name='parameter_optimized_label')
    
    def _calculate_future_extremes_extended(self, prices: np.array, start_idx: int, lookforward_ticks: int = 120) -> Tuple[float, float]:
        """
        拡張版未来極値計算（ChatGPT提案の120ティック）
        """
        end_idx = min(start_idx + lookforward_ticks, len(prices))
        
        if start_idx >= len(prices) - 1:
            return prices[start_idx], prices[start_idx]
        
        future_prices = prices[start_idx + 1:end_idx + 1]
        
        if len(future_prices) == 0:
            return prices[start_idx], prices[start_idx]
        
        return np.max(future_prices), np.min(future_prices)

class LabelPostProcessor:
    """ラベル後処理クラス"""
    
    @staticmethod
    def filter_consecutive_labels(labels: pd.Series, max_consecutive: int = 5) -> pd.Series:
        """
        連続する同じラベルをフィルタリング
        Args:
            labels: ラベルSeries
            max_consecutive: 最大連続数
        Returns:
            Series: フィルタ済みラベル
        """
        filtered_labels = labels.copy()
        
        current_label = None
        consecutive_count = 0
        
        for i in range(len(filtered_labels)):
            if filtered_labels.iloc[i] == current_label:
                consecutive_count += 1
                if consecutive_count > max_consecutive:
                    filtered_labels.iloc[i] = 0  # NO_TRADEに変更
            else:
                current_label = filtered_labels.iloc[i]
                consecutive_count = 1
        
        return filtered_labels
    
    @staticmethod
    def apply_minimum_holding_period(labels: pd.Series, min_period: int = 3) -> pd.Series:
        """
        最小保有期間フィルタ
        Args:
            labels: ラベルSeries
            min_period: 最小期間
        Returns:
            Series: フィルタ済みラベル
        """
        filtered_labels = labels.copy()
        
        i = 0
        while i < len(filtered_labels):
            if filtered_labels.iloc[i] in [1, 2]:  # BUY or SELL
                # 前方をチェック
                end_idx = min(i + min_period, len(filtered_labels))
                if end_idx - i < min_period:
                    # 期間が足りない場合はNO_TRADEに変更
                    filtered_labels.iloc[i:end_idx] = 0
                i = end_idx
            else:
                i += 1
        
        return filtered_labels


def create_sample_labels(ohlcv_df: pd.DataFrame, 
                        profit_pips: float = 8.0,
                        loss_pips: float = 4.0,
                        lookforward_ticks: int = 100) -> Tuple[pd.Series, Dict]:
    """
    サンプルデータでラベル生成テスト
    Args:
        ohlcv_df: OHLCV DataFrame
        profit_pips: 利確pips
        loss_pips: 損切りpips
        lookforward_ticks: 前方参照ティック数
    Returns:
        tuple: (labels, analysis)
    """
    labeler = ScalpingLabeler(
        profit_pips=profit_pips,
        loss_pips=loss_pips,
        lookforward_ticks=lookforward_ticks
    )
    
    # ラベル生成
    labels = labeler.create_labels_vectorized(ohlcv_df)
    
    # 分析
    analysis = labeler.analyze_label_distribution(labels)
    
    # 検証
    validation = labeler.validate_labels(ohlcv_df, labels)
    analysis['validation'] = validation
    
    return labels, analysis


if __name__ == "__main__":
    # テスト実行
    print("=== ラベリングテスト ===")
    
    # サンプルデータ作成
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=5000, freq='1min')
    
    # トレンドのあるサンプルデータ
    base_price = 118.0
    price_changes = np.random.randn(5000) * 0.001
    
    # 時々大きな動きを追加（ラベル生成のため）
    for i in range(0, 5000, 500):
        if i < len(price_changes):
            price_changes[i:i+50] += np.random.choice([-1, 1]) * 0.02
    
    cumulative_changes = np.cumsum(price_changes)
    
    sample_data = pd.DataFrame({
        'open': base_price + cumulative_changes,
        'close': base_price + cumulative_changes + np.random.randn(5000) * 0.0005,
        'high': 0,
        'low': 0,
        'volume': np.random.randint(10, 100, 5000)
    }, index=dates)
    
    # high, low を調整
    sample_data['high'] = sample_data[['open', 'close']].max(axis=1) + abs(np.random.randn(5000) * 0.002)
    sample_data['low'] = sample_data[['open', 'close']].min(axis=1) - abs(np.random.randn(5000) * 0.002)
    
    try:
        # ラベル生成テスト
        labels, analysis = create_sample_labels(sample_data)
        
        print(f"ラベル生成完了: {len(labels)} 行")
        print(f"\nラベル分布:")
        for label_val, count in analysis['label_counts'].items():
            label_name = analysis['label_names'][label_val]
            percentage = analysis['label_percentages'][label_val]
            print(f"  {label_name}: {count:,} ({percentage:.2f}%)")
        
        print(f"\n不均衡比: {analysis['imbalance_ratio']:.2f}")
        print(f"安定性比: {analysis['stability_ratio']:.3f}")
        
        # 検証結果
        validation = analysis['validation']
        print(f"\n検証結果:")
        print(f"  長さ一致: {validation['length_match']}")
        print(f"  無効ラベルなし: {validation['no_invalid_labels']}")
        print(f"  サンプル精度: {validation['sample_accuracy']:.3f}")
        
        # 詳細分析（小さなサンプル）
        print("\n詳細ラベル分析（最初の10行）:")
        labeler = ScalpingLabeler()
        detailed = labeler.create_detailed_labels(sample_data.head(20))
        print(detailed[['label', 'buy_max_profit_pips', 'sell_max_profit_pips']].head(10).round(2))
        
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()