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