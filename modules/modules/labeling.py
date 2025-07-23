import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import logging
from .utils import calculate_mid_price, apply_spread_correction, pips_to_price

logger = logging.getLogger(__name__)

class LabelEngine:
    """
    スキャルピング用ラベル生成エンジン
    スプレッド0.7pips固定で厳密な計算
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.trading_config = config['trading']
        self.labels_config = config['labels']
        
        # 重要なパラメータ
        self.spread_pips = self.trading_config['spread_pips']
        self.pip_value = self.trading_config['pip_value']
        self.tp_pips = self.trading_config['tp_pips']
        self.sl_pips = self.trading_config['sl_pips']
        self.future_window = self.trading_config['future_window']
        
        logger.info(f"ラベリング設定 - スプレッド: {self.spread_pips}pips, "
                   f"TP: {self.tp_pips}pips, SL: {self.sl_pips}pips, "
                   f"未来窓: {self.future_window}ティック")
    
    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        3クラス分類ラベルを生成
        BUY(0), SELL(1), NO_TRADE(2)
        """
        logger.info("ラベル生成開始")
        
        # MID価格を計算（スプレッド補正の基準）
        df['mid_price'] = calculate_mid_price(df['BID'].values, df['ASK'].values)
        
        # ラベル配列を初期化（デフォルト：NO_TRADE）
        labels = np.full(len(df), self.labels_config['no_trade_class'], dtype=int)
        
        # 各ティックについてラベルを判定
        for i in range(len(df) - self.future_window):
            if i % 10000 == 0:
                logger.info(f"ラベリング進捗: {i}/{len(df)-self.future_window}")
            
            current_price = df.iloc[i]['mid_price']
            
            # 未来価格データを取得
            future_prices = df.iloc[i+1:i+1+self.future_window]['mid_price'].values
            
            # BUYラベル判定
            if self._check_buy_condition(current_price, future_prices):
                labels[i] = self.labels_config['buy_class']
            # SELLラベル判定
            elif self._check_sell_condition(current_price, future_prices):
                labels[i] = self.labels_config['sell_class']
            # それ以外はNO_TRADE（既にデフォルト設定済み）
        
        df['label'] = labels
        
        # ラベル分布を確認
        label_counts = pd.Series(labels).value_counts()
        logger.info(f"ラベル分布: BUY={label_counts.get(0, 0)}, "
                   f"SELL={label_counts.get(1, 0)}, "
                   f"NO_TRADE={label_counts.get(2, 0)}")
        
        return df
    
    def _check_buy_condition(self, entry_price: float, future_prices: np.ndarray) -> bool:
        """
        BUY条件をチェック
        - スプレッド補正を考慮したエントリー価格から
        - 将来100ティック以内に +4.0pips以上の上昇があり
        - かつ -5.0pips以上の逆行がない
        """
        # BUY エントリー価格（スプレッド補正済み）
        buy_entry_price = apply_spread_correction(
            entry_price, self.spread_pips, self.pip_value, 'buy'
        )
        
        # 目標価格とストップ価格
        tp_price = buy_entry_price + pips_to_price(self.tp_pips, self.pip_value)
        sl_price = buy_entry_price - pips_to_price(self.sl_pips, self.pip_value)
        
        # 各ティックで条件をチェック
        for future_price in future_prices:
            # 利確条件達成
            if future_price >= tp_price:
                return True
            # 損切条件に達した場合は失敗
            if future_price <= sl_price:
                return False
        
        return False
    
    def _check_sell_condition(self, entry_price: float, future_prices: np.ndarray) -> bool:
        """
        SELL条件をチェック
        - スプレッド補正を考慮したエントリー価格から
        - 将来100ティック以内に -4.0pips以上の下落があり
        - かつ +5.0pips以上の逆行がない
        """
        # SELL エントリー価格（スプレッド補正済み）
        sell_entry_price = apply_spread_correction(
            entry_price, self.spread_pips, self.pip_value, 'sell'
        )
        
        # 目標価格とストップ価格
        tp_price = sell_entry_price - pips_to_price(self.tp_pips, self.pip_value)
        sl_price = sell_entry_price + pips_to_price(self.sl_pips, self.pip_value)
        
        # 各ティックで条件をチェック
        for future_price in future_prices:
            # 利確条件達成
            if future_price <= tp_price:
                return True
            # 損切条件に達した場合は失敗
            if future_price >= sl_price:
                return False
        
        return False
    
    def validate_labels(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        ラベルの妥当性を検証
        """
        labels = df['label'].values
        
        # 基本統計
        total_samples = len(labels)
        buy_count = np.sum(labels == self.labels_config['buy_class'])
        sell_count = np.sum(labels == self.labels_config['sell_class'])
        no_trade_count = np.sum(labels == self.labels_config['no_trade_class'])
        
        # 割合計算
        buy_ratio = buy_count / total_samples
        sell_ratio = sell_count / total_samples
        no_trade_ratio = no_trade_count / total_samples
        
        # バランス評価
        signal_ratio = (buy_count + sell_count) / total_samples
        
        validation_result = {
            'total_samples': total_samples,
            'buy_count': buy_count,
            'sell_count': sell_count,
            'no_trade_count': no_trade_count,
            'buy_ratio': buy_ratio,
            'sell_ratio': sell_ratio,
            'no_trade_ratio': no_trade_ratio,
            'signal_ratio': signal_ratio,
            'is_balanced': 0.3 < signal_ratio < 0.7,  # シグナル比率が適切か
            'buy_sell_balance': abs(buy_count - sell_count) / max(buy_count, sell_count, 1)
        }
        
        logger.info(f"ラベル検証結果:")
        logger.info(f"  総サンプル数: {total_samples:,}")
        logger.info(f"  BUY: {buy_count:,} ({buy_ratio:.1%})")
        logger.info(f"  SELL: {sell_count:,} ({sell_ratio:.1%})")
        logger.info(f"  NO_TRADE: {no_trade_count:,} ({no_trade_ratio:.1%})")
        logger.info(f"  シグナル比率: {signal_ratio:.1%}")
        logger.info(f"  バランス良好: {'Yes' if validation_result['is_balanced'] else 'No'}")
        
        return validation_result
    
    def analyze_label_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        ラベル品質の詳細分析
        """
        logger.info("ラベル品質分析開始")
        
        # 時間別分布分析
        df['hour'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME']).dt.hour
        hourly_distribution = df.groupby(['hour', 'label']).size().unstack(fill_value=0)
        
        # 価格変動との相関分析
        df['price_change_1'] = df['mid_price'].shift(-1) - df['mid_price']
        df['price_change_5'] = df['mid_price'].shift(-5) - df['mid_price']
        df['price_change_10'] = df['mid_price'].shift(-10) - df['mid_price']
        
        buy_samples = df[df['label'] == self.labels_config['buy_class']]
        sell_samples = df[df['label'] == self.labels_config['sell_class']]
        
        quality_metrics = {
            'hourly_distribution': hourly_distribution.to_dict(),
            'buy_avg_1tick_change': buy_samples['price_change_1'].mean(),
            'sell_avg_1tick_change': sell_samples['price_change_1'].mean(),
            'buy_avg_5tick_change': buy_samples['price_change_5'].mean(),
            'sell_avg_5tick_change': sell_samples['price_change_5'].mean(),
            'buy_avg_10tick_change': buy_samples['price_change_10'].mean(),
            'sell_avg_10tick_change': sell_samples['price_change_10'].mean(),
        }
        
        logger.info("ラベル品質分析完了")
        return quality_metrics
    
    def create_balanced_dataset(self, df: pd.DataFrame, balance_method: str = 'undersample') -> pd.DataFrame:
        """
        クラス不均衡を調整
        """
        logger.info(f"データセットバランス調整: {balance_method}")
        
        # 各クラスのサンプル数を確認
        buy_mask = df['label'] == self.labels_config['buy_class']
        sell_mask = df['label'] == self.labels_config['sell_class']
        no_trade_mask = df['label'] == self.labels_config['no_trade_class']
        
        buy_count = buy_mask.sum()
        sell_count = sell_mask.sum()
        no_trade_count = no_trade_mask.sum()
        
        if balance_method == 'undersample':
            # 最小クラスに合わせてアンダーサンプリング
            min_count = min(buy_count, sell_count)
            target_no_trade = min_count * 2  # NO_TRADEは少し多めに残す
            
            # 各クラスからランダムサンプリング
            buy_indices = df[buy_mask].sample(min_count, random_state=42).index
            sell_indices = df[sell_mask].sample(min_count, random_state=42).index
            no_trade_indices = df[no_trade_mask].sample(min(target_no_trade, no_trade_count), random_state=42).index
            
            # 結合（時系列順序を保持）
            selected_indices = sorted(list(buy_indices) + list(sell_indices) + list(no_trade_indices))
            balanced_df = df.loc[selected_indices].copy()
            
        elif balance_method == 'weighted':
            # クラス重みを計算（モデル学習時に使用）
            total_samples = len(df)
            class_weights = {
                self.labels_config['buy_class']: total_samples / (3 * buy_count),
                self.labels_config['sell_class']: total_samples / (3 * sell_count),
                self.labels_config['no_trade_class']: total_samples / (3 * no_trade_count)
            }
            
            df['class_weight'] = df['label'].map(class_weights)
            balanced_df = df.copy()
        
        else:
            balanced_df = df.copy()
        
        logger.info(f"バランス調整完了: {len(balanced_df):,} サンプル")
        return balanced_df