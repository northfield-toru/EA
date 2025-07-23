import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import logging
from .utils import calculate_mid_price, memory_usage_mb

logger = logging.getLogger(__name__)

class FeatureEngine:
    """
    ティックデータ用特徴量エンジニアリング
    未来リーク完全防止・メモリ効率重視
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.features_config = config['features']
        self.trading_config = config['trading']
        
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        テクニカル指標を計算（未来リーク防止・逐次計算）
        """
        logger.info("テクニカル指標の計算を開始")
        
        # MID価格を基準とした計算
        df['mid_price'] = calculate_mid_price(df['BID'].values, df['ASK'].values)
        df['spread'] = df['ASK'] - df['BID']
        df['spread_pips'] = df['spread'] / self.trading_config['pip_value']
        
        # ボリューム関連（ティック数ベース）
        df['tick_volume'] = 1  # 各ティックを1ボリュームとして扱う
        df['volume_ma'] = self._rolling_mean(df['tick_volume'], 20)
        df['volume_change'] = df['tick_volume'].pct_change()
        
        # 価格変動関連
        df['price_change'] = df['mid_price'].diff()
        df['price_change_pct'] = df['mid_price'].pct_change()
        df['price_volatility'] = self._rolling_std(df['mid_price'], 20)
        
        # 移動平均線
        for period in self.features_config['sma_periods']:
            df[f'sma_{period}'] = self._rolling_mean(df['mid_price'], period)
            df[f'price_vs_sma_{period}'] = df['mid_price'] - df[f'sma_{period}']
        
        for period in self.features_config['ema_periods']:
            df[f'ema_{period}'] = self._exponential_moving_average(df['mid_price'], period)
            df[f'price_vs_ema_{period}'] = df['mid_price'] - df[f'ema_{period}']
        
        # ボリンジャーバンド
        df = self._calculate_bollinger_bands(df)
        
        # MACD
        df = self._calculate_macd(df)
        
        # RSI
        df = self._calculate_rsi(df)
        
        # ATR
        df = self._calculate_atr(df)
        
        # CCI
        df = self._calculate_cci(df)
        
        # 追加指標
        df = self._calculate_momentum_indicators(df)
        
        logger.info(f"特徴量計算完了 - メモリ使用量: {memory_usage_mb():.1f}MB")
        
        return df
    
    def _rolling_mean(self, series: pd.Series, window: int) -> pd.Series:
        """メモリ効率的な移動平均"""
        return series.rolling(window=window, min_periods=1).mean()
    
    def _rolling_std(self, series: pd.Series, window: int) -> pd.Series:
        """メモリ効率的な移動標準偏差"""
        return series.rolling(window=window, min_periods=1).std()
    
    def _exponential_moving_average(self, series: pd.Series, period: int) -> pd.Series:
        """指数移動平均（EMA）"""
        return series.ewm(span=period, adjust=False).mean()
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """ボリンジャーバンド計算"""
        period = self.features_config['bollinger_bands']['period']
        std_dev = self.features_config['bollinger_bands']['std_dev']
        
        df['bb_middle'] = self._rolling_mean(df['mid_price'], period)
        bb_std = self._rolling_std(df['mid_price'], period)
        
        df['bb_upper'] = df['bb_middle'] + (bb_std * std_dev)
        df['bb_lower'] = df['bb_middle'] - (bb_std * std_dev)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['mid_price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """MACD計算"""
        fast = self.features_config['macd']['fast_period']
        slow = self.features_config['macd']['slow_period']
        signal = self.features_config['macd']['signal_period']
        
        ema_fast = self._exponential_moving_average(df['mid_price'], fast)
        ema_slow = self._exponential_moving_average(df['mid_price'], slow)
        
        df['macd_line'] = ema_fast - ema_slow
        df['macd_signal'] = self._exponential_moving_average(df['macd_line'], signal)
        df['macd_histogram'] = df['macd_line'] - df['macd_signal']
        
        return df
    
    def _calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSI計算"""
        period = self.features_config['rsi']['period']
        
        delta = df['mid_price'].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """ATR（Average True Range）計算"""
        period = self.features_config['atr']['period']
        
        # ティックデータの場合、HIGHとLOWがないのでBIDとASKを使用
        df['high'] = df[['BID', 'ASK']].max(axis=1)
        df['low'] = df[['BID', 'ASK']].min(axis=1)
        
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['mid_price'].shift(1))
        df['tr3'] = abs(df['low'] - df['mid_price'].shift(1))
        
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = self._exponential_moving_average(df['true_range'], period)
        
        # 不要なカラムを削除
        df.drop(['high', 'low', 'tr1', 'tr2', 'tr3', 'true_range'], axis=1, inplace=True)
        
        return df
    
    def _calculate_cci(self, df: pd.DataFrame) -> pd.DataFrame:
        """CCI（Commodity Channel Index）計算"""
        period = self.features_config['cci']['period']
        
        typical_price = df['mid_price']  # ティックデータではmid_priceを使用
        sma_tp = self._rolling_mean(typical_price, period)
        mad = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )
        
        df['cci'] = (typical_price - sma_tp) / (0.015 * mad)
        
        return df
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """モメンタム系指標"""
        # 価格モメンタム
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['mid_price'] / df['mid_price'].shift(period) - 1
        
        # Rate of Change (ROC)
        for period in [5, 10]:
            df[f'roc_{period}'] = ((df['mid_price'] - df['mid_price'].shift(period)) / 
                                  df['mid_price'].shift(period) * 100)
        
        # Williams %R
        high_14 = df['ASK'].rolling(window=14).max()
        low_14 = df['BID'].rolling(window=14).min()
        df['williams_r'] = ((high_14 - df['mid_price']) / (high_14 - low_14)) * -100
        
        return df
    
    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        シーケンスデータの作成（時系列モデル用）
        """
        sequence_length = self.features_config['sequence_length']
        
        # 特徴量カラムを特定
        feature_columns = [col for col in df.columns if col not in 
                          ['DATE', 'TIME', 'BID', 'ASK', 'datetime', 'label']]
        
        # NaNを前方埋めで処理
        df[feature_columns] = df[feature_columns].fillna(method='ffill')
        df[feature_columns] = df[feature_columns].fillna(0)  # 最初のNaNは0で埋める
        
        # 正規化（各特徴量を0-1スケールに）
        feature_data = df[feature_columns].values
        feature_data = self._normalize_features(feature_data)
        
        # シーケンス作成
        sequences = []
        timestamps = []
        labels = []
        
        for i in range(sequence_length, len(df)):
            seq = feature_data[i-sequence_length:i]
            sequences.append(seq)
            timestamps.append(df.iloc[i]['datetime'] if 'datetime' in df.columns else i)
            
            if 'label' in df.columns:
                labels.append(df.iloc[i]['label'])
        
        X = np.array(sequences)
        y = np.array(labels, dtype=np.int32) if labels else None  # 整数型を明示
        
        logger.info(f"シーケンス作成完了: {X.shape}, 特徴量数: {len(feature_columns)}")
        if y is not None:
            logger.info(f"ラベル形状: {y.shape}, ラベル型: {y.dtype}")
            logger.info(f"ラベル分布: {np.unique(y, return_counts=True)}")
        
        return X, y, feature_columns, timestamps
    
    def _normalize_features(self, data: np.ndarray) -> np.ndarray:
        """
        特徴量正規化（Min-Maxスケーリング）
        """
        # 各特徴量ごとに正規化
        normalized_data = np.zeros_like(data)
        
        for i in range(data.shape[1]):
            column = data[:, i]
            min_val = np.min(column)
            max_val = np.max(column)
            
            if max_val != min_val:
                normalized_data[:, i] = (column - min_val) / (max_val - min_val)
            else:
                normalized_data[:, i] = 0.5  # 定数の場合は0.5に設定
        
        return normalized_data
    
    def get_feature_importance_names(self) -> list:
        """特徴量の重要度分析用の名前リストを取得"""
        feature_names = []
        
        # 基本特徴量
        feature_names.extend(['mid_price', 'spread', 'spread_pips', 'price_change', 
                             'price_change_pct', 'price_volatility'])
        
        # 移動平均
        for period in self.features_config['sma_periods']:
            feature_names.extend([f'sma_{period}', f'price_vs_sma_{period}'])
        
        for period in self.features_config['ema_periods']:
            feature_names.extend([f'ema_{period}', f'price_vs_ema_{period}'])
        
        # テクニカル指標
        feature_names.extend(['bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position'])
        feature_names.extend(['macd_line', 'macd_signal', 'macd_histogram'])
        feature_names.extend(['rsi', 'atr', 'cci'])
        
        # モメンタム指標
        for period in [5, 10, 20]:
            feature_names.append(f'momentum_{period}')
        for period in [5, 10]:
            feature_names.append(f'roc_{period}')
        feature_names.append('williams_r')
        
        return feature_names