"""
USDJPY スキャルピングEA用 特徴量エンジニアリング
基本テクニカル指標の計算と特徴量生成
未来リーク防止を徹底した実装
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
import talib
from scipy import stats

from utils import USDJPYUtils

# ロガー設定
logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """テクニカル指標計算クラス"""
    
    def __init__(self):
        self.utils = USDJPYUtils()
    
    def calculate_sma(self, data: pd.Series, period: int) -> pd.Series:
        """単純移動平均（SMA）"""
        return data.rolling(window=period, min_periods=period).mean()
    
    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """指数移動平均（EMA）"""
        return data.ewm(span=period, adjust=False).mean()
    
    def calculate_bollinger_bands(self, data: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """
        ボリンジャーバンド
        Args:
            data: 価格データ
            period: 期間
            std_dev: 標準偏差倍率
        Returns:
            dict: upper, middle, lower, bandwidth, %b
        """
        sma = self.calculate_sma(data, period)
        std = data.rolling(window=period, min_periods=period).std()
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        # バンド幅とポジション
        bandwidth = (upper - lower) / sma * 100
        percent_b = (data - lower) / (upper - lower) * 100
        
        return {
            'bb_upper': upper,
            'bb_middle': sma,
            'bb_lower': lower,
            'bb_bandwidth': bandwidth,
            'bb_percent_b': percent_b
        }
    
    def calculate_macd(self, data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """
        MACD
        Args:
            data: 価格データ
            fast: 短期EMA期間
            slow: 長期EMA期間
            signal: シグナル線期間
        Returns:
            dict: macd, signal, histogram
        """
        ema_fast = self.calculate_ema(data, fast)
        ema_slow = self.calculate_ema(data, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_histogram': histogram
        }
    
    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """
        RSI（相対力指数）
        Args:
            data: 価格データ
            period: 期間
        Returns:
            Series: RSI値
        """
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        ATR（Average True Range）
        Args:
            high: 高値
            low: 安値  
            close: 終値
            period: 期間
        Returns:
            Series: ATR値
        """
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period, min_periods=period).mean()
        
        return atr
    
    def calculate_cci(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        CCI（Commodity Channel Index）
        Args:
            high: 高値
            low: 安値
            close: 終値
            period: 期間
        Returns:
            Series: CCI値
        """
        typical_price = (high + low + close) / 3
        sma_tp = self.calculate_sma(typical_price, period)
        
        # 平均偏差計算
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        
        return cci
    
    def calculate_volume_features(self, volume: pd.Series, period: int = 20) -> Dict[str, pd.Series]:
        """
        ボリューム関連特徴量
        Args:
            volume: ボリューム（ティック数）
            period: 期間
        Returns:
            dict: ボリューム特徴量
        """
        # ボリューム移動平均
        vol_sma = self.calculate_sma(volume, period)
        
        # ボリューム比率
        vol_ratio = volume / vol_sma
        
        # ボリューム変化率
        vol_change = volume.pct_change()
        
        # ボリューム標準化
        vol_std = volume.rolling(window=period, min_periods=period).std()
        vol_zscore = (volume - vol_sma) / vol_std
        
        return {
            'volume_sma': vol_sma,
            'volume_ratio': vol_ratio,
            'volume_change': vol_change,
            'volume_zscore': vol_zscore
        }
    
    def calculate_price_features(self, ohlc: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        価格関連特徴量
        Args:
            ohlc: OHLC DataFrame
        Returns:
            dict: 価格特徴量
        """
        # 価格変化率
        returns = ohlc['close'].pct_change()
        
        # ローソク足の実体とヒゲ
        body = abs(ohlc['close'] - ohlc['open'])
        upper_shadow = ohlc['high'] - ohlc[['open', 'close']].max(axis=1)
        lower_shadow = ohlc[['open', 'close']].min(axis=1) - ohlc['low']
        
        # レンジ
        range_hl = ohlc['high'] - ohlc['low']
        
        # 終値位置（高安値の中での位置）
        close_position = (ohlc['close'] - ohlc['low']) / range_hl
        
        return {
            'returns': returns,
            'body_size': body,
            'upper_shadow': upper_shadow,
            'lower_shadow': lower_shadow,
            'range_hl': range_hl,
            'close_position': close_position
        }


class FeatureEngineer:
    """特徴量エンジニアリングメインクラス"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.utils = USDJPYUtils()
        
    def create_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        基本特徴量を生成
        Args:
            df: OHLCV DataFrame
        Returns:
            DataFrame: 特徴量付きDataFrame
        """
        logger.info("基本特徴量生成開始...")
        
        result_df = df.copy()
        
        # 価格特徴量
        price_features = self.indicators.calculate_price_features(df)
        for name, series in price_features.items():
            result_df[name] = series
        
        # ボリンジャーバンド
        bb_features = self.indicators.calculate_bollinger_bands(df['close'])
        for name, series in bb_features.items():
            result_df[name] = series
        
        # MACD
        macd_features = self.indicators.calculate_macd(df['close'])
        for name, series in macd_features.items():
            result_df[name] = series
        
        # RSI
        result_df['rsi'] = self.indicators.calculate_rsi(df['close'])
        
        # ATR
        result_df['atr'] = self.indicators.calculate_atr(df['high'], df['low'], df['close'])
        
        # CCI
        result_df['cci'] = self.indicators.calculate_cci(df['high'], df['low'], df['close'])
        
        # ボリューム特徴量
        vol_features = self.indicators.calculate_volume_features(df['volume'])
        for name, series in vol_features.items():
            result_df[name] = series
        
        # 移動平均
        for period in [5, 10, 20, 50]:
            result_df[f'sma_{period}'] = self.indicators.calculate_sma(df['close'], period)
            result_df[f'ema_{period}'] = self.indicators.calculate_ema(df['close'], period)
        
        logger.info(f"基本特徴量生成完了: {len(result_df.columns)} 列")
        return result_df
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        高度な特徴量を生成
        Args:
            df: 基本特徴量付きDataFrame
        Returns:
            DataFrame: 高度特徴量付きDataFrame
        """
        logger.info("高度な特徴量生成開始...")
        
        result_df = df.copy()
        
        # 価格の勢い（Momentum）
        for period in [3, 5, 10]:
            result_df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        
        # ボラティリティ
        for period in [5, 10, 20]:
            result_df[f'volatility_{period}'] = df['returns'].rolling(period).std()
        
        # 価格とボリンジャーバンドの関係
        result_df['bb_squeeze'] = (result_df['bb_upper'] - result_df['bb_lower']) / result_df['bb_middle']
        result_df['bb_position'] = (df['close'] - result_df['bb_lower']) / (result_df['bb_upper'] - result_df['bb_lower'])
        
        # 移動平均の傾き
        for period in [5, 10, 20]:
            ma_col = f'sma_{period}'
            if ma_col in result_df.columns:
                result_df[f'sma_{period}_slope'] = result_df[ma_col].diff()
        
        # RSIの勢い
        result_df['rsi_momentum'] = result_df['rsi'].diff()
        
        # MACDの強さ
        if 'macd_histogram' in result_df.columns:
            result_df['macd_strength'] = result_df['macd_histogram'].rolling(5).mean()
        
        # ATR正規化された価格変動
        if 'atr' in result_df.columns:
            result_df['normalized_range'] = result_df['range_hl'] / result_df['atr']
            result_df['normalized_body'] = result_df['body_size'] / result_df['atr']
        
        logger.info(f"高度な特徴量生成完了: {len(result_df.columns)} 列")
        return result_df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        時間関連特徴量を生成
        Args:
            df: DataFrame（DatetimeIndexが必要）
        Returns:
            DataFrame: 時間特徴量付きDataFrame
        """
        logger.info("時間特徴量生成開始...")
        
        result_df = df.copy()
        
        # 既存の時間特徴量があれば使用、なければ作成
        if 'hour' not in result_df.columns:
            result_df['hour'] = result_df.index.hour
        if 'minute' not in result_df.columns:
            result_df['minute'] = result_df.index.minute
        if 'weekday' not in result_df.columns:
            result_df['weekday'] = result_df.index.weekday
        
        # 時間帯の特徴量エンコーディング
        # 市場オープン時間の近似
        result_df['is_tokyo_open'] = ((result_df['hour'] >= 0) & (result_df['hour'] < 9)).astype(int)
        result_df['is_london_open'] = ((result_df['hour'] >= 9) & (result_df['hour'] < 17)).astype(int)
        result_df['is_ny_open'] = ((result_df['hour'] >= 17) & (result_df['hour'] < 24)).astype(int)
        
        # 週末フラグ
        result_df['is_weekend'] = (result_df['weekday'] >= 5).astype(int)
        
        # 市場セッションを数値化
        if 'market_session' in result_df.columns:
            # 文字列を数値に変換
            session_map = {'TOKYO': 0, 'LONDON': 1, 'NY': 2, 'OTHER': 3}
            result_df['market_session_encoded'] = result_df['market_session'].map(session_map).fillna(3)
            # 元の文字列列を削除
            result_df = result_df.drop('market_session', axis=1)
        
        # 時間の循環特徴量（sin/cos エンコーディング）
        result_df['hour_sin'] = np.sin(2 * np.pi * result_df['hour'] / 24)
        result_df['hour_cos'] = np.cos(2 * np.pi * result_df['hour'] / 24)
        result_df['minute_sin'] = np.sin(2 * np.pi * result_df['minute'] / 60)
        result_df['minute_cos'] = np.cos(2 * np.pi * result_df['minute'] / 60)
        result_df['weekday_sin'] = np.sin(2 * np.pi * result_df['weekday'] / 7)
        result_df['weekday_cos'] = np.cos(2 * np.pi * result_df['weekday'] / 7)
        
        logger.info(f"時間特徴量生成完了: {len(result_df.columns)} 列")
        return result_df
    
    def create_lag_features(self, df: pd.DataFrame, target_cols: List[str], lags: List[int] = [1, 2, 3, 5]) -> pd.DataFrame:
        """
        ラグ特徴量を生成
        Args:
            df: DataFrame
            target_cols: ラグを作成する列名
            lags: ラグ期間のリスト
        Returns:
            DataFrame: ラグ特徴量付きDataFrame
        """
        logger.info(f"ラグ特徴量生成開始: {target_cols}")
        
        result_df = df.copy()
        
        for col in target_cols:
            if col in df.columns:
                for lag in lags:
                    result_df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        logger.info(f"ラグ特徴量生成完了: {len(result_df.columns)} 列")
        return result_df
    
    def create_all_features(self, df: pd.DataFrame, include_advanced: bool = True, include_lags: bool = False) -> pd.DataFrame:
        """
        全特徴量を生成
        Args:
            df: OHLCV DataFrame
            include_advanced: 高度な特徴量を含めるか
            include_lags: ラグ特徴量を含めるか
        Returns:
            DataFrame: 全特徴量付きDataFrame
        """
        logger.info("全特徴量生成開始...")
        
        # 基本特徴量
        result_df = self.create_base_features(df)
        
        # 時間特徴量
        result_df = self.create_time_features(result_df)
        
        # 高度な特徴量
        if include_advanced:
            result_df = self.create_advanced_features(result_df)
        
        # ラグ特徴量
        if include_lags:
            key_cols = ['close', 'returns', 'rsi', 'macd', 'bb_percent_b']
            result_df = self.create_lag_features(result_df, key_cols)
        
        # 無限値・NaN値の処理
        result_df = result_df.replace([np.inf, -np.inf], np.nan)
        
        logger.info(f"全特徴量生成完了: {len(result_df.columns)} 列")
        logger.info(f"NaN値の数: {result_df.isna().sum().sum()}")
        
        return result_df
    
    def get_feature_importance_analysis(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        特徴量重要度の簡易分析
        Args:
            df: 特徴量DataFrame
            target_col: ターゲット列名
        Returns:
            DataFrame: 特徴量重要度
        """
        logger.info("特徴量重要度分析開始...")
        
        # 数値型列のみ選択
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        importance_data = []
        
        for col in numeric_cols:
            if col in df.columns and target_col in df.columns:
                # 相関係数
                correlation = df[col].corr(df[target_col])
                
                # 欠損値の割合
                missing_ratio = df[col].isna().sum() / len(df)
                
                importance_data.append({
                    'feature': col,
                    'correlation': abs(correlation) if not np.isnan(correlation) else 0,
                    'missing_ratio': missing_ratio
                })
        
        importance_df = pd.DataFrame(importance_data)
        importance_df = importance_df.sort_values('correlation', ascending=False)
        
        logger.info(f"特徴量重要度分析完了: {len(importance_df)} 特徴量")
        
        return importance_df


def create_sample_features(ohlcv_df: pd.DataFrame, sample_size: int = 1000) -> pd.DataFrame:
    """
    サンプルデータで特徴量生成テスト
    Args:
        ohlcv_df: OHLCV DataFrame
        sample_size: サンプルサイズ
    Returns:
        DataFrame: 特徴量付きサンプルデータ
    """
    # サンプル抽出
    if len(ohlcv_df) > sample_size:
        sample_df = ohlcv_df.tail(sample_size).copy()
    else:
        sample_df = ohlcv_df.copy()
    
    # 特徴量生成
    engineer = FeatureEngineer()
    features_df = engineer.create_all_features(sample_df, include_advanced=True, include_lags=True)
    
    return features_df


if __name__ == "__main__":
    # テスト実行
    print("=== 特徴量エンジニアリングテスト ===")
    
    # サンプルデータ作成
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    
    sample_data = pd.DataFrame({
        'open': 118.0 + np.cumsum(np.random.randn(1000) * 0.001),
        'high': 0,
        'low': 0,
        'close': 0,
        'volume': np.random.randint(10, 100, 1000)
    }, index=dates)
    
    # high, low を調整
    sample_data['high'] = sample_data['open'] + abs(np.random.randn(1000) * 0.005)
    sample_data['low'] = sample_data['open'] - abs(np.random.randn(1000) * 0.005)
    sample_data['close'] = sample_data['open'] + np.random.randn(1000) * 0.002
    
    try:
        # 特徴量生成テスト
        features_df = create_sample_features(sample_data)
        
        print(f"特徴量生成完了: {len(features_df.columns)} 列")
        print("\n特徴量一覧:")
        print(features_df.columns.tolist())
        print("\nデータサンプル:")
        print(features_df.head()[['close', 'rsi', 'macd', 'bb_percent_b']].round(4))
        
        # 欠損値チェック
        missing_counts = features_df.isna().sum()
        print(f"\n欠損値の多い特徴量（上位10）:")
        print(missing_counts.sort_values(ascending=False).head(10))
        
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()