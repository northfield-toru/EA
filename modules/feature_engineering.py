import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple

class FeatureEngineer:
    """特徴量エンジニアリングクラス"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.indicators_config = config['indicators']
        self.feature_window = config['data']['feature_window']
        self.logger = logging.getLogger(__name__)
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        テクニカル指標を計算
        
        Args:
            df: ティックデータ（MID価格含む）
            
        Returns:
            DataFrame: 指標を追加したデータフレーム
        """
        df = df.copy()
        
        # ボリンジャーバンド
        if self.indicators_config['use_bbands']:
            df = self._calculate_bollinger_bands(df)
            
        # MACD
        if self.indicators_config['use_macd']:
            df = self._calculate_macd(df)
            
        # RSI
        if self.indicators_config['use_rsi']:
            df = self._calculate_rsi(df)
            
        # ATR
        if self.indicators_config['use_atr']:
            df = self._calculate_atr(df)
            
        # CCI
        if self.indicators_config['use_cci']:
            df = self._calculate_cci(df)
            
        # Volume（存在する場合）
        if self.indicators_config['use_volume'] and 'VOLUME' in df.columns:
            df = self._calculate_volume_indicators(df)
            
        # SMA
        if self.indicators_config['use_sma']:
            df = self._calculate_sma(df)
            
        # EMA
        if self.indicators_config['use_ema']:
            df = self._calculate_ema(df)
            
        # 価格変動率
        df = self._calculate_price_changes(df)
        
        self.logger.info(f"Calculated indicators, features shape: {df.shape}")
        return df
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """ボリンジャーバンド計算"""
        period = self.indicators_config.get('bbands_period', 20)
        std_dev = self.indicators_config.get('bbands_std', 2)
        
        # 移動平均とローリング標準偏差
        df['BB_SMA'] = df['MID'].rolling(window=period, min_periods=1).mean()
        bb_std = df['MID'].rolling(window=period, min_periods=1).std()
        
        # バンド計算
        df['BB_Upper'] = df['BB_SMA'] + (bb_std * std_dev)
        df['BB_Lower'] = df['BB_SMA'] - (bb_std * std_dev)
        
        # BB位置とバンド幅
        df['BB_Position'] = (df['MID'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['MID']
        
        return df
    
    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """MACD計算"""
        fast = self.indicators_config.get('macd_fast', 12)
        slow = self.indicators_config.get('macd_slow', 26)
        signal = self.indicators_config.get('macd_signal', 9)
        
        # EMA計算
        ema_fast = df['MID'].ewm(span=fast).mean()
        ema_slow = df['MID'].ewm(span=slow).mean()
        
        # MACD線
        df['MACD'] = ema_fast - ema_slow
        
        # シグナル線
        df['MACD_Signal'] = df['MACD'].ewm(span=signal).mean()
        
        # ヒストグラム
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        return df
    
    def _calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSI計算"""
        period = self.indicators_config.get('rsi_period', 14)
        
        # 価格変動
        delta = df['MID'].diff()
        
        # 上昇・下落分離
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # 平均値計算（EMA）
        avg_gain = gain.ewm(span=period).mean()
        avg_loss = loss.ewm(span=period).mean()
        
        # RSI計算
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """ATR計算"""
        period = self.indicators_config.get('atr_period', 14)
        
        # High, Low, Closeの代わりにASK, BID, MIDを使用
        df['TR1'] = df['ASK'] - df['BID']  # スプレッド
        df['TR2'] = abs(df['ASK'] - df['MID'].shift(1))
        df['TR3'] = abs(df['BID'] - df['MID'].shift(1))
        
        # True Range
        df['TR'] = df[['TR1', 'TR2', 'TR3']].max(axis=1)
        
        # ATR
        df['ATR'] = df['TR'].ewm(span=period).mean()
        
        # 不要な列を削除
        df.drop(['TR1', 'TR2', 'TR3', 'TR'], axis=1, inplace=True)
        
        return df
    
    def _calculate_cci(self, df: pd.DataFrame) -> pd.DataFrame:
        """CCI計算"""
        period = self.indicators_config.get('cci_period', 14)
        
        # Typical Price (MID価格を使用)
        tp = df['MID']
        
        # SMAとMean Deviation
        sma_tp = tp.rolling(window=period, min_periods=1).mean()
        mad = tp.rolling(window=period, min_periods=1).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        
        # CCI
        df['CCI'] = (tp - sma_tp) / (0.015 * mad)
        
        return df
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """ボリューム指標計算"""
        if 'VOLUME' not in df.columns:
            return df
            
        # ボリューム移動平均
        df['Volume_SMA'] = df['VOLUME'].rolling(window=20, min_periods=1).mean()
        
        # ボリューム比
        df['Volume_Ratio'] = df['VOLUME'] / df['Volume_SMA']
        
        return df
    
    def _calculate_sma(self, df: pd.DataFrame) -> pd.DataFrame:
        """SMA計算"""
        periods = self.indicators_config.get('sma_periods', [5, 10, 20])
        
        for period in periods:
            df[f'SMA_{period}'] = df['MID'].rolling(window=period, min_periods=1).mean()
            
        return df
    
    def _calculate_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """EMA計算"""
        periods = self.indicators_config.get('ema_periods', [5, 10, 20])
        
        for period in periods:
            df[f'EMA_{period}'] = df['MID'].ewm(span=period).mean()
            
        return df
    
    def _calculate_price_changes(self, df: pd.DataFrame) -> pd.DataFrame:
        """価格変動率計算"""
        # 基本的な価格変動
        df['Price_Change'] = df['MID'].pct_change()
        df['Price_Change_5'] = df['MID'].pct_change(5)
        df['Price_Change_10'] = df['MID'].pct_change(10)
        
        # ログリターン
        df['Log_Return'] = np.log(df['MID'] / df['MID'].shift(1))
        
        # 価格レンジ
        df['Spread'] = df['ASK'] - df['BID']
        df['Spread_Ratio'] = df['Spread'] / df['MID']
        
        return df
    
    def create_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        特徴量配列を作成
        
        Args:
            df: 指標計算済みデータフレーム
            
        Returns:
            Tuple: (特徴量配列, 特徴量名リスト)
        """
        # 特徴量として使用する列を選択
        feature_columns = []
        
        # 基本価格情報
        feature_columns.extend(['MID', 'BID', 'ASK', 'Spread', 'Spread_Ratio'])
        
        # 価格変動
        feature_columns.extend(['Price_Change', 'Price_Change_5', 'Price_Change_10', 'Log_Return'])
        
        # 各指標を条件に応じて追加
        if self.indicators_config['use_bbands']:
            feature_columns.extend(['BB_Position', 'BB_Width'])
            
        if self.indicators_config['use_macd']:
            feature_columns.extend(['MACD', 'MACD_Signal', 'MACD_Histogram'])
            
        if self.indicators_config['use_rsi']:
            feature_columns.append('RSI')
            
        if self.indicators_config['use_atr']:
            feature_columns.append('ATR')
            
        if self.indicators_config['use_cci']:
            feature_columns.append('CCI')
            
        if self.indicators_config['use_volume'] and 'VOLUME' in df.columns:
            feature_columns.extend(['Volume_Ratio'])
            
        if self.indicators_config['use_sma']:
            periods = self.indicators_config.get('sma_periods', [5, 10, 20])
            feature_columns.extend([f'SMA_{p}' for p in periods])
            
        if self.indicators_config['use_ema']:
            periods = self.indicators_config.get('ema_periods', [5, 10, 20])
            feature_columns.extend([f'EMA_{p}' for p in periods])
        
        # 存在する特徴量のみを選択
        available_features = [col for col in feature_columns if col in df.columns]
        
        if not available_features:
            raise ValueError("No valid features found!")
        
        # NaN処理
        feature_df = df[available_features].fillna(method='ffill').fillna(0)
        
        self.logger.info(f"Selected features: {available_features}")
        
        return feature_df.values, available_features
    
    def create_sequences(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        シーケンス（時系列窓）を作成
        
        Args:
            features: 特徴量配列
            labels: ラベル配列
            
        Returns:
            Tuple: (シーケンス特徴量, 対応ラベル)
        """
        X, y = [], []
        
        # 未来リーク防止: feature_window分の過去データのみ使用
        for i in range(self.feature_window, len(features)):
            # 過去のfeature_window分のデータを使用
            X.append(features[i-self.feature_window:i])
            y.append(labels[i])
        
        X = np.array(X)
        y = np.array(y)
        
        self.logger.info(f"Created sequences: X shape {X.shape}, y shape {y.shape}")
        
        return X, y
    
    def normalize_features(self, X_train: np.ndarray, X_val: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        特徴量の正規化
        
        Args:
            X_train: 訓練データ
            X_val: 検証データ（オプション）
            
        Returns:
            Tuple: (正規化済み訓練データ, 正規化済み検証データ, 正規化パラメータ)
        """
        # 訓練データから正規化パラメータを計算
        # 形状: (samples, timesteps, features)
        
        # 各特徴量ごとに平均と標準偏差を計算
        mean = np.mean(X_train, axis=(0, 1))  # (features,)
        std = np.std(X_train, axis=(0, 1))    # (features,)
        
        # 標準偏差が0の場合は1に置換（ゼロ除算回避）
        std = np.where(std == 0, 1, std)
        
        # 正規化実行
        X_train_normalized = (X_train - mean) / std
        
        normalization_params = {
            'mean': mean,
            'std': std
        }
        
        X_val_normalized = None
        if X_val is not None:
            X_val_normalized = (X_val - mean) / std
        
        self.logger.info("Feature normalization completed")
        
        return X_train_normalized, X_val_normalized, normalization_params