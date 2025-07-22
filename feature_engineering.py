import pandas as pd
import numpy as np

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    MID価格をもとにテクニカル指標を計算し、DataFrameに追加。

    Parameters:
        df (pd.DataFrame): ['timestamp', 'bid', 'ask', 'mid']列を含むティックデータ

    Returns:
        pd.DataFrame: テクニカル指標を追加したDataFrame
    """

    df = df.copy()

    # ===============================
    # トレンド系
    # ===============================
    df['sma_20'] = df['mid'].rolling(window=20).mean()
    df['sma_50'] = df['mid'].rolling(window=50).mean()
    df['ema_20'] = df['mid'].ewm(span=20, adjust=False).mean()

    # ===============================
    # ボリンジャーバンド
    # ===============================
    rolling_std = df['mid'].rolling(window=20).std()
    df['bb_upper'] = df['sma_20'] + 2 * rolling_std
    df['bb_lower'] = df['sma_20'] - 2 * rolling_std

    # ===============================
    # RSI（14）
    # ===============================
    delta = df['mid'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # ===============================
    # MACD（12, 26, 9）
    # ===============================
    ema_12 = df['mid'].ewm(span=12, adjust=False).mean()
    ema_26 = df['mid'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # ===============================
    # CCI（14）
    # ===============================
    tp = df['mid']  # ティックではHigh/Lowなし → MIDを代用
    ma = tp.rolling(window=14).mean()
    md = tp.rolling(window=14).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    df['cci_14'] = (tp - ma) / (0.015 * md + 1e-9)

    # ===============================
    # ATR（14） - ティック近似
    # ===============================
    prev_mid = df['mid'].shift(1)
    df['atr_14'] = (df['mid'] - prev_mid).abs().rolling(window=14).mean()

    # ===============================
    # Volume変化率（ティック数ベース）
    # ===============================
    df['volume'] = 1  # 1tick = 1volumeと見なす
    df['vol_change'] = df['volume'].rolling(window=10).sum().pct_change()

    # ===============================
    # 欠損除去
    # ===============================
    df.dropna(inplace=True)

    return df
