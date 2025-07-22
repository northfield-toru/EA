import pandas as pd
import numpy as np

def create_labels(df: pd.DataFrame, tp_pips=4.0, sl_pips=6.0, confidence_threshold=0.58) -> np.ndarray:
    """
    ティックデータに基づいて BUY / SELL / NO_TRADE のラベルを作成する（one-hot形式）

    Parameters:
        df (pd.DataFrame): ['timestamp', 'bid', 'ask', 'mid'] 列を含むデータ
        tp_pips (float): 利確幅（pips）
        sl_pips (float): 損切幅（pips）
        confidence_threshold (float): エントリー判断に使用（現時点では未使用）

    Returns:
        np.ndarray: One-hotラベル配列（shape: [len, 3]）
    """
    mid_prices = df['mid'].values
    labels = []

    pip_value = 0.01  # USDJPYティックでのpips幅（小数第2位で1pips）

    for i in range(len(mid_prices)):
        label = [0, 0, 1]  # デフォルト: NO_TRADE

        # 未来10ティックを確認
        if i + 10 < len(mid_prices):
            future_prices = mid_prices[i+1:i+11]
            entry_price = mid_prices[i]
            tp = entry_price + tp_pips * pip_value
            sl = entry_price - sl_pips * pip_value

            if any(p >= tp for p in future_prices):
                label = [1, 0, 0]  # BUY
            elif any(p <= sl for p in future_prices):
                label = [0, 1, 0]  # SELL

        labels.append(label)

    return np.array(labels)
