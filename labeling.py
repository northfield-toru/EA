import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical

def create_labels(df, future_window=100, tp_pips=8.0, sl_pips=4.0):
    """
    ティックデータからラベル（BUY/SELL/NO_TRADE）を生成
    """
    mid_prices = df['mid']
    labels = []

    for i in range(len(mid_prices)):
        if i + future_window >= len(mid_prices):
            labels.append(0)  # NO_TRADE
            continue

        future_slice = mid_prices[i+1:i+1+future_window]
        current_price = mid_prices[i]

        max_up = (future_slice.max() - current_price) * 100  # pips
        max_down = (current_price - future_slice.min()) * 100  # pips

        if max_up >= tp_pips and max_down <= sl_pips:
            labels.append(1)  # BUY
        elif max_down >= tp_pips and max_up <= sl_pips:
            labels.append(2)  # SELL
        else:
            labels.append(0)  # NO_TRADE

    # カテゴリカル（one-hot）に変換
    return to_categorical(labels, num_classes=3)
