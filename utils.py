import json
import os

def load_config(config_path="config.json"):
    """設定ファイル(config.json)を読み込んで辞書で返す"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    return config

def price_to_pips(price_diff: float) -> float:
    """
    価格差をpipsに変換（USDJPY用、1pips = 0.01）
    """
    return price_diff * 100

def pips_to_price(pips: float) -> float:
    """
    pipsを価格差に変換（USDJPY用）
    """
    return pips / 100

def compute_spread(bid: float, ask: float) -> float:
    """
    bidとaskからスプレッド（pips）を計算
    """
    return price_to_pips(abs(ask - bid))

def apply_fixed_spread(mid_price: float, spread_pips: float = 0.7) -> tuple:
    """
    MID価格からスプレッド付きのBID/ASKを生成（スプレッドは片側0.35pips）
    """
    half_spread = pips_to_price(spread_pips / 2)
    return mid_price - half_spread, mid_price + half_spread

def format_timestamp(date_str: str, time_str: str) -> str:
    """
    DATE列とTIME列を結合して標準的なタイムスタンプに整形
    例: "2025.01.01" + "22:05:18.532" → "2025-01-01 22:05:18.532"
    """
    date_part = date_str.replace('.', '-')
    return f"{date_part} {time_str}"

def clip_probability(p: float, eps: float = 1e-6) -> float:
    """
    log関数用などに0~1の範囲で値をクリップ
    """
    return max(eps, min(1 - eps, p))
