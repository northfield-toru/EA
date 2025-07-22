"""
USDJPY スキャルピングEA用 ユーティリティ関数（修正版）
共通的な計算処理、pips変換、スプレッド処理など
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime
import re

# ロガー設定
logger = logging.getLogger(__name__)

class USDJPYUtils:
    """USDJPY専用のユーティリティクラス（修正版）"""
    
    # USDJPY固有設定
    SPREAD_PIPS = 0.7  # 0.7pips
    PIP_VALUE = 0.01   # 1pip = 0.01
    PAIR_NAME = "USDJPY"
    
    @staticmethod
    def price_to_pips(price_diff):
        """
        価格差をpips単位に変換
        Args:
            price_diff (float): 価格差
        Returns:
            float: pips単位の値
        """
        return price_diff / USDJPYUtils.PIP_VALUE
    
    @staticmethod
    def pips_to_price(pips):
        """
        pips単位の値を価格差に変換
        Args:
            pips (float): pips単位の値
        Returns:
            float: 価格差
        """
        return pips * USDJPYUtils.PIP_VALUE
    
    @staticmethod
    def calculate_mid_price(bid, ask):
        """
        MID価格を計算
        Args:
            bid (float or Series): bid価格
            ask (float or Series): ask価格
        Returns:
            float or Series: MID価格
        """
        return (bid + ask) / 2.0
    
    @staticmethod
    def calculate_spread_pips(bid, ask):
        """
        スプレッドをpips単位で計算
        Args:
            bid (float or Series): bid価格
            ask (float or Series): ask価格
        Returns:
            float or Series: スプレッド（pips）
        """
        return USDJPYUtils.price_to_pips(ask - bid)
    
    @staticmethod
    def validate_price_data(df, required_columns=['bid', 'ask']):
        """
        価格データの妥当性チェック
        Args:
            df (DataFrame): 価格データ
            required_columns (list): 必須カラム名
        Returns:
            tuple: (is_valid, error_messages)
        """
        errors = []
        
        # 必須カラムの存在チェック
        for col in required_columns:
            if col not in df.columns:
                errors.append(f"必須カラム '{col}' が見つかりません")
        
        if errors:
            return False, errors
        
        # 価格データの妥当性チェック
        if (df['bid'] <= 0).any():
            errors.append("bid価格に0以下の値が含まれています")
        
        if (df['ask'] <= 0).any():
            errors.append("ask価格に0以下の値が含まれています")
        
        if (df['ask'] < df['bid']).any():
            errors.append("ask < bid の異常な価格データが含まれています")
        
        # スプレッドの異常チェック（50pips以上を異常とみなす）
        spread_pips = USDJPYUtils.calculate_spread_pips(df['bid'], df['ask'])
        
        # 警告レベル（10pips以上）
        high_spread_10 = (spread_pips > 10.0).sum()
        if high_spread_10 > 0:
            logger.warning(f"高いスプレッド（10pips超）検出: {high_spread_10} 行 - 処理続行")
            
        # エラーレベル（50pips以上）
        if (spread_pips > 50.0).any():
            extreme_spread_count = (spread_pips > 50.0).sum()
            logger.warning(f"異常スプレッド（50pips超）検出: {extreme_spread_count} 行 - 警告のみ、処理続行")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def parse_timestamp_pattern1(timestamp_str):
        """
        パターン1のタイムスタンプを解析
        Format: '2003.05.04 21:00:00.626'
        """
        try:
            return datetime.strptime(timestamp_str, '%Y.%m.%d %H:%M:%S.%f')
        except ValueError:
            # ミリ秒がない場合
            try:
                return datetime.strptime(timestamp_str, '%Y.%m.%d %H:%M:%S')
            except ValueError:
                logger.error(f"タイムスタンプの解析に失敗: {timestamp_str}")
                return None
    
    @staticmethod
    def parse_timestamp_pattern2_flexible(date_str, time_str):
        """
        パターン2のタイムスタンプを柔軟に解析（修正版）
        Args:
            date_str: '2025.06.16' または '2025.06.16  ' (スペース含む可能性)
            time_str: '12:59:31.677' または複数カラムに分かれている可能性
        """
        try:
            # 🔧 修正1: 文字列の前後空白を除去
            date_clean = str(date_str).strip()
            time_clean = str(time_str).strip()
            
            # 🔧 修正2: NaN値チェック
            if pd.isna(date_str) or pd.isna(time_str) or date_clean == 'nan' or time_clean == 'nan':
                return None
            
            # 🔧 修正3: 日付・時刻の結合（複数スペースも対応）
            datetime_str = f"{date_clean} {time_clean}"
            
            # 🔧 修正4: 複数の形式を試行
            timestamp_formats = [
                '%Y.%m.%d %H:%M:%S.%f',    # 基本形式
                '%Y.%m.%d %H:%M:%S',       # ミリ秒なし
                '%Y-%m-%d %H:%M:%S.%f',    # ハイフン形式
                '%Y-%m-%d %H:%M:%S',       # ハイフン・ミリ秒なし
                '%Y/%m/%d %H:%M:%S.%f',    # スラッシュ形式
                '%Y/%m/%d %H:%M:%S'        # スラッシュ・ミリ秒なし
            ]
            
            for fmt in timestamp_formats:
                try:
                    return datetime.strptime(datetime_str, fmt)
                except ValueError:
                    continue
            
            # 🔧 修正5: 正規表現による柔軟解析
            # YYYY.MM.DD HH:MM:SS.fff 形式を抽出
            pattern = r'(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})\s+(\d{1,2}):(\d{1,2}):(\d{1,2})(?:\.(\d+))?'
            match = re.match(pattern, datetime_str)
            
            if match:
                year, month, day, hour, minute, second, microsecond = match.groups()
                
                # ミリ秒の処理
                if microsecond:
                    # 6桁に正規化（マイクロ秒）
                    if len(microsecond) <= 3:
                        microsecond = microsecond.ljust(3, '0') + '000'  # ミリ秒→マイクロ秒
                    elif len(microsecond) <= 6:
                        microsecond = microsecond.ljust(6, '0')
                    else:
                        microsecond = microsecond[:6]  # 6桁に切り詰め
                    microsecond = int(microsecond)
                else:
                    microsecond = 0
                
                return datetime(
                    int(year), int(month), int(day),
                    int(hour), int(minute), int(second),
                    microsecond
                )
            
            logger.error(f"タイムスタンプの解析に失敗: {date_str} {time_str}")
            return None
            
        except Exception as e:
            logger.error(f"タイムスタンプ解析エラー: {date_str} {time_str} - {e}")
            return None
    
    @staticmethod
    def parse_timestamp_pattern2(date_str, time_str):
        """
        パターン2のタイムスタンプを解析（既存互換性維持）
        Args:
            date_str: '2025.01.01'
            time_str: '22:05:18.532'
        """
        # 新しい柔軟な解析メソッドを使用
        return USDJPYUtils.parse_timestamp_pattern2_flexible(date_str, time_str)
    
    @staticmethod
    def get_market_session(dt):
        """
        市場セッションを判定
        Args:
            dt (datetime): 判定する時刻（UTC想定）
        Returns:
            str: 'TOKYO', 'LONDON', 'NY', 'OTHER'
        """
        hour = dt.hour
        
        # 大まかなセッション時間（UTC）
        if 0 <= hour < 9:
            return 'TOKYO'
        elif 9 <= hour < 17:
            return 'LONDON'  
        elif 17 <= hour < 24:
            return 'NY'
        else:
            return 'OTHER'
    
    @staticmethod
    def create_time_features(dt):
        """
        時間特徴量を生成
        Args:
            dt (datetime): 時刻
        Returns:
            dict: 時間特徴量
        """
        return {
            'hour': dt.hour,
            'minute': dt.minute,
            'weekday': dt.weekday(),  # 0=Monday, 6=Sunday
            'is_weekend': dt.weekday() >= 5,
            'market_session': USDJPYUtils.get_market_session(dt)
        }
    
    @staticmethod
    def detect_csv_pattern(filepath, sample_lines=5):
        """
        CSVファイルのパターンを自動判定（修正版）
        Args:
            filepath (str): CSVファイルパス
            sample_lines (int): 判定用サンプル行数
        Returns:
            str: 'pattern1' or 'pattern2'
        """
        try:
            # 複数エンコーディングを試行
            encodings = ['utf-8', 'utf-8-sig', 'cp932', 'shift_jis']
            
            for encoding in encodings:
                try:
                    with open(filepath, 'r', encoding=encoding) as f:
                        lines = []
                        for i in range(sample_lines):
                            line = f.readline().strip()
                            if line:
                                lines.append(line)
                    
                    if not lines:
                        continue
                    
                    # ヘッダーの存在チェック
                    first_line = lines[0]
                    
                    # Pattern2の特徴的なヘッダーをチェック
                    if any(header in first_line for header in ['<DATE>', '<TIME>', '<BID>', '<ASK>']):
                        logger.info(f"パターン2（タブ区切り・ヘッダーあり）として判定 (エンコーディング: {encoding})")
                        return 'pattern2'
                    
                    # カンマ区切りかタブ区切りかチェック
                    comma_count = first_line.count(',')
                    tab_count = first_line.count('\t')
                    space_count = len(first_line.split()) - 1  # スペース区切りもチェック
                    
                    if tab_count >= 2 or space_count >= 6:  # タブまたは複数スペース
                        logger.info(f"パターン2（区切り文字判定）として判定 (エンコーディング: {encoding})")
                        return 'pattern2'
                    elif comma_count >= 2:
                        logger.info(f"パターン1（カンマ区切り・ヘッダーなし）として判定 (エンコーディング: {encoding})")
                        return 'pattern1'
                    
                    break  # エンコーディング成功なので抜ける
                    
                except UnicodeDecodeError:
                    continue  # 次のエンコーディングを試行
            
            # デフォルトはパターン2
            logger.warning("パターンの自動判定に失敗。パターン2として処理します")
            return 'pattern2'
            
        except Exception as e:
            logger.error(f"CSV パターン判定エラー: {e}")
            return 'pattern2'  # デフォルト
    
    @staticmethod
    def memory_usage_mb():
        """
        現在のメモリ使用量を取得（MB）
        """
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0


class DataValidationError(Exception):
    """データ検証エラー用例外クラス"""
    pass


def setup_logging(log_level=logging.INFO, log_file=None):
    """
    ログ設定のセットアップ
    Args:
        log_level: ログレベル
        log_file: ログファイルパス（Noneならコンソール出力のみ）
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


if __name__ == "__main__":
    # 動作テスト（修正版）
    print("=== USDJPY Utils テスト（修正版） ===")
    
    # pips変換テスト
    price_diff = 0.08
    pips = USDJPYUtils.price_to_pips(price_diff)
    print(f"価格差 {price_diff} = {pips} pips")
    
    # MID価格計算テスト
    bid, ask = 118.94, 118.99
    mid = USDJPYUtils.calculate_mid_price(bid, ask)
    spread_pips = USDJPYUtils.calculate_spread_pips(bid, ask)
    print(f"BID: {bid}, ASK: {ask}")
    print(f"MID: {mid}, Spread: {spread_pips} pips")
    
    # 🔧 タイムスタンプ解析テスト（修正版）
    test_dates = ['2025.06.15', '2025.01.01']
    test_times = ['21:05:08.290', '22:05:18.532']
    
    for date_str, time_str in zip(test_dates, test_times):
        result = USDJPYUtils.parse_timestamp_pattern2_flexible(date_str, time_str)
        print(f"タイムスタンプテスト: {date_str} {time_str} → {result}")
    
    # 時間特徴量テスト
    test_dt = datetime.now()
    time_features = USDJPYUtils.create_time_features(test_dt)
    print(f"時間特徴量: {time_features}")
    
    print("テスト完了（修正版）")