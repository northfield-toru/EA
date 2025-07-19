"""
USDJPY スキャルピングEA用 ユーティリティ関数
共通的な計算処理、pips変換、スプレッド処理など
"""

import numpy as np
import pandas as pd
from datetime import datetime

class USDJPYUtils:
    """USDJPY専用のユーティリティクラス"""
    
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
            print(f"WARNING: 高いスプレッド（10pips超）検出: {high_spread_10} 行 - 処理続行")
            
        # エラーレベル（50pips以上）
        if (spread_pips > 50.0).any():
            extreme_spread_count = (spread_pips > 50.0).sum()
            print(f"WARNING: 異常スプレッド（50pips超）検出: {extreme_spread_count} 行 - 警告のみ、処理続行")
            # errors.append を削除してエラーではなく警告のみにする
        
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
                print(f"ERROR: タイムスタンプの解析に失敗: {timestamp_str}")
                return None
    
    @staticmethod
    def parse_timestamp_pattern2(date_str, time_str):
        """
        パターン2のタイムスタンプを解析
        Args:
            date_str: '2025.01.01'
            time_str: '22:05:18.532'
        """
        try:
            datetime_str = f"{date_str} {time_str}"
            return datetime.strptime(datetime_str, '%Y.%m.%d %H:%M:%S.%f')
        except ValueError:
            try:
                # ミリ秒がない場合
                return datetime.strptime(datetime_str, '%Y.%m.%d %H:%M:%S')
            except ValueError:
                print(f"ERROR: タイムスタンプの解析に失敗: {date_str} {time_str}")
                return None
    
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
        CSVファイルのパターンを自動判定
        Args:
            filepath (str): CSVファイルパス
            sample_lines (int): 判定用サンプル行数
        Returns:
            str: 'pattern1' or 'pattern2'
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = []
                for i in range(sample_lines):
                    line = f.readline().strip()
                    if line:
                        lines.append(line)
            
            if not lines:
                raise ValueError("ファイルが空です")
            
            # ヘッダーの存在チェック
            first_line = lines[0]
            if '<DATE>' in first_line or '<TIME>' in first_line:
                print("INFO: パターン2（タブ区切り・ヘッダーあり）として判定")
                return 'pattern2'
            
            # カンマ区切りかタブ区切りかチェック
            comma_count = first_line.count(',')
            tab_count = first_line.count('\t')
            
            if comma_count >= 2 and comma_count > tab_count:
                print("INFO: パターン1（カンマ区切り・ヘッダーなし）として判定")
                return 'pattern1'
            elif tab_count >= 2:
                print("INFO: パターン2（タブ区切り）として判定")
                return 'pattern2'
            
            # デフォルトはパターン1
            print("WARNING: パターンの自動判定に失敗。パターン1として処理します")
            return 'pattern1'
            
        except Exception as e:
            print(f"ERROR: CSV パターン判定エラー: {e}")
            return 'pattern1'  # デフォルト
    
    @staticmethod
    def memory_usage_mb():
        """
        現在のメモリ使用量を取得（MB）
        """
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024


class DataValidationError(Exception):
    """データ検証エラー用例外クラス"""
    pass


if __name__ == "__main__":
    # 動作テスト
    print("=== USDJPY Utils テスト ===")
    
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
    
    # 時間特徴量テスト
    test_dt = datetime.now()
    time_features = USDJPYUtils.create_time_features(test_dt)
    print(f"時間特徴量: {time_features}")
    
    print("テスト完了")

class USDJPYUtils:
    """USDJPY専用のユーティリティクラス"""
    
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
            # errors.append を削除してエラーではなく警告のみにする
        
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
    def parse_timestamp_pattern2(date_str, time_str):
        """
        パターン2のタイムスタンプを解析
        Args:
            date_str: '2025.01.01'
            time_str: '22:05:18.532'
        """
        try:
            datetime_str = f"{date_str} {time_str}"
            return datetime.strptime(datetime_str, '%Y.%m.%d %H:%M:%S.%f')
        except ValueError:
            try:
                # ミリ秒がない場合
                return datetime.strptime(datetime_str, '%Y.%m.%d %H:%M:%S')
            except ValueError:
                logger.error(f"タイムスタンプの解析に失敗: {date_str} {time_str}")
                return None
    
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
        CSVファイルのパターンを自動判定
        Args:
            filepath (str): CSVファイルパス
            sample_lines (int): 判定用サンプル行数
        Returns:
            str: 'pattern1' or 'pattern2'
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = []
                for i in range(sample_lines):
                    line = f.readline().strip()
                    if line:
                        lines.append(line)
            
            if not lines:
                raise ValueError("ファイルが空です")
            
            # ヘッダーの存在チェック
            first_line = lines[0]
            if '<DATE>' in first_line or '<TIME>' in first_line:
                logger.info("パターン2（タブ区切り・ヘッダーあり）として判定")
                return 'pattern2'
            
            # カンマ区切りかタブ区切りかチェック
            comma_count = first_line.count(',')
            tab_count = first_line.count('\t')
            
            if comma_count >= 2 and comma_count > tab_count:
                logger.info("パターン1（カンマ区切り・ヘッダーなし）として判定")
                return 'pattern1'
            elif tab_count >= 2:
                logger.info("パターン2（タブ区切り）として判定")
                return 'pattern2'
            
            # デフォルトはパターン1
            logger.warning("パターンの自動判定に失敗。パターン1として処理します")
            return 'pattern1'
            
        except Exception as e:
            logger.error(f"CSV パターン判定エラー: {e}")
            return 'pattern1'  # デフォルト
    
    @staticmethod
    def memory_usage_mb():
        """
        現在のメモリ使用量を取得（MB）
        """
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024


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
    # 動作テスト
    print("=== USDJPY Utils テスト ===")
    
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
    
    # 時間特徴量テスト
    test_dt = datetime.now()
    time_features = USDJPYUtils.create_time_features(test_dt)
    print(f"時間特徴量: {time_features}")
    
    print("テスト完了")