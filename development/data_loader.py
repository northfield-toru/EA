"""
修正版データローダー（エンコーディング・BOM対応）
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import os
from typing import Iterator, Tuple, Optional
import gc

from utils import USDJPYUtils, DataValidationError

# ロガー設定
logger = logging.getLogger(__name__)

class TickDataLoader:
    """ティックデータローダー（修正版）"""
    
    def __init__(self, chunk_size=1000000):
        self.chunk_size = chunk_size
        self.utils = USDJPYUtils()
    
    def detect_csv_encoding_and_columns(self, filepath: str):
        """CSVファイルのエンコーディングとカラムを検出"""
        encodings = ['utf-8', 'utf-8-sig', 'cp932', 'shift_jis', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    first_line = f.readline().strip()
                    print(f"🔍 エンコーディング {encoding} で確認:")
                    print(f"   ヘッダー: {repr(first_line)}")
                    
                    # タブ区切りかカンマ区切りか判定
                    tab_count = first_line.count('\t')
                    comma_count = first_line.count(',')
                    
                    print(f"   タブ数: {tab_count}, カンマ数: {comma_count}")
                    
                    if tab_count > 0:
                        # タブ区切りの場合、カラム名を取得
                        columns = [col.strip() for col in first_line.split('\t')]
                        print(f"   カラム: {columns}")
                        
                        # 期待されるカラムが含まれているかチェック
                        expected_cols = ['<DATE>', '<TIME>', '<BID>', '<ASK>']
                        if all(col in columns for col in expected_cols):
                            print(f"✅ エンコーディング {encoding} で正常読み込み可能")
                            return encoding, 'pattern2', columns
                    
                    elif comma_count >= 2:
                        print(f"✅ エンコーディング {encoding} でパターン1判定")
                        return encoding, 'pattern1', None
                        
            except Exception as e:
                print(f"❌ エンコーディング {encoding} 失敗: {e}")
                continue
        
        # デフォルト
        print("⚠️ 自動検出失敗、utf-8・pattern2をデフォルト使用")
        return 'utf-8', 'pattern2', ['<DATE>', '<TIME>', '<BID>', '<ASK>', '<LAST>', '<VOLUME>', '<FLAGS>']
    
    def load_tick_data_pattern2_safe(self, filepath: str, start_row: int = 0, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        安全なパターン2データ読み込み（タブ区切り専用）
        """
        print(f"🔍 Pattern2データ読み込み（タブ区切り）: {filepath}")
        
        try:
            # シンプルなタブ区切り読み込み
            skip_rows = start_row + 1 if start_row == 0 else start_row
            
            # まず全カラムで読み込み
            df = pd.read_csv(
                filepath,
                sep='\t',
                skiprows=skip_rows if start_row > 0 else 0,
                nrows=nrows,
                engine='python',  # より柔軟
                encoding='utf-8-sig'  # BOM対応
            )
            
            print(f"✅ 読み込み成功: {len(df)} 行")
            print(f"📊 検出カラム: {list(df.columns)}")
            
            # 必要カラムのマッピング
            required_mapping = {
                '<DATE>': 'date_str',
                '<TIME>': 'time_str',
                '<BID>': 'bid',
                '<ASK>': 'ask'
            }
            
            # カラム存在確認
            missing_cols = []
            for req_col in required_mapping.keys():
                if req_col not in df.columns:
                    missing_cols.append(req_col)
            
            if missing_cols:
                raise ValueError(f"必要カラムが見つかりません: {missing_cols}")
            
            # 必要カラムのみ選択・リネーム
            selected_df = df[list(required_mapping.keys())].copy()
            selected_df = selected_df.rename(columns=required_mapping)
            
            # データ型変換
            selected_df['bid'] = pd.to_numeric(selected_df['bid'], errors='coerce')
            selected_df['ask'] = pd.to_numeric(selected_df['ask'], errors='coerce')
            
            print(f"✅ カラム選択完了: {len(selected_df)} 行")
            
            # タイムスタンプ解析（修正版utils使用）
            print("🔧 タイムスタンプ解析中...")
            selected_df['timestamp'] = selected_df.apply(
                lambda row: self.utils.parse_timestamp_pattern2_flexible(
                    row['date_str'], row['time_str']
                ), 
                axis=1
            )
            
            # 解析失敗した行を確認
            invalid_mask = selected_df['timestamp'].isna()
            invalid_count = invalid_mask.sum()
            
            if invalid_count > 0:
                print(f"⚠️ 無効なタイムスタンプ {invalid_count} 行を除去")
                # サンプル表示
                if invalid_count <= 5:
                    print("無効な行のサンプル:")
                    invalid_samples = selected_df[invalid_mask][['date_str', 'time_str']].head()
                    for idx, row in invalid_samples.iterrows():
                        print(f"  {row['date_str']} | {row['time_str']}")
                
                selected_df = selected_df[~invalid_mask].copy()
            
            if len(selected_df) == 0:
                raise ValueError("全てのタイムスタンプ解析に失敗しました")
            
            # 不要列削除
            selected_df = selected_df.drop(['date_str', 'time_str'], axis=1)
            
            # データ妥当性チェック（警告のみ）
            is_valid, errors = self.utils.validate_price_data(selected_df)
            if not is_valid:
                print(f"⚠️ データ検証警告: {errors}")
            
            # MID価格計算
            selected_df['mid'] = self.utils.calculate_mid_price(selected_df['bid'], selected_df['ask'])
            
            # タイムスタンプでソート
            selected_df = selected_df.sort_values('timestamp').reset_index(drop=True)
            
            print(f"✅ Pattern2処理完了: {len(selected_df)} 行")
            return selected_df
            
        except Exception as e:
            print(f"❌ Pattern2データ読み込みエラー: {e}")
            raise
    
    def load_tick_data_auto_safe(self, filepath: str, start_row: int = 0, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        安全な自動判定データ読み込み
        """
        print(f"🔍 安全な自動判定読み込み: {filepath}")
        
        # まずエンコーディングとパターンを検出
        encoding, pattern, columns = self.detect_csv_encoding_and_columns(filepath)
        
        try:
            if pattern == 'pattern1':
                return self.load_tick_data_pattern1(filepath, start_row, nrows)
            else:
                return self.load_tick_data_pattern2_safe(filepath, start_row, nrows)
        except Exception as e:
            print(f"❌ 自動判定失敗: {e}")
            print("🔄 フォールバック: パターン1を試行...")
            try:
                return self.load_tick_data_pattern1(filepath, start_row, nrows)
            except Exception as e2:
                print(f"❌ パターン1も失敗: {e2}")
                raise
    
    # 既存のメソッドはそのまま保持
    def load_tick_data_pattern1(self, filepath: str, start_row: int = 0, nrows: Optional[int] = None) -> pd.DataFrame:
        """パターン1の読み込み（既存のまま）"""
        logger.info(f"パターン1データ読み込み開始: {filepath}")
        
        try:
            df = pd.read_csv(
                filepath,
                header=None,
                names=['timestamp_str', 'bid', 'ask'],
                skiprows=start_row,
                nrows=nrows,
                dtype={'bid': 'float64', 'ask': 'float64'},
                engine='c'
            )
            
            logger.info(f"読み込み完了: {len(df):,} 行")
            
            # タイムスタンプ解析
            logger.info("タイムスタンプ解析中...")
            df['timestamp'] = df['timestamp_str'].apply(self.utils.parse_timestamp_pattern1)
            
            # 解析失敗した行を除去
            invalid_mask = df['timestamp'].isna()
            if invalid_mask.sum() > 0:
                logger.warning(f"無効なタイムスタンプ {invalid_mask.sum()} 行を除去")
                df = df[~invalid_mask].copy()
            
            # 不要列削除
            df = df.drop('timestamp_str', axis=1)
            
            # データ妥当性チェック
            is_valid, errors = self.utils.validate_price_data(df)
            if not is_valid:
                print(f"⚠️ データ検証警告: {errors}")
            
            # MID価格計算
            df['mid'] = self.utils.calculate_mid_price(df['bid'], df['ask'])
            
            # タイムスタンプでソート
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"処理完了: {len(df):,} 行")
            return df
            
        except Exception as e:
            logger.error(f"パターン1データ読み込みエラー: {e}")
            raise
    
    def tick_to_ohlcv_1min(self, tick_df: pd.DataFrame) -> pd.DataFrame:
        """既存のOHLCV変換（そのまま保持）"""
        logger.info("1分足OHLCV変換開始...")
        
        if len(tick_df) == 0:
            logger.warning("空のデータセット")
            return pd.DataFrame()
        
        # 1分間隔でリサンプル（MID価格基準）
        tick_df_indexed = tick_df.set_index('timestamp')
        
        # OHLCV計算
        ohlcv = tick_df_indexed['mid'].resample('1min').agg({
            'open': 'first',
            'high': 'max', 
            'low': 'min',
            'close': 'last'
        }).dropna()
        
        # ティック数をVolumeとして使用
        volume = tick_df_indexed.resample('1min').size().reindex(ohlcv.index, fill_value=0)
        ohlcv['volume'] = volume
        
        # Bid/Ask情報も保持（スプレッド分析用）
        bid_info = tick_df_indexed['bid'].resample('1min').agg({
            'bid_open': 'first',
            'bid_close': 'last'
        }).reindex(ohlcv.index, method='ffill')
        
        ask_info = tick_df_indexed['ask'].resample('1min').agg({
            'ask_open': 'first', 
            'ask_close': 'last'
        }).reindex(ohlcv.index, method='ffill')
        
        # 統合
        ohlcv = pd.concat([ohlcv, bid_info, ask_info], axis=1)

        # 時間特徴量追加
        ohlcv['hour'] = ohlcv.index.hour
        ohlcv['minute'] = ohlcv.index.minute
        ohlcv['weekday'] = ohlcv.index.weekday
        ohlcv['is_weekend'] = ohlcv.index.weekday >= 5
        
        # 市場セッション
        ohlcv['market_session'] = ohlcv.index.to_series().apply(self.utils.get_market_session)
        
        logger.info(f"1分足変換完了: {len(ohlcv):,} 本")
        return ohlcv


def load_sample_data(filepath: str, sample_size: int = 100000) -> pd.DataFrame:
    """
    修正版サンプルデータ読み込み（安全版）
    """
    loader = TickDataLoader()
    
    print(f"🔍 修正版サンプルデータ読み込み: {sample_size:,} 行")
    tick_df = loader.load_tick_data_auto_safe(filepath, nrows=sample_size)
    
    print("🔧 1分足変換...")
    ohlcv_df = loader.tick_to_ohlcv_1min(tick_df)
    
    return ohlcv_df