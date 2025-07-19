"""
USDJPY スキャルピングEA用 データローダー
2つのCSVパターンに対応し、ティック→1分足OHLCV変換を行う
メモリ効率を重視したチャンク処理対応
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
from typing import Iterator, Tuple, Optional
import gc

from utils import USDJPYUtils, DataValidationError

logger = logging.getLogger(__name__)

class TickDataLoader:
    """ティックデータローダー"""
    
    def __init__(self, chunk_size=1000000):
        """
        Args:
            chunk_size (int): チャンクサイズ（行数）
        """
        self.chunk_size = chunk_size
        self.utils = USDJPYUtils()
        
    def load_tick_data_pattern1(self, filepath: str, start_row: int = 0, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        パターン1のCSVを読み込み（カンマ区切り・ヘッダーなし）
        Format: timestamp,bid,ask
        
        Args:
            filepath: CSVファイルパス
            start_row: 開始行
            nrows: 読み込み行数（Noneなら全行）
        Returns:
            DataFrame: ティックデータ
        """
        logger.info(f"パターン1データ読み込み開始: {filepath}")
        
        try:
            df = pd.read_csv(
                filepath,
                header=None,
                names=['timestamp_str', 'bid', 'ask'],
                skiprows=start_row,
                nrows=nrows,
                dtype={'bid': 'float64', 'ask': 'float64'},
                engine='c'  # 高速化
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
                raise DataValidationError(f"データ検証エラー: {errors}")
            
            # MID価格計算
            df['mid'] = self.utils.calculate_mid_price(df['bid'], df['ask'])
            
            # タイムスタンプでソート
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"処理完了: {len(df):,} 行")
            return df
            
        except Exception as e:
            logger.error(f"パターン1データ読み込みエラー: {e}")
            raise
    
    def load_tick_data_pattern2(self, filepath: str, start_row: int = 0, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        パターン2のCSVを読み込み（タブ区切り・ヘッダーあり）
        Format: <DATE>	<TIME>	<BID>	<ASK>	<LAST>	<VOLUME>	<FLAGS>
        
        Args:
            filepath: CSVファイルパス
            start_row: 開始行（ヘッダーを除く）
            nrows: 読み込み行数（Noneなら全行）
        Returns:
            DataFrame: ティックデータ
        """
        logger.info(f"パターン2データ読み込み開始: {filepath}")
        
        try:
            # ヘッダー行をスキップする場合の調整
            skip_rows = start_row + 1 if start_row == 0 else start_row
            
            df = pd.read_csv(
                filepath,
                sep='\t',
                skiprows=skip_rows if start_row > 0 else 0,
                nrows=nrows,
                usecols=['<DATE>', '<TIME>', '<BID>', '<ASK>'],
                dtype={'<BID>': 'float64', '<ASK>': 'float64'},
                engine='c'
            )
            
            # カラム名を標準化
            df = df.rename(columns={
                '<DATE>': 'date_str',
                '<TIME>': 'time_str', 
                '<BID>': 'bid',
                '<ASK>': 'ask'
            })
            
            logger.info(f"読み込み完了: {len(df):,} 行")
            
            # タイムスタンプ解析
            logger.info("タイムスタンプ解析中...")
            df['timestamp'] = df.apply(
                lambda row: self.utils.parse_timestamp_pattern2(row['date_str'], row['time_str']), 
                axis=1
            )
            
            # 解析失敗した行を除去
            invalid_mask = df['timestamp'].isna()
            if invalid_mask.sum() > 0:
                logger.warning(f"無効なタイムスタンプ {invalid_mask.sum()} 行を除去")
                df = df[~invalid_mask].copy()
            
            # 不要列削除
            df = df.drop(['date_str', 'time_str'], axis=1)
            
            # データ妥当性チェック
            is_valid, errors = self.utils.validate_price_data(df)
            if not is_valid:
                raise DataValidationError(f"データ検証エラー: {errors}")
            
            # MID価格計算
            df['mid'] = self.utils.calculate_mid_price(df['bid'], df['ask'])
            
            # タイムスタンプでソート
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"処理完了: {len(df):,} 行")
            return df
            
        except Exception as e:
            logger.error(f"パターン2データ読み込みエラー: {e}")
            raise
    
    def load_tick_data_auto(self, filepath: str, start_row: int = 0, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        CSVパターンを自動判定してデータを読み込み
        
        Args:
            filepath: CSVファイルパス
            start_row: 開始行
            nrows: 読み込み行数
        Returns:
            DataFrame: ティックデータ
        """
        pattern = self.utils.detect_csv_pattern(filepath)
        
        if pattern == 'pattern1':
            return self.load_tick_data_pattern1(filepath, start_row, nrows)
        else:
            return self.load_tick_data_pattern2(filepath, start_row, nrows)
    
    def tick_to_ohlcv_1min(self, tick_df: pd.DataFrame) -> pd.DataFrame:
        """
        ティックデータから1分足OHLCVを生成
        
        Args:
            tick_df: ティックデータ（timestamp, bid, ask, mid列必須）
        Returns:
            DataFrame: 1分足OHLCV
        """
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
    
    def process_large_file_chunks(self, filepath: str, output_dir: str = "processed_data") -> Iterator[str]:
        """
        大容量ファイルをチャンク処理し、中間ファイルを生成
        
        Args:
            filepath: 入力CSVファイル
            output_dir: 出力ディレクトリ
        Yields:
            str: 処理済み中間ファイルパス
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # ファイルサイズとパターン判定
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        logger.info(f"ファイルサイズ: {file_size:.1f} MB")
        
        pattern = self.utils.detect_csv_pattern(filepath)
        
        # チャンクサイズの動的調整
        if file_size > 1000:  # 1GB以上
            chunk_size = self.chunk_size // 2
        else:
            chunk_size = self.chunk_size
        
        logger.info(f"チャンクサイズ: {chunk_size:,} 行")
        
        chunk_num = 0
        start_row = 0
        
        while True:
            try:
                # メモリ使用量チェック
                memory_mb = self.utils.memory_usage_mb()
                logger.info(f"メモリ使用量: {memory_mb:.1f} MB")
                
                # チャンク読み込み
                if pattern == 'pattern1':
                    chunk_df = self.load_tick_data_pattern1(filepath, start_row, chunk_size)
                else:
                    chunk_df = self.load_tick_data_pattern2(filepath, start_row, chunk_size)
                
                if len(chunk_df) == 0:
                    logger.info("全データ処理完了")
                    break
                
                # 1分足変換
                ohlcv_df = self.tick_to_ohlcv_1min(chunk_df)
                
                if len(ohlcv_df) > 0:
                    # HDF5形式で保存（高速・圧縮）
                    output_file = os.path.join(output_dir, f"ohlcv_chunk_{chunk_num:04d}.h5")
                    ohlcv_df.to_hdf(output_file, key='ohlcv', mode='w', format='table', complevel=9)
                    
                    logger.info(f"チャンク {chunk_num} 保存完了: {output_file} ({len(ohlcv_df)} 本)")
                    yield output_file
                
                chunk_num += 1
                start_row += chunk_size
                
                # メモリクリア
                del chunk_df, ohlcv_df
                gc.collect()
                
            except Exception as e:
                logger.error(f"チャンク {chunk_num} 処理エラー: {e}")
                break
    
    def merge_chunk_files(self, chunk_files: list, output_file: str) -> pd.DataFrame:
        """
        チャンクファイルを統合
        
        Args:
            chunk_files: チャンクファイルパスのリスト
            output_file: 統合後のファイルパス
        Returns:
            DataFrame: 統合されたデータ
        """
        logger.info(f"チャンクファイル統合開始: {len(chunk_files)} ファイル")
        
        all_data = []
        
        for chunk_file in chunk_files:
            try:
                chunk_df = pd.read_hdf(chunk_file, key='ohlcv')
                all_data.append(chunk_df)
                logger.info(f"読み込み完了: {chunk_file} ({len(chunk_df)} 本)")
            except Exception as e:
                logger.error(f"チャンクファイル読み込みエラー: {chunk_file} - {e}")
        
        if not all_data:
            raise ValueError("統合可能なデータがありません")
        
        # 統合
        merged_df = pd.concat(all_data, axis=0).sort_index()
        
        # 重複削除
        merged_df = merged_df[~merged_df.index.duplicated(keep='first')]
        
        # 保存
        merged_df.to_hdf(output_file, key='ohlcv', mode='w', format='table', complevel=9)
        
        logger.info(f"統合完了: {output_file} ({len(merged_df)} 本)")
        return merged_df


def load_sample_data(filepath: str, sample_size: int = 100000) -> pd.DataFrame:
    """
    サンプルデータを読み込み（開発・テスト用）
    
    Args:
        filepath: CSVファイルパス
        sample_size: サンプルサイズ
    Returns:
        DataFrame: サンプルデータ
    """
    loader = TickDataLoader()
    
    logger.info(f"サンプルデータ読み込み: {sample_size:,} 行")
    tick_df = loader.load_tick_data_auto(filepath, nrows=sample_size)
    
    logger.info("1分足変換...")
    ohlcv_df = loader.tick_to_ohlcv_1min(tick_df)
    
    return ohlcv_df


if __name__ == "__main__":
    # テスト実行
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法: python data_loader.py <csv_file_path> [sample_size]")
        sys.exit(1)
    
    filepath = sys.argv[1]
    sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    
    print("=== データローダーテスト ===")
    
    try:
        # サンプルデータ読み込みテスト
        ohlcv_df = load_sample_data(filepath, sample_size)
        
        print(f"読み込み完了: {len(ohlcv_df)} 本")
        print("\nデータサンプル:")
        print(ohlcv_df.head())
        print("\n基本統計:")
        print(ohlcv_df.describe())
        
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()