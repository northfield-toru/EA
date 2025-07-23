import pandas as pd
import numpy as np
from typing import Dict, Any, Iterator, Tuple, List
import logging
import os
from datetime import datetime
from .utils import validate_data_integrity, parse_datetime, memory_usage_mb

logger = logging.getLogger(__name__)

class TickDataLoader:
    """
    ティックデータの効率的な読み込み・前処理
    メモリ効率を最優先に設計
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_config = config['data']
        self.trading_config = config['trading']
        
        self.chunk_size = self.data_config['chunk_size']
        self.use_columns = self.data_config['use_columns']
        
    def load_tick_data(self, file_path: str = None, validate: bool = True) -> pd.DataFrame:
        """
        ティックデータを一括読み込み
        """
        if file_path is None:
            file_path = self.data_config['input_file']
        
        logger.info(f"ティックデータ読み込み開始: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"データファイルが見つかりません: {file_path}")
        
        try:
            # CSVファイルのサイズ確認
            file_size_mb = os.path.getsize(file_path) / 1024 / 1024
            logger.info(f"ファイルサイズ: {file_size_mb:.1f}MB")
            
            # ファイル形式の判定
            separator = self._detect_separator(file_path)
            
            # まず先頭行を読んでカラム名を確認
            sample_df = pd.read_csv(file_path, sep=separator, nrows=1)
            actual_columns = sample_df.columns.tolist()
            logger.info(f"実際のカラム名: {actual_columns}")
            
            # カラム名のマッピングを作成
            column_mapping = self._create_column_mapping(actual_columns)
            logger.info(f"カラムマッピング: {column_mapping}")
            
            # 使用するカラムを実際のカラム名に変換
            actual_use_columns = [column_mapping.get(col, col) for col in self.use_columns]
            existing_columns = [col for col in actual_use_columns if col in actual_columns]
            
            if len(existing_columns) < 4:
                logger.warning(f"必要なカラムが不足しています。使用可能: {existing_columns}")
                # 最低限BIDとASKがあれば処理を続行
                if not any('bid' in col.lower() for col in actual_columns) or not any('ask' in col.lower() for col in actual_columns):
                    raise ValueError("BIDまたはASKカラムが見つかりません")
            
            # データ読み込み
            dtype_dict = {}
            for col in existing_columns:
                if any(price_col in col.lower() for price_col in ['bid', 'ask', 'price']):
                    dtype_dict[col] = 'float64'
                else:
                    dtype_dict[col] = 'str'
            
            df = pd.read_csv(
                file_path,
                sep=separator,
                usecols=existing_columns,
                dtype=dtype_dict,
                engine='c',  # 高速化
                low_memory=False
            )
            
            # カラム名を標準形式に統一
            df = self._standardize_column_names(df, column_mapping)
            
            logger.info(f"データ読み込み完了: {len(df):,} レコード")
            logger.info(f"使用カラム: {df.columns.tolist()}")
            logger.info(f"メモリ使用量: {memory_usage_mb():.1f}MB")
            
            # データ検証
            if validate:
                validate_data_integrity(df, self.config)
            
            # 前処理
            df = self._preprocess_data(df)
            
            return df
            
        except Exception as e:
            logger.error(f"データ読み込みエラー: {e}")
            raise
    
    def _create_column_mapping(self, actual_columns: list) -> dict:
        """
        実際のカラム名から標準カラム名へのマッピングを作成
        """
        mapping = {}
        
        # 各カラムについて最適なマッチングを探す
        for target_col in self.use_columns:
            best_match = None
            
            # 完全一致を最優先
            if target_col in actual_columns:
                best_match = target_col
            else:
                # 部分一致・大小文字無視で検索
                target_lower = target_col.lower()
                for actual_col in actual_columns:
                    actual_lower = actual_col.lower()
                    
                    if target_lower == actual_lower:
                        best_match = actual_col
                        break
                    elif target_lower in actual_lower or actual_lower in target_lower:
                        # より具体的なマッチング
                        if target_col == 'DATE' and any(keyword in actual_lower for keyword in ['date', 'dt']):
                            best_match = actual_col
                        elif target_col == 'TIME' and any(keyword in actual_lower for keyword in ['time', 'tm']):
                            best_match = actual_col
                        elif target_col == 'BID' and 'bid' in actual_lower:
                            best_match = actual_col
                        elif target_col == 'ASK' and 'ask' in actual_lower:
                            best_match = actual_col
            
            if best_match:
                mapping[target_col] = best_match
            else:
                # デフォルトの推測
                if target_col == 'DATE' and len(actual_columns) > 0:
                    mapping[target_col] = actual_columns[0]  # 通常最初のカラム
                elif target_col == 'TIME' and len(actual_columns) > 1:
                    mapping[target_col] = actual_columns[1]  # 通常2番目のカラム
                else:
                    logger.warning(f"カラム {target_col} に対応する実際のカラムが見つかりません")
        
        return mapping
    
    def _standardize_column_names(self, df: pd.DataFrame, column_mapping: dict) -> pd.DataFrame:
        """
        カラム名を標準形式に統一
        """
        # 逆マッピングを作成
        reverse_mapping = {v: k for k, v in column_mapping.items()}
        
        # カラム名を変更
        df = df.rename(columns=reverse_mapping)
        
        # 不足しているカラムの補完
        required_columns = ['DATE', 'TIME', 'BID', 'ASK']
        for col in required_columns:
            if col not in df.columns:
                if col == 'DATE' and 'TIME' in df.columns:
                    # TIMEカラムに日付も含まれている可能性
                    df['DATE'] = df['TIME'].str.split(' ').str[0] if ' ' in df['TIME'].iloc[0] else '2025.01.01'
                elif col == 'TIME' and 'DATE' in df.columns:
                    # DATEカラムに時刻も含まれている可能性
                    df['TIME'] = df['DATE'].str.split(' ').str[1] if ' ' in df['DATE'].iloc[0] else '00:00:00'
                else:
                    logger.warning(f"必須カラム {col} が見つかりません。ダミー値で補完します。")
                    if col == 'DATE':
                        df['DATE'] = '2025.01.01'
                    elif col == 'TIME':
                        df['TIME'] = df.index.astype(str) + '.000'  # インデックスベースの時刻
                    elif col in ['BID', 'ASK']:
                        # 他の価格カラムから推定
                        price_cols = [c for c in df.columns if any(p in c.lower() for p in ['price', 'close', 'last'])]
                        if price_cols:
                            base_price = df[price_cols[0]]
                            if col == 'BID':
                                df['BID'] = base_price - 0.0005  # 仮のスプレッド
                            else:  # ASK
                                df['ASK'] = base_price + 0.0005
                        else:
                            df[col] = 157.0  # USDJPYのデフォルト値
        
        return df
    
    def load_tick_data_chunked(self, file_path: str = None) -> Iterator[pd.DataFrame]:
        """
        チャンク単位でティックデータを読み込み（メモリ効率重視）
        """
        if file_path is None:
            file_path = self.data_config['input_file']
        
        logger.info(f"チャンク読み込み開始: {file_path}")
        logger.info(f"チャンクサイズ: {self.chunk_size:,} レコード")
        
        separator = self._detect_separator(file_path)
        
        chunk_reader = pd.read_csv(
            file_path,
            sep=separator,
            usecols=self.use_columns,
            dtype={
                'BID': 'float64',
                'ASK': 'float64',
                'DATE': 'str',
                'TIME': 'str'
            },
            chunksize=self.chunk_size,
            engine='c',
            low_memory=True
        )
        
        chunk_count = 0
        total_records = 0
        
        for chunk in chunk_reader:
            chunk_count += 1
            total_records += len(chunk)
            
            if chunk_count % 10 == 0:
                logger.info(f"チャンク {chunk_count} 処理中 - "
                           f"累計レコード数: {total_records:,}")
            
            # チャンク前処理
            chunk = self._preprocess_data(chunk)
            
            yield chunk
        
        logger.info(f"チャンク読み込み完了: {chunk_count} チャンク, "
                   f"総レコード数: {total_records:,}")
    
    def _detect_separator(self, file_path: str) -> str:
        """
        CSVファイルの区切り文字を自動検出
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
        
        # タブ区切りかカンマ区切りかを判定
        if '\t' in first_line:
            return '\t'
        elif ',' in first_line:
            return ','
        else:
            logger.warning("区切り文字が不明です。タブ区切りとして処理します。")
            return '\t'
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        データの前処理
        """
        # 日時カラム統合
        df['datetime'] = df.apply(
            lambda row: parse_datetime(row['DATE'], row['TIME']), 
            axis=1
        )
        
        # 時系列ソート（必須）
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # 異常値検出・除去
        df = self._remove_outliers(df)
        
        # 重複データ除去
        initial_count = len(df)
        df = df.drop_duplicates(subset=['datetime'], keep='first')
        removed_count = initial_count - len(df)
        
        if removed_count > 0:
            logger.info(f"重複データ除去: {removed_count:,} レコード")
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        異常値の検出・除去
        """
        initial_count = len(df)
        
        # 価格異常値の除去
        # BID/ASKが0以下
        valid_price_mask = (df['BID'] > 0) & (df['ASK'] > 0)
        
        # スプレッドが異常に大きい（1000pips以上）
        spread_pips = (df['ASK'] - df['BID']) / self.trading_config['pip_value']
        valid_spread_mask = spread_pips < 1000
        
        # ASK < BIDの異常ケース
        valid_order_mask = df['ASK'] >= df['BID']
        
        # 価格変動が異常に大きい（前ティックから1000pips以上変動）
        price_change = df['BID'].diff().abs() / self.trading_config['pip_value']
        valid_change_mask = (price_change < 1000) | (price_change.isna())
        
        # 全条件を満たすデータのみ残す
        valid_mask = valid_price_mask & valid_spread_mask & valid_order_mask & valid_change_mask
        df_clean = df[valid_mask].copy()
        
        removed_count = initial_count - len(df_clean)
        if removed_count > 0:
            logger.info(f"異常値除去: {removed_count:,} レコード ({removed_count/initial_count*100:.2f}%)")
        
        return df_clean
    
    def get_data_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        データ統計情報の取得
        """
        stats = {
            'total_records': len(df),
            'date_range': {
                'start': df['datetime'].min().isoformat(),
                'end': df['datetime'].max().isoformat(),
                'duration_hours': (df['datetime'].max() - df['datetime'].min()).total_seconds() / 3600
            },
            'price_statistics': {
                'bid_min': float(df['BID'].min()),
                'bid_max': float(df['BID'].max()),
                'bid_mean': float(df['BID'].mean()),
                'ask_min': float(df['ASK'].min()),
                'ask_max': float(df['ASK'].max()),
                'ask_mean': float(df['ASK'].mean()),
            },
            'spread_statistics': {
                'spread_pips_min': float((df['ASK'] - df['BID']).min() / self.trading_config['pip_value']),
                'spread_pips_max': float((df['ASK'] - df['BID']).max() / self.trading_config['pip_value']),
                'spread_pips_mean': float((df['ASK'] - df['BID']).mean() / self.trading_config['pip_value']),
                'spread_pips_std': float((df['ASK'] - df['BID']).std() / self.trading_config['pip_value'])
            },
            'time_statistics': {
                'avg_tick_interval_seconds': (df['datetime'].diff().dt.total_seconds().mean()),
                'ticks_per_hour': len(df) / ((df['datetime'].max() - df['datetime'].min()).total_seconds() / 3600),
                'trading_hours_coverage': len(df['datetime'].dt.hour.unique())
            }
        }
        
        return stats
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str, include_labels: bool = True):
        """
        前処理済みデータの保存
        """
        logger.info(f"処理済みデータ保存開始: {output_path}")
        
        # 保存するカラムを選択
        save_columns = ['datetime', 'BID', 'ASK', 'mid_price']
        
        if include_labels and 'label' in df.columns:
            save_columns.append('label')
        
        # 特徴量カラムも含める場合
        feature_columns = [col for col in df.columns if col.startswith(('sma_', 'ema_', 'rsi', 'macd_', 'bb_', 'atr', 'cci'))]
        save_columns.extend(feature_columns)
        
        # 存在するカラムのみ保存
        available_columns = [col for col in save_columns if col in df.columns]
        
        # 保存実行
        df[available_columns].to_csv(
            output_path,
            index=False,
            float_format='%.6f'
        )
        
        file_size_mb = os.path.getsize(output_path) / 1024 / 1024
        logger.info(f"保存完了: {output_path} ({file_size_mb:.1f}MB)")
    
    def create_sample_dataset(self, df: pd.DataFrame, sample_ratio: float = 0.1) -> pd.DataFrame:
        """
        サンプルデータセット作成（開発・テスト用）
        """
        logger.info(f"サンプルデータセット作成: {sample_ratio*100:.1f}%")
        
        # 時系列順序を保持してサンプリング
        sample_size = int(len(df) * sample_ratio)
        
        # 均等間隔でサンプリング
        step_size = len(df) // sample_size
        sample_indices = np.arange(0, len(df), step_size)[:sample_size]
        
        sample_df = df.iloc[sample_indices].copy().reset_index(drop=True)
        
        logger.info(f"サンプル作成完了: {len(sample_df):,} レコード")
        return sample_df
    
    def split_data_by_date(self, df: pd.DataFrame, split_dates: List[str]) -> List[pd.DataFrame]:
        """
        日付でデータを分割
        """
        split_dates = [pd.to_datetime(date) for date in split_dates]
        split_dates.sort()
        
        data_splits = []
        start_date = df['datetime'].min()
        
        for split_date in split_dates:
            split_df = df[(df['datetime'] >= start_date) & (df['datetime'] < split_date)].copy()
            if len(split_df) > 0:
                data_splits.append(split_df)
            start_date = split_date
        
        # 最後の期間
        final_split = df[df['datetime'] >= start_date].copy()
        if len(final_split) > 0:
            data_splits.append(final_split)
        
        logger.info(f"データ分割完了: {len(data_splits)} 期間")
        for i, split in enumerate(data_splits):
            logger.info(f"  期間{i+1}: {len(split):,} レコード "
                       f"({split['datetime'].min()} - {split['datetime'].max()})")
        
        return data_splits