import pandas as pd

def load_tick_data(filepath: str) -> pd.DataFrame:
    """
    タブ区切りのティックデータCSV（パターン2）を読み込み、
    MID価格とtimestamp列を追加して返す。

    Parameters:
        filepath (str): CSVファイルパス（例: 'data/usdjpy_ticks.csv'）

    Returns:
        pd.DataFrame: 整形されたDataFrame（timestamp, bid, ask, mid）
    """
    # データ読み込み（タブ区切り）
    df = pd.read_csv(filepath, sep='\t', usecols=['<DATE>', '<TIME>', '<BID>', '<ASK>'])

    # timestamp列作成（datetime型に変換）
    df['timestamp'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])

    # カラム名統一
    df.rename(columns={
        '<BID>': 'bid',
        '<ASK>': 'ask'
    }, inplace=True)

    # MID価格列を追加
    df['mid'] = (df['bid'] + df['ask']) / 2

    # 並び順を整える
    df = df[['timestamp', 'bid', 'ask', 'mid']]

    return df
