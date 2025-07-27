# USDJPYスキャルピング向けAI学習モデル 2025年版

実用性重視の堅牢なUSDJPYスキャルピング自動売買AI学習システム

## 📋 概要

本プロジェクトは、USDJPYのティックデータを使用してスキャルピング取引に最適な機械学習モデルを構築するシステムです。未来リーク防止、クラス重み自動調整、詳細なログ出力など、実運用を想定した設計になっています。

### 主な特徴

- **3クラス分類**: BUY/SELL/NO_TRADE の判定
- **未来リーク完全防止**: 特徴量計算時の厳格な時系列制約
- **設定一元管理**: config.json での全パラメータ管理
- **複数モデル対応**: Conv1D, LSTM, Hybrid, Transformer
- **自動クラス重み調整**: 不均衡データへの対応
- **詳細ログ出力**: 分析・デバッグ用の豊富な情報
- **GPU最適化**: TensorFlow-GPU対応

## 🚀 クイックスタート

### 1. 環境セットアップ

```bash
# Conda環境作成・アクティベート
conda create -n forex_ai python=3.8.20
conda activate forex_ai

# 必要パッケージインストール
pip install tensorflow-gpu==2.10.0
pip install numpy==1.24.4 pandas==2.0.3 scikit-learn==1.3.2
pip install matplotlib==3.7.5 seaborn==0.13.2 psutil==7.0.0
```

### 2. データ準備

ティックデータを以下の形式で `data/usdjpy_ticks.csv` に配置：

```
<DATE>	<TIME>	<BID>	<ASK>	<LAST>	<VOLUME>	<FLAGS>
20250127	000001	149.123	149.125	149.124	100	0
20250127	000002	149.124	149.126	149.125	150	0
...
```

### 3. 設定調整

`config.json` で取引パラメータを調整：

```json
{
  "trading": {
    "tp_pips": 4.0,
    "sl_pips": 5.0,
    "spread_pips": 0.7,
    "trade_threshold": 0.5
  },
  "data": {
    "feature_window": 64,
    "future_window": 100,
    "sample_rate": 1
  },
  "model": {
    "type": "Hybrid",
    "epochs": 100,
    "batch_size": 512
  }
}
```

### 4. 訓練実行

```bash
python main.py
```

### 5. 推論実行

```bash
# 最新データの単一予測
python predict.py --model models/forex_model_20250127_1430.h5 --single --signal

# バッチ予測
python predict.py --model models/forex_model_20250127_1430.h5 --output results.csv
```

## 📁 ディレクトリ構成

```
project_root/
├── config.json                  # 設定ファイル
├── main.py                      # 訓練メインスクリプト
├── predict.py                   # 推論スクリプト
├── README.md                    # このファイル
│
├── data/
│   └── usdjpy_ticks.csv         # ティックデータ
│
├── models/
│   └── *.h5                     # 学習済みモデル
│
├── logs/
│   ├── model_summary.csv        # 全モデルの性能サマリ
│   └── YYYYMMDD_HHMM/           # 各訓練セッションのログ
│       ├── training.log         # 訓練ログ
│       ├── label_distribution.* # ラベル分布
│       ├── threshold_analysis.* # 閾値分析
│       ├── confusion_matrix.png # 混同行列
│       ├── training_history.*   # 訓練履歴
│       ├── class_weights.json   # クラス重み
│       ├── normalization_params.json  # 正規化パラメータ
│       └── config.json          # 使用した設定
│
└── modules/
    ├── feature_engineering.py   # 特徴量エンジニアリング
    ├── labeling.py              # ラベル生成
    ├── model.py                 # モデル定義
    ├── train.py                 # 訓練ロジック
    └── utils.py                 # ユーティリティ関数
```

## ⚙️ 設定詳細

### 取引設定 (trading)

| パラメータ | 説明 | デフォルト |
|-----------|------|----------|
| `tp_pips` | テイクプロフィット (pips) | 4.0 |
| `sl_pips` | ストップロス (pips) | 5.0 |
| `spread_pips` | 固定スプレッド (pips) | 0.7 |
| `trade_threshold` | 取引信頼度閾値 | 0.5 |

### データ設定 (data)

| パラメータ | 説明 | デフォルト |
|-----------|------|----------|
| `feature_window` | 特徴量時系列長 | 64 |
| `future_window` | 未来参照範囲 | 100 |
| `sample_rate` | データサンプリング率 | 1 |

### 指標設定 (indicators)

各テクニカル指標のON/OFF切り替えと詳細パラメータ：

- Bollinger Bands (`use_bbands`)
- MACD (`use_macd`)
- RSI (`use_rsi`)
- ATR (`use_atr`)
- CCI (`use_cci`)
- Volume (`use_volume`)
- SMA/EMA (`use_sma`, `use_ema`)

### モデル設定 (model)

| パラメータ | 説明 | 選択肢/デフォルト |
|-----------|------|-----------------|
| `type` | モデルタイプ | Conv1D/LSTM/Hybrid/Transformer |
| `learning_rate` | 学習率 | 0.001 |
| `batch_size` | バッチサイズ | 512 |
| `epochs` | エポック数 | 100 |

## 🔬 出力・分析

### 自動生成される分析

1. **ラベル分布分析**
   - BUY/SELL/NO_TRADE の数量・割合
   - クラス不均衡の検出

2. **閾値分析**
   - 信頼度閾値別のF1スコア
   - 予測分布の変化

3. **モデル性能評価**
   - 混同行列
   - クラス別F1スコア
   - 訓練履歴可視化

4. **予測信頼度分析**
   - クラス別信頼度統計
   - 低信頼度予測の比率

### ログファイル

- `training.log`: 詳細な訓練ログ
- `model_summary.csv`: 全モデルの比較用サマリ
- `classification_report.json`: 詳細評価結果
- `threshold_analysis.csv`: 閾値別性能データ

## 🎯 使用例

### カスタム設定での訓練

```bash
# config.jsonを編集してから実行
python main.py
```

### 複数モデルの比較

```bash
# 設定を変更して複数回実行
# logs/model_summary.csv で性能比較
```

### リアルタイム予測

```python
from modules.utils import load_tick_data
from predict import ForexPredictor

# 予測器初期化
predictor = ForexPredictor('models/best_model.h5')

# 最新データで予測
recent_data = load_tick_data('data/recent_ticks.csv')
signal = predictor.get_trading_signal(recent_data)

print(f"推奨アクション: {signal['action']}")
print(f"信頼度: {signal['confidence']:.3f}")
```

## ⚠️ 注意事項

### データ品質

- ティックデータの品質が結果に大きく影響
- 異常値・欠損値の事前チェックを推奨
- 十分なデータ量（最低1ヶ月分）が必要

### 過学習対策

- 時系列分割による検証
- Early Stopping実装済み
- ドロップアウト・正則化適用

### メモリ管理

- 大容量データ時の自動チャンク処理
- GPU/CPUメモリ監視機能
- バッチサイズ自動調整

### 実運用時の考慮

- スプレッド変動の影響
- スリッページの考慮
- レイテンシとモデル更新頻度のバランス

## 🛠️ トラブルシューティング

### よくある問題

1. **GPU認識されない**
   ```bash
   # CUDA/cuDNN の確認
   nvidia-smi
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

2. **メモリ不足**
   - `batch_size` を小さくする
   - `sample_rate` を上げる（データを間引く）

3. **ラベル偏り**
   - `tp_pips`, `sl_pips` の調整
   - `future_window` の調整

4. **予測精度低下**
   - 特徴量の見直し
   - モデルタイプ変更
   - ハイパーパラメータチューニング

### エラーログ確認

```bash
# 最新の訓練ログ確認
tail -f logs/YYYYMMDD_HHMM/training.log

# エラー情報確認
cat logs/error_YYYYMMDD_HHMM/error.txt
```

## 📈 パフォーマンス最適化

### GPU活用

- TensorFlow-GPU 2.10.0 推奨
- mixed precision training 対応
- バッチサイズ最適化

### メモリ効率

- データジェネレーター使用
- 特徴量キャッシュ機能
- 段階的データ読み込み

## 🔄 モデル更新ワークフロー

1. 新しいデータ追加
2. config.json 調整
3. 訓練実行 (`python main.py`)
4. 性能評価 (`logs/model_summary.csv`)
5. 最良モデル選択
6. 本番環境デプロイ

## 📊 性能ベンチマーク

### 推奨環境性能

- **CPU**: AMD Ryzen 7 3700X クラス以上
- **GPU**: NVIDIA RTX 2070 SUPER (8GB VRAM) 以上
- **RAM**: 16GB以上
- **ストレージ**: SSD推奨

### 処理時間目安

- データ読み込み: 1M records/分
- 特徴量計算: 100K records/分
- モデル訓練: 1-2時間（100エポック）
- 推論: 1000 predictions/秒

## 🤝 貢献・カスタマイズ

### 新しい指標追加

1. `modules/feature_engineering.py` に計算関数追加
2. `config.json` に設定項目追加
3. `create_features()` メソッドに組み込み

### 新しいモデル追加

1. `modules/model.py` にモデル定義追加
2. `build_model()` メソッドに条件分岐追加

## 📝 ライセンス

このプロジェクトは私的利用・研究目的での使用を想定しています。商用利用時は適切なライセンス確認を行ってください。

## 🔗 関連リソース

- [TensorFlow公式ドキュメント](https://www.tensorflow.org/)
- [scikit-learn ユーザーガイド](https://scikit-learn.org/stable/user_guide.html)
- [MT5 Python API](https://www.mql5.com/en/docs/integration/python_metatrader5)

---

**