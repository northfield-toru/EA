"""
USDJPY スキャルピングEA用 学習パイプライン
時系列分割、Walk-forward validation、実トレード指標評価
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import TimeSeriesSplit
import os
import json
from datetime import datetime
from typing import Tuple, Dict, List

# 自作モジュール
from utils import USDJPYUtils
from data_loader import TickDataLoader, load_sample_data
from feature_engineering import FeatureEngineer
from labeling import ScalpingLabeler
from model import ScalpingCNNLSTM, FocalLoss

class ScalpingTrainer:
    """スキャルピングモデル学習管理クラス"""
    
    def __init__(self, 
                 data_path: str,
                 sequence_length: int = 30,
                 profit_pips: float = 6.0,      # 緩和: 8.0 → 6.0
                 loss_pips: float = 6.0,        # 緩和: 4.0 → 6.0
                 lookforward_ticks: int = 100,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 use_binary_classification: bool = False):  # 2値分類オプション
        """
        Args:
            data_path: ティックデータパス
            sequence_length: 時系列長
            profit_pips: 利確pips
            loss_pips: 損切りpips
            lookforward_ticks: 前方参照ティック数
            train_ratio: 学習データ比率
            val_ratio: 検証データ比率
            test_ratio: テストデータ比率
        """
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.profit_pips = profit_pips
        self.loss_pips = loss_pips
        self.lookforward_ticks = lookforward_ticks
        
        # データ分割比率
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比率の合計は1.0である必要があります"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.use_binary_classification = use_binary_classification
        
        # コンポーネント初期化（柔軟条件版）
        self.utils = USDJPYUtils()
        self.loader = TickDataLoader()
        self.feature_engineer = FeatureEngineer()
        self.labeler = ScalpingLabeler(
            profit_pips, loss_pips, lookforward_ticks,
            use_flexible_conditions=True  # 柔軟条件を使用
        )
        
        # データ格納
        self.ohlcv_data = None
        self.features_data = None
        self.labels_data = None
        self.model = None
        
        # 結果格納
        self.train_results = {}
        
        print(f"学習パイプライン初期化完了")
        print(f"利確:{profit_pips}pips, 損切:{loss_pips}pips, 前方参照:{lookforward_ticks}ティック")
        print(f"データ分割: Train:{train_ratio*100:.1f}%, Val:{val_ratio*100:.1f}%, Test:{test_ratio*100:.1f}%")
    
    def load_and_prepare_data(self, sample_size: int = None) -> Dict:
        """
        データ読み込みと前処理
        Args:
            sample_size: サンプルサイズ（Noneなら全データ）
        Returns:
            dict: 処理結果
        """
        print("=== データ読み込み・前処理開始 ===")
        
        # 1. ティック→1分足変換
        if sample_size:
            print(f"サンプルデータ読み込み: {sample_size:,} 行")
            self.ohlcv_data = load_sample_data(self.data_path, sample_size)
        else:
            print("全データ読み込み...")
            tick_data = self.loader.load_tick_data_auto(self.data_path)
            self.ohlcv_data = self.loader.tick_to_ohlcv_1min(tick_data)
        
        print(f"1分足データ: {len(self.ohlcv_data)} 本")
        
        # 2. 特徴量生成
        print("特徴量生成...")
        self.features_data = self.feature_engineer.create_all_features(
            self.ohlcv_data,
            include_advanced=True,
            include_lags=True
        )
        
        print(f"特徴量数: {len(self.features_data.columns)}")
        
        # 3. ラベル生成（2値分類 or 3値分類）
        print("ラベル生成...")
        if self.use_binary_classification:
            self.labels_data = self.labeler.create_binary_labels_vectorized(self.features_data)
            n_classes = 2
            print("2値分類モード: TRADE vs NO_TRADE")
        else:
            self.labels_data = self.labeler.create_labels_vectorized(self.features_data)
            n_classes = 3
            print("3値分類モード: BUY vs SELL vs NO_TRADE")
        
        # ラベル分布表示
        if self.use_binary_classification:
            label_names = {0: 'NO_TRADE', 1: 'TRADE'}
        else:
            label_names = {0: 'NO_TRADE', 1: 'BUY', 2: 'SELL'}
        
        print("ラベル分布:")
        total = len(self.labels_data)
        for label_val, count in label_dist.items():
            percentage = count / total * 100
            print(f"  {label_names[label_val]}: {count:,} ({percentage:.2f}%)")
        
        # 4. 欠損値処理
        print("データクリーニング...")
        complete_mask = ~(self.features_data.isna().any(axis=1) | self.labels_data.isna())
        
        self.features_data = self.features_data[complete_mask]
        self.labels_data = self.labels_data[complete_mask]
        
        print(f"完全データ: {len(self.features_data)} 行 ({len(self.features_data)/len(self.ohlcv_data)*100:.1f}%)")
    
    def split_data_timeseries(self) -> Tuple:
        """
        時系列順でのデータ分割
        Returns:
            tuple: 分割されたデータ
        """
        print("=== 時系列データ分割 ===")
        
        total_rows = len(self.features_data)
        
        # 分割点計算
        train_end = int(total_rows * self.train_ratio)
        val_end = int(total_rows * (self.train_ratio + self.val_ratio))
        
        print(f"総データ数: {total_rows:,}")
        print(f"学習データ: 0 〜 {train_end:,} ({train_end:,} 行)")
        print(f"検証データ: {train_end:,} 〜 {val_end:,} ({val_end-train_end:,} 行)")
        print(f"テストデータ: {val_end:,} 〜 {total_rows:,} ({total_rows-val_end:,} 行)")
        
        # データ分割
        train_features = self.features_data.iloc[:train_end]
        train_labels = self.labels_data.iloc[:train_end]
        
        val_features = self.features_data.iloc[train_end:val_end]
        val_labels = self.labels_data.iloc[train_end:val_end]
        
        test_features = self.features_data.iloc[val_end:]
        test_labels = self.labels_data.iloc[val_end:]
        
        # 各セットのラベル分布確認
        for name, labels in [('Train', train_labels), ('Val', val_labels), ('Test', test_labels)]:
            dist = labels.value_counts(normalize=True) * 100
            print(f"{name} ラベル分布: NO_TRADE:{dist.get(0,0):.1f}%, BUY:{dist.get(1,0):.1f}%, SELL:{dist.get(2,0):.1f}%")
        
        return (train_features, train_labels, val_features, val_labels, test_features, test_labels)
    
    def train_model(self, 
                   train_features: pd.DataFrame,
                   train_labels: pd.Series,
                   val_features: pd.DataFrame,
                   val_labels: pd.Series,
                   epochs: int = 100,
                   batch_size: int = 64,
                   **kwargs) -> Dict:
        """
        モデル学習実行
        Args:
            train_features: 学習用特徴量
            train_labels: 学習用ラベル
            val_features: 検証用特徴量
            val_labels: 検証用ラベル
            epochs: エポック数
            batch_size: バッチサイズ
        Returns:
            dict: 学習結果
        """
        print("=== モデル学習開始 ===")
        
        # モデル初期化（軽量化版使用）
        n_features = len(train_features.columns)
        self.model = ScalpingCNNLSTM(
            sequence_length=self.sequence_length,
            n_features=n_features,
            n_classes=3,
            cnn_filters=[16, 32],      # 軽量化
            kernel_sizes=[3, 5],       # 軽量化  
            lstm_units=32,             # 軽量化
            dropout_rate=0.5,          # 強化
            learning_rate=0.001
        )
        
        # シーケンスデータ準備
        print("シーケンス準備...")
        X_train, y_train_cat, y_train_raw = self.model.prepare_sequences(train_features, train_labels)
        X_val, y_val_cat, y_val_raw = self.model.prepare_sequences(val_features, val_labels)
        
        # クラス重み計算
        class_weights = self.model.calculate_class_weights(y_train_raw)
        
        # モデル構築
        self.model.build_model()
        
        # 学習実行
        history = self.model.train(
            X_train, y_train_cat,
            X_val, y_val_cat,
            epochs=epochs,
            batch_size=batch_size,
            **kwargs
        )
        
        # 学習結果保存
        self.train_results = {
            'history': history,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'class_weights': class_weights,
            'final_train_loss': history['loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'best_val_loss': min(history['val_loss'])
        }
        
        print("学習完了")
        return self.train_results
    
    def evaluate_model(self, 
                      test_features: pd.DataFrame, 
                      test_labels: pd.Series) -> Dict:
        """
        モデル評価（分類指標 + 実トレード指標）
        Args:
            test_features: テスト用特徴量
            test_labels: テスト用ラベル
        Returns:
            dict: 評価結果
        """
        print("=== モデル評価開始 ===")
        
        if self.model is None:
            raise ValueError("モデルが学習されていません")
        
        # シーケンス準備
        X_test, y_test_cat, y_test_raw = self.model.prepare_sequences(test_features, test_labels)
        
        # 予測実行
        pred_proba, pred_class = self.model.predict(X_test)
        
        # 分類指標
        accuracy = np.mean(pred_class == y_test_raw)
        f1 = f1_score(y_test_raw, pred_class, average='weighted')
        
        print(f"テストデータ評価:")
        print(f"  精度: {accuracy:.3f}")
        print(f"  F1スコア: {f1:.3f}")
        
        # 詳細レポート
        label_names = ['NO_TRADE', 'BUY', 'SELL']
        class_report = classification_report(y_test_raw, pred_class, target_names=label_names)
        print("\n分類レポート:")
        print(class_report)
        
        # 混同行列
        conf_matrix = confusion_matrix(y_test_raw, pred_class)
        print(f"\n混同行列:")
        print(conf_matrix)
        
        # 実トレード指標計算
        trade_metrics = self.calculate_trading_metrics(
            y_test_raw, pred_class, pred_proba
        )
        
        # 結果統合
        evaluation_results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'test_samples': len(X_test),
            **trade_metrics
        }
        
        return evaluation_results
    
    def calculate_trading_metrics(self, 
                                 y_true: np.array, 
                                 y_pred: np.array, 
                                 pred_proba: np.array) -> Dict:
        """
        実トレード指標計算
        Args:
            y_true: 実際のラベル
            y_pred: 予測ラベル
            pred_proba: 予測確率
        Returns:
            dict: トレード指標
        """
        metrics = {}
        
        # BUY/SELLの勝率計算
        buy_mask = y_pred == 1
        sell_mask = y_pred == 2
        
        if buy_mask.sum() > 0:
            buy_accuracy = np.mean(y_true[buy_mask] == 1)
            metrics['buy_win_rate'] = buy_accuracy
            metrics['buy_signals'] = buy_mask.sum()
        else:
            metrics['buy_win_rate'] = 0.0
            metrics['buy_signals'] = 0
        
        if sell_mask.sum() > 0:
            sell_accuracy = np.mean(y_true[sell_mask] == 2)
            metrics['sell_win_rate'] = sell_accuracy
            metrics['sell_signals'] = sell_mask.sum()
        else:
            metrics['sell_win_rate'] = 0.0
            metrics['sell_signals'] = 0
        
        # 期待利益計算（簡易版）
        total_signals = metrics['buy_signals'] + metrics['sell_signals']
        if total_signals > 0:
            total_wins = (metrics['buy_signals'] * metrics['buy_win_rate'] + 
                         metrics['sell_signals'] * metrics['sell_win_rate'])
            total_losses = total_signals - total_wins
            
            # 利確8pips, 損切4pipsでの期待利益
            expected_profit = (total_wins * self.profit_pips - total_losses * self.loss_pips)
            metrics['expected_profit_pips'] = expected_profit
            metrics['expected_profit_per_trade'] = expected_profit / total_signals if total_signals > 0 else 0
        else:
            metrics['expected_profit_pips'] = 0
            metrics['expected_profit_per_trade'] = 0
        
        # 信頼度分析
        high_confidence_threshold = 0.7
        high_conf_mask = np.max(pred_proba, axis=1) > high_confidence_threshold
        
        if high_conf_mask.sum() > 0:
            high_conf_accuracy = np.mean(y_true[high_conf_mask] == y_pred[high_conf_mask])
            metrics['high_confidence_accuracy'] = high_conf_accuracy
            metrics['high_confidence_signals'] = high_conf_mask.sum()
        else:
            metrics['high_confidence_accuracy'] = 0.0
            metrics['high_confidence_signals'] = 0
        
        print(f"\n実トレード指標:")
        print(f"  BUY勝率: {metrics['buy_win_rate']:.3f} ({metrics['buy_signals']} シグナル)")
        print(f"  SELL勝率: {metrics['sell_win_rate']:.3f} ({metrics['sell_signals']} シグナル)")
        print(f"  期待利益: {metrics['expected_profit_pips']:.1f} pips")
        print(f"  1トレード当たり期待利益: {metrics['expected_profit_per_trade']:.2f} pips")
        print(f"  高信頼度精度: {metrics['high_confidence_accuracy']:.3f} ({metrics['high_confidence_signals']} シグナル)")
        
        return metrics
    
    def walk_forward_validation(self, 
                               n_splits: int = 5,
                               epochs: int = 50,
                               batch_size: int = 64) -> Dict:
        """
        Walk-forward validation実行
        Args:
            n_splits: 分割数
            epochs: エポック数
            batch_size: バッチサイズ
        Returns:
            dict: WF検証結果
        """
        print("=== Walk-Forward Validation開始 ===")
        
        if self.features_data is None:
            raise ValueError("データが読み込まれていません")
        
        # 時系列分割器
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        wf_results = {
            'fold_results': [],
            'mean_accuracy': 0,
            'mean_f1': 0,
            'mean_profit_per_trade': 0
        }
        
        fold = 0
        for train_idx, test_idx in tscv.split(self.features_data):
            fold += 1
            print(f"\n--- Fold {fold}/{n_splits} ---")
            
            # データ分割
            train_features = self.features_data.iloc[train_idx]
            train_labels = self.labels_data.iloc[train_idx]
            test_features = self.features_data.iloc[test_idx]
            test_labels = self.labels_data.iloc[test_idx]
            
            # さらに学習・検証分割
            train_size = int(len(train_features) * 0.85)
            val_features = train_features.iloc[train_size:]
            val_labels = train_labels.iloc[train_size:]
            train_features = train_features.iloc[:train_size]
            train_labels = train_labels.iloc[:train_size]
            
            try:
                # モデル学習
                fold_model = ScalpingCNNLSTM(
                    sequence_length=self.sequence_length,
                    n_features=len(train_features.columns),
                    n_classes=3
                )
                
                # シーケンス準備
                X_train, y_train_cat, y_train_raw = fold_model.prepare_sequences(train_features, train_labels)
                X_val, y_val_cat, y_val_raw = fold_model.prepare_sequences(val_features, val_labels)
                X_test, y_test_cat, y_test_raw = fold_model.prepare_sequences(test_features, test_labels)
                
                # クラス重み計算
                fold_model.calculate_class_weights(y_train_raw)
                
                # 学習
                fold_model.build_model()
                fold_model.train(
                    X_train, y_train_cat,
                    X_val, y_val_cat,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=0
                )
                
                # 評価
                pred_proba, pred_class = fold_model.predict(X_test)
                
                # 指標計算
                accuracy = np.mean(pred_class == y_test_raw)
                f1 = f1_score(y_test_raw, pred_class, average='weighted')
                
                # トレード指標
                trade_metrics = self.calculate_trading_metrics(y_test_raw, pred_class, pred_proba)
                
                fold_result = {
                    'fold': fold,
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'profit_per_trade': trade_metrics['expected_profit_per_trade'],
                    'train_samples': len(X_train),
                    'test_samples': len(X_test)
                }
                
                wf_results['fold_results'].append(fold_result)
                
                print(f"Fold {fold} 結果: Acc={accuracy:.3f}, F1={f1:.3f}, Profit={trade_metrics['expected_profit_per_trade']:.2f}pips")
                
            except Exception as e:
                print(f"Fold {fold} エラー: {e}")
                continue
        
        # 平均値計算
        if wf_results['fold_results']:
            wf_results['mean_accuracy'] = np.mean([r['accuracy'] for r in wf_results['fold_results']])
            wf_results['mean_f1'] = np.mean([r['f1_score'] for r in wf_results['fold_results']])
            wf_results['mean_profit_per_trade'] = np.mean([r['profit_per_trade'] for r in wf_results['fold_results']])
            
            print(f"\n=== Walk-Forward Validation結果 ===")
            print(f"平均精度: {wf_results['mean_accuracy']:.3f}")
            print(f"平均F1スコア: {wf_results['mean_f1']:.3f}")
            print(f"平均1トレード利益: {wf_results['mean_profit_per_trade']:.2f} pips")
        
        return wf_results
    
    def save_results(self, output_dir: str = "training_results"):
        """学習結果保存"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 設定保存
        config = {
            'sequence_length': self.sequence_length,
            'profit_pips': self.profit_pips,
            'loss_pips': self.loss_pips,
            'lookforward_ticks': self.lookforward_ticks,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'timestamp': timestamp
        }
        
        with open(f"{output_dir}/config_{timestamp}.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        # モデル保存
        if self.model:
            self.model.save_model(f"{output_dir}/model_{timestamp}.h5")
        
        print(f"結果保存完了: {output_dir}")


def run_flexible_training_pipeline(data_path: str, 
                                  sample_size: int = 500000,
                                  epochs: int = 30,
                                  batch_size: int = 64,
                                  use_binary: bool = False) -> Dict:
    """
    柔軟条件での学習パイプライン実行
    Args:
        data_path: データパス
        sample_size: サンプルサイズ
        epochs: エポック数
        batch_size: バッチサイズ
        use_binary: 2値分類を使用するか
    Returns:
        dict: 全結果
    """
    print("=== USDJPY スキャルピングEA 柔軟条件学習パイプライン ===")
    
    # トレーナー初期化（柔軟条件）
    trainer = ScalpingTrainer(data_path, use_binary_classification=use_binary)
    
    # データ準備
    data_info = trainer.load_and_prepare_data(sample_size)
    
    # データ分割
    train_features, train_labels, val_features, val_labels, test_features, test_labels = trainer.split_data_timeseries()
    
    # モデル学習
    train_results = trainer.train_model(
        train_features, train_labels,
        val_features, val_labels,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # モデル評価
    eval_results = trainer.evaluate_model(test_features, test_labels)
    
    # 結果統合
    all_results = {
        'data_info': data_info,
        'train_results': train_results,
        'eval_results': eval_results
    }
    
    # 結果保存
    trainer.save_results()
    
    print("=== 柔軟条件学習パイプライン完了 ===")
    return all_results


def run_full_training_pipeline(data_path: str, 
                              sample_size: int = 500000,  # 増量: 100000 → 500000
                              epochs: int = 50,           # 増加: 100 → 50（早期停止で制御）
                              batch_size: int = 64) -> Dict:
    """
    完全な学習パイプライン実行
    Args:
        data_path: データパス
        sample_size: サンプルサイズ
        epochs: エポック数
        batch_size: バッチサイズ
    Returns:
        dict: 全結果
    """
    print("=== USDJPY スキャルピングEA 完全学習パイプライン ===")
    
    # トレーナー初期化
    trainer = ScalpingTrainer(data_path)
    
    # データ準備
    data_info = trainer.load_and_prepare_data(sample_size)
    
    # データ分割
    train_features, train_labels, val_features, val_labels, test_features, test_labels = trainer.split_data_timeseries()
    
    # モデル学習
    train_results = trainer.train_model(
        train_features, train_labels,
        val_features, val_labels,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # モデル評価
    eval_results = trainer.evaluate_model(test_features, test_labels)
    
    # 結果統合
    all_results = {
        'data_info': data_info,
        'train_results': train_results,
        'eval_results': eval_results
    }
    
    # 結果保存
    trainer.save_results()
    
    print("=== 学習パイプライン完了 ===")
    return all_results


if __name__ == "__main__":
    # 学習パイプラインテスト
    import sys
    
    data_path = "data/usdjpy_ticks.csv" if len(sys.argv) < 2 else sys.argv[1]
    sample_size = 500000  # 増量: テスト用サンプルサイズ
    
    print("=== 学習パイプライン テスト実行 ===")
    
    try:
        results = run_full_training_pipeline(
            data_path=data_path,
            sample_size=sample_size,
            epochs=30,  # 軽量化テスト用
            batch_size=64
        )
        
        print("テスト完了")
        
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()