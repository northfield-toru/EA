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
        
        # コンポーネント初期化（ChatGPT提案の緩和条件）
        self.utils = USDJPYUtils()
        self.loader = TickDataLoader()
        self.feature_engineer = FeatureEngineer()
        self.labeler = ScalpingLabeler(
            profit_pips, loss_pips, lookforward_ticks,
            use_or_conditions=True  # ChatGPT提案: OR条件使用
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
        データ読み込みと前処理（厳格ラベリング版）
        """
        print("=== データ読み込み・前処理開始（厳格版） ===")
        
        # 1. データ読み込み（同じ）
        if sample_size:
            print(f"サンプルデータ読み込み: {sample_size:,} 行")
            self.ohlcv_data = load_sample_data(self.data_path, sample_size)
        else:
            print("全データ読み込み...")
            tick_data = self.loader.load_tick_data_auto(self.data_path)
            self.ohlcv_data = self.loader.tick_to_ohlcv_1min(tick_data)
        
        print(f"1分足データ: {len(self.ohlcv_data)} 本")
        
        # 2. 特徴量生成（同じ）
        print("特徴量生成...")
        self.features_data = self.feature_engineer.create_all_features(
            self.ohlcv_data,
            include_advanced=True,
            include_lags=True
        )
        
        print(f"特徴量数: {len(self.features_data.columns)}")
        
        # 3. 🔧 厳格ラベル生成
        print("厳格ラベル生成...")
        
        # より厳格な設定のラベラーを使用
        strict_labeler = ScalpingLabeler(
            profit_pips=6.0,         # 利確目標（適度）
            loss_pips=4.0,           # 損切り許容（厳格）
            lookforward_ticks=80,    # 観測期間短縮
            use_or_conditions=False  # AND条件必須
        )
        
        if self.use_binary_classification:
            self.labels_data = strict_labeler.create_binary_labels_strict(self.features_data)
            n_classes = 2
            print("厳格2値分類モード: TRADE vs NO_TRADE")
            label_dist = dict(zip(*np.unique(self.labels_data, return_counts=True)))
            label_names = {0: 'NO_TRADE', 1: 'TRADE'}
        else:
            self.labels_data = strict_labeler.create_labels_vectorized(self.features_data)
            n_classes = 3
            print("3値分類モード: BUY vs SELL vs NO_TRADE")
            label_dist = dict(zip(*np.unique(self.labels_data, return_counts=True)))
            label_names = {0: 'NO_TRADE', 1: 'BUY', 2: 'SELL'}
        
        # ラベル分布表示
        print("厳格ラベル分布:")
        total = len(self.labels_data)
        for label_val, count in label_dist.items():
            percentage = count / total * 100
            print(f"  {label_names[label_val]}: {count:,} ({percentage:.2f}%)")
        
        # バランス評価
        if self.use_binary_classification:
            trade_ratio = label_dist.get(1, 0) / total
            if 0.1 <= trade_ratio <= 0.4:
                print("✅ 理想的なTRADE比率です")
            elif trade_ratio > 0.4:
                print("⚠️ TRADEが多すぎます - 条件をより厳格に")
            else:
                print("⚠️ TRADEが少なすぎます - 条件を緩和検討")
        
        # 4. データクリーニング（同じ）
        print("データクリーニング...")
        complete_mask = ~(self.features_data.isna().any(axis=1) | self.labels_data.isna())
        
        self.features_data = self.features_data[complete_mask]
        self.labels_data = self.labels_data[complete_mask]
        
        print(f"完全データ: {len(self.features_data)} 行 ({len(self.features_data)/len(self.ohlcv_data)*100:.1f}%)")
        
        return {
            'ohlcv_rows': len(self.ohlcv_data),
            'feature_columns': len(self.features_data.columns),
            'complete_rows': len(self.features_data),
            'label_distribution': label_dist,
            'n_classes': n_classes,
            'labeling_method': 'strict_and_conditions'
        }

    def run_strict_training_pipeline(data_path: str, 
                                    sample_size: int = 500000,
                                    epochs: int = 50,
                                    batch_size: int = 64) -> Dict:
        """
        厳格条件での学習パイプライン実行（ChatGPT提案準拠）
        """
        print("=== USDJPY スキャルピングEA 厳格学習パイプライン ===")
        
        # 厳格設定のトレーナー初期化
        trainer = ScalpingTrainer(
            data_path, 
            use_binary_classification=True  # 2値分類推奨
        )
        
        # データ準備（厳格ラベリング使用）
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
            'eval_results': eval_results,
            'approach': 'strict_and_conditions',
            'target_balance': 'TRADE:NO_TRADE = 1:3 to 1:5'
        }
        
        # 結果保存
        trainer.save_results()
        
        print("=== 厳格学習パイプライン完了 ===")
        return all_results

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
        
        # 🔧 FIX: 2値分類 vs 3値分類に応じてクラス数を動的設定
        if self.use_binary_classification:
            n_classes = 2
            print("2値分類モデル (TRADE vs NO_TRADE)")
        else:
            n_classes = 3
            print("3値分類モデル (BUY vs SELL vs NO_TRADE)")
        
        # モデル初期化（軽量化版使用）
        n_features = len(train_features.columns)
        self.model = ScalpingCNNLSTM(
            sequence_length=self.sequence_length,
            n_features=n_features,
            n_classes=n_classes,  # 🔧 FIX: 動的に設定
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
            'best_val_loss': min(history['val_loss']),
            'n_classes': n_classes,
            'classification_type': '2値分類' if self.use_binary_classification else '3値分類'
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
        
        # 🔧 FIX: 2値分類 vs 3値分類に応じてラベル名を動的変更
        if self.use_binary_classification:
            label_names = ['NO_TRADE', 'TRADE']
            expected_classes = 2
        else:
            label_names = ['NO_TRADE', 'BUY', 'SELL']
            expected_classes = 3
        
        # 実際のクラス数チェック
        actual_classes = len(np.unique(y_test_raw))
        if actual_classes != expected_classes:
            print(f"警告: 期待クラス数({expected_classes})と実際のクラス数({actual_classes})が不一致")
            # 実際のクラス数に合わせてラベル名を調整
            if actual_classes == 2:
                label_names = ['NO_TRADE', 'TRADE']
            elif actual_classes == 3:
                label_names = ['NO_TRADE', 'BUY', 'SELL']
            else:
                # より多くのクラスがある場合
                label_names = [f'CLASS_{i}' for i in range(actual_classes)]
        
        # 詳細レポート
        try:
            class_report = classification_report(
                y_test_raw, pred_class, 
                target_names=label_names,
                labels=list(range(len(label_names))),  # 明示的にラベルを指定
                zero_division=0  # ゼロ除算警告を抑制
            )
            print("\n分類レポート:")
            print(class_report)
        except Exception as e:
            print(f"分類レポート生成エラー: {e}")
            # フォールバック: シンプルなレポート
            class_report = f"精度: {accuracy:.3f}, F1: {f1:.3f}"
            print(f"簡易レポート: {class_report}")
        
        # 混同行列
        try:
            conf_matrix = confusion_matrix(
                y_test_raw, pred_class,
                labels=list(range(len(label_names)))  # 明示的にラベルを指定
            )
            print(f"\n混同行列:")
            print(conf_matrix)
        except Exception as e:
            print(f"混同行列生成エラー: {e}")
            conf_matrix = np.array([[0]])  # ダミー行列
        
        # 🔧 FIX: 2値分類の場合のトレード指標を調整
        if self.use_binary_classification:
            trade_metrics = self.calculate_binary_trading_metrics(
                y_test_raw, pred_class, pred_proba
            )
        else:
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
            'classification_type': '2値分類' if self.use_binary_classification else '3値分類',
            **trade_metrics
        }
        
        return evaluation_results

    def calculate_binary_trading_metrics(self, 
                                       y_true: np.array, 
                                       y_pred: np.array, 
                                       pred_proba: np.array) -> Dict:
        """
        2値分類用の実トレード指標計算
        Args:
            y_true: 実際のラベル (0=NO_TRADE, 1=TRADE)
            y_pred: 予測ラベル (0=NO_TRADE, 1=TRADE)
            pred_proba: 予測確率
        Returns:
            dict: トレード指標
        """
        metrics = {}
        
        # TRADEシグナルの勝率
        trade_mask = y_pred == 1
        no_trade_mask = y_pred == 0
        
        if trade_mask.sum() > 0:
            trade_accuracy = np.mean(y_true[trade_mask] == 1)
            metrics['trade_win_rate'] = trade_accuracy
            metrics['trade_signals'] = trade_mask.sum()
        else:
            metrics['trade_win_rate'] = 0.0
            metrics['trade_signals'] = 0
        
        # NO_TRADEの精度
        if no_trade_mask.sum() > 0:
            no_trade_accuracy = np.mean(y_true[no_trade_mask] == 0)
            metrics['no_trade_accuracy'] = no_trade_accuracy
            metrics['no_trade_signals'] = no_trade_mask.sum()
        else:
            metrics['no_trade_accuracy'] = 0.0
            metrics['no_trade_signals'] = 0
        
        # 期待利益計算（簡易版）
        total_signals = metrics['trade_signals']
        if total_signals > 0:
            correct_trades = metrics['trade_signals'] * metrics['trade_win_rate']
            wrong_trades = total_signals - correct_trades
            
            # 正解時は利確、不正解時は損切り
            expected_profit = (correct_trades * self.profit_pips - wrong_trades * self.loss_pips)
            metrics['expected_profit_pips'] = expected_profit
            metrics['expected_profit_per_trade'] = expected_profit / total_signals
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
        
        print(f"\n2値分類トレード指標:")
        print(f"  TRADE勝率: {metrics['trade_win_rate']:.3f} ({metrics['trade_signals']} シグナル)")
        print(f"  NO_TRADE精度: {metrics['no_trade_accuracy']:.3f} ({metrics['no_trade_signals']} シグナル)")
        print(f"  期待利益: {metrics['expected_profit_pips']:.1f} pips")
        print(f"  1トレード当たり期待利益: {metrics['expected_profit_per_trade']:.2f} pips")
        print(f"  高信頼度精度: {metrics['high_confidence_accuracy']:.3f} ({metrics['high_confidence_signals']} シグナル)")
        
        return metrics
    
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
                                  batch_size: int = 64,
                                  use_binary: bool = True) -> Dict:  # デフォルトを2値分類に変更
        """
        完全な学習パイプライン実行
        Args:
            data_path: データパス
            sample_size: サンプルサイズ
            epochs: エポック数
            batch_size: バッチサイズ
            use_binary: 2値分類を使用するか
        Returns:
            dict: 全結果
        """
        print("=== USDJPY スキャルピングEA 完全学習パイプライン ===")
        
        # トレーナー初期化（2値分類デフォルト）
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
        
        print("=== 学習パイプライン完了 ===")
        return all_results

    def load_and_prepare_data_phase2a(self, 
                                      sample_size: int = None, 
                                      approach: str = "profit_focused") -> Dict:
        """
        Phase 2A用データ準備（勝率重視）
        Args:
            sample_size: サンプルサイズ
            approach: "profit_focused" or "ultra_conservative"
        Returns:
            dict: 処理結果
        """
        print(f"=== Phase 2A データ準備開始（{approach}）===")
        
        # 1. データ読み込み
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
        
        # 3. Phase 2A専用ラベル生成
        print(f"Phase 2A {approach} ラベル生成...")
        
        # 勝率重視設定のラベラー
        profit_labeler = ScalpingLabeler(
            profit_pips=6.0,         # ベース設定（内部で9pipsに調整）
            loss_pips=4.0,           # ベース設定（内部で3pipsに調整）
            lookforward_ticks=80,    # 短期観測
            use_or_conditions=False  # AND条件必須
        )
        
        if approach == "profit_focused":
            self.labels_data = profit_labeler.create_profit_focused_labels(self.features_data)
        elif approach == "ultra_conservative":
            self.labels_data = profit_labeler.create_ultra_conservative_labels(self.features_data)
        else:
            raise ValueError(f"未知のアプローチ: {approach}")
        
        # ラベル分布分析
        label_dist = dict(zip(*np.unique(self.labels_data, return_counts=True)))
        label_names = {0: 'NO_TRADE', 1: 'TRADE'}
        
        print(f"Phase 2A ラベル分布:")
        total = len(self.labels_data)
        for label_val, count in label_dist.items():
            percentage = count / total * 100
            print(f"  {label_names[label_val]}: {count:,} ({percentage:.2f}%)")
        
        # 期待勝率評価
        trade_ratio = label_dist.get(1, 0) / total
        expected_win_rate = 0.70 if approach == "profit_focused" else 0.80  # 期待勝率
        
        print(f"\n期待指標:")
        print(f"  目標勝率: {expected_win_rate:.0%}")
        print(f"  期待利益/トレード: +{6 if approach == 'profit_focused' else 10}pips")
        print(f"  シグナル頻度: {trade_ratio:.1%} (目標: 10-25%)")
        
        # 4. データクリーニング
        print("データクリーニング...")
        complete_mask = ~(self.features_data.isna().any(axis=1) | self.labels_data.isna())
        
        self.features_data = self.features_data[complete_mask]
        self.labels_data = self.labels_data[complete_mask]
        
        print(f"完了データ: {len(self.features_data)} 行 ({len(self.features_data)/len(self.ohlcv_data)*100:.1f}%)")
        
        return {
            'ohlcv_rows': len(self.ohlcv_data),
            'feature_columns': len(self.features_data.columns),
            'complete_rows': len(self.features_data),
            'label_distribution': label_dist,
            'approach': approach,
            'expected_win_rate': expected_win_rate,
            'trade_ratio': trade_ratio
        }

class BinaryScalpingTrainer:
    """2値分類専用トレーナー"""
    
    def __init__(self, data_path: str, **kwargs):
        # 基本的なセットアップは ScalpingTrainer と同じ
        pass
    
    def create_buy_sell_models(self, features_df: pd.DataFrame) -> tuple:
        """
        BUYモデルとSELLモデルを個別に作成
        Returns:
            tuple: (buy_model, sell_model, buy_labels, sell_labels)
        """
        
        # BUY用ラベル: BUY vs NOT_BUY
        buy_labeler = ScalpingLabeler(
            profit_pips=4.0,  # 緩和
            loss_pips=8.0,    # 拡大
            use_or_conditions=True
        )
        
        buy_labels = []
        for i in range(len(features_df) - 1):
            future_max, future_min = buy_labeler._calculate_future_extremes(
                features_df['close'].values, i
            )
            if buy_labeler._check_buy_condition_relaxed(
                features_df['close'].iloc[i], future_max, future_min
            ):
                buy_labels.append(1)  # BUY
            else:
                buy_labels.append(0)  # NOT_BUY
        
        # SELL用ラベル: SELL vs NOT_SELL  
        sell_labels = []
        for i in range(len(features_df) - 1):
            future_max, future_min = buy_labeler._calculate_future_extremes(
                features_df['close'].values, i
            )
            if buy_labeler._check_sell_condition_relaxed(
                features_df['close'].iloc[i], future_max, future_min
            ):
                sell_labels.append(1)  # SELL
            else:
                sell_labels.append(0)  # NOT_SELL
        
        # 最後の行を追加（未来データなし）
        buy_labels.append(0)
        sell_labels.append(0)
        
        print(f"BUY分布: {np.bincount(buy_labels)}")
        print(f"SELL分布: {np.bincount(sell_labels)}")
        
        return buy_labels, sell_labels
    
    def train_dual_models(self, features_df: pd.DataFrame) -> dict:
        """
        BUY/SELLモデルを個別に学習
        """
        buy_labels, sell_labels = self.create_buy_sell_models(features_df)
        
        # データ分割
        train_end = int(len(features_df) * 0.7)
        val_end = int(len(features_df) * 0.85)
        
        # BUYモデル学習
        buy_model = ScalpingCNNLSTM(n_classes=2, learning_rate=0.001)
        buy_model.build_model()
        
        X_train, y_train_buy, _ = buy_model.prepare_sequences(
            features_df.iloc[:train_end], 
            pd.Series(buy_labels[:train_end])
        )
        X_val, y_val_buy, _ = buy_model.prepare_sequences(
            features_df.iloc[train_end:val_end], 
            pd.Series(buy_labels[train_end:val_end])
        )
        
        buy_history = buy_model.train(X_train, y_train_buy, X_val, y_val_buy, epochs=30)
        
        # SELLモデル学習
        sell_model = ScalpingCNNLSTM(n_classes=2, learning_rate=0.001)
        sell_model.build_model()
        
        _, y_train_sell, _ = sell_model.prepare_sequences(
            features_df.iloc[:train_end], 
            pd.Series(sell_labels[:train_end])
        )
        _, y_val_sell, _ = sell_model.prepare_sequences(
            features_df.iloc[train_end:val_end], 
            pd.Series(sell_labels[train_end:val_end])
        )
        
        sell_history = sell_model.train(X_train, y_train_sell, X_val, y_val_sell, epochs=30)
        
        return {
            'buy_model': buy_model,
            'sell_model': sell_model,
            'buy_history': buy_history,
            'sell_history': sell_history
        }

def run_balanced_training_pipeline(data_path: str, approach: str = "relaxed") -> dict:
    """
    バランス重視の学習パイプライン
    Args:
        approach: "relaxed" or "binary"
    """
    print(f"=== バランス重視学習パイプライン ({approach}) ===")
    
    if approach == "binary":
        trainer = BinaryScalpingTrainer(data_path)
        return trainer.train_dual_models()
    else:
        # relaxed approach
        trainer = ScalpingTrainer(data_path)
        trainer.labeler = ScalpingLabeler(
            profit_pips=4.0,
            loss_pips=8.0, 
            use_or_conditions=True
        )
        
        # 通常の学習フローだが、balanced_labels_vectorized を使用
        # （実装は上記の create_balanced_labels_vectorized を使用）
        
        return trainer  # 簡略化

def run_phase2b_realistic_pipeline(data_path: str, 
                                  tp_pips: float = 6.0,
                                  sl_pips: float = 4.0,
                                  sample_size: int = 500000,
                                  epochs: int = 35) -> Dict:
    """
    Phase 2B: 現実的利益パイプライン実行
    """
    print("=" * 60)
    print("    Phase 2B: REALISTIC PROFIT パイプライン")
    print(f"    設定: TP={tp_pips}pips, SL={sl_pips}pips")
    print("    目標: 勝率55-65%, 利益+1〜3pips/トレード")
    print("=" * 60)
    
    # トレーナー初期化
    trainer = ScalpingTrainer(data_path, use_binary_classification=True)
    
    # データ読み込み・特徴量生成
    if sample_size:
        trainer.ohlcv_data = load_sample_data(data_path, sample_size)
    else:
        tick_data = trainer.loader.load_tick_data_auto(data_path)
        trainer.ohlcv_data = trainer.loader.tick_to_ohlcv_1min(tick_data)
    
    trainer.features_data = trainer.feature_engineer.create_all_features(
        trainer.ohlcv_data, include_advanced=True, include_lags=True
    )
    
    # Phase 2B現実的ラベル生成
    print(f"Phase 2B 現実的ラベル生成...")
    realistic_labeler = ScalpingLabeler(
        profit_pips=tp_pips,
        loss_pips=sl_pips,
        lookforward_ticks=80,
        use_or_conditions=False
    )
    
    trainer.labels_data = realistic_labeler.create_realistic_profit_labels(
        trainer.features_data, tp_pips=tp_pips, sl_pips=sl_pips
    )
    
    # データクリーニング・分割
    complete_mask = ~(trainer.features_data.isna().any(axis=1) | trainer.labels_data.isna())
    trainer.features_data = trainer.features_data[complete_mask]
    trainer.labels_data = trainer.labels_data[complete_mask]
    
    train_features, train_labels, val_features, val_labels, test_features, test_labels = trainer.split_data_timeseries()
    
    # モデル学習・評価
    train_results = trainer.train_model(train_features, train_labels, val_features, val_labels, epochs=epochs)
    eval_results = trainer.evaluate_model(test_features, test_labels)
    
    # 結果分析
    print("\n" + "=" * 50)
    print("   Phase 2B REALISTIC 結果分析")
    print("=" * 50)
    
    trade_win_rate = eval_results.get('trade_win_rate', 0)
    profit_per_trade = eval_results.get('expected_profit_per_trade', 0)
    accuracy = eval_results.get('accuracy', 0)
    
    print(f"🎯 現実的指標:")
    print(f"  TRADE勝率: {trade_win_rate:.1%}")
    print(f"  利益/トレード: {profit_per_trade:+.2f}pips")
    print(f"  全体精度: {accuracy:.1%}")
    
    trainer.save_results()
    
    return {
        'phase': '2B_realistic',
        'approach': 'realistic_profit',
        'params': {'tp_pips': tp_pips, 'sl_pips': sl_pips},
        'eval_results': eval_results,
        'train_results': train_results
    }

def run_phase2b_momentum_pipeline(data_path: str,
                                 body_ratio: float = 0.7,
                                 min_body_pips: float = 4.0,
                                 sample_size: int = 500000,
                                 epochs: int = 35) -> Dict:
    """
    Phase 2B: モメンタム最適化パイプライン実行
    """
    print("=" * 60)
    print("    Phase 2B: MOMENTUM OPTIMIZED パイプライン")
    print(f"    設定: 実体比率={body_ratio}, 最小実体={min_body_pips}pips")
    print("    目標: 強いトレンドの初動を捉える")
    print("=" * 60)
    
    # トレーナー初期化
    trainer = ScalpingTrainer(data_path, use_binary_classification=True)
    
    # データ読み込み・特徴量生成
    if sample_size:
        trainer.ohlcv_data = load_sample_data(data_path, sample_size)
    else:
        tick_data = trainer.loader.load_tick_data_auto(data_path)
        trainer.ohlcv_data = trainer.loader.tick_to_ohlcv_1min(tick_data)
    
    trainer.features_data = trainer.feature_engineer.create_all_features(
        trainer.ohlcv_data, include_advanced=True, include_lags=True
    )
    
    # Phase 2Bモメンタムラベル生成
    print(f"Phase 2B モメンタムラベル生成...")
    momentum_labeler = ScalpingLabeler(
        profit_pips=6.0,
        loss_pips=4.0,
        lookforward_ticks=80,
        use_or_conditions=False
    )
    
    trainer.labels_data = momentum_labeler.create_momentum_optimized_labels(
        trainer.features_data, body_ratio=body_ratio, min_body_pips=min_body_pips
    )
    
    # データクリーニング・分割
    complete_mask = ~(trainer.features_data.isna().any(axis=1) | trainer.labels_data.isna())
    trainer.features_data = trainer.features_data[complete_mask]
    trainer.labels_data = trainer.labels_data[complete_mask]
    
    train_features, train_labels, val_features, val_labels, test_features, test_labels = trainer.split_data_timeseries()
    
    # モデル学習・評価
    train_results = trainer.train_model(train_features, train_labels, val_features, val_labels, epochs=epochs)
    eval_results = trainer.evaluate_model(test_features, test_labels)
    
    # 結果分析
    print("\n" + "=" * 50)
    print("   Phase 2B MOMENTUM 結果分析")
    print("=" * 50)
    
    trade_win_rate = eval_results.get('trade_win_rate', 0)
    profit_per_trade = eval_results.get('expected_profit_per_trade', 0)
    accuracy = eval_results.get('accuracy', 0)
    
    print(f"📈 モメンタム指標:")
    print(f"  TRADE勝率: {trade_win_rate:.1%}")
    print(f"  利益/トレード: {profit_per_trade:+.2f}pips")
    print(f"  全体精度: {accuracy:.1%}")
    
    trainer.save_results()
    
    return {
        'phase': '2B_momentum',
        'approach': 'momentum_optimized',
        'params': {'body_ratio': body_ratio, 'min_body_pips': min_body_pips},
        'eval_results': eval_results,
        'train_results': train_results
    }

def run_phase2b_conservative_pipeline(data_path: str,
                                     tp_pips: float = 5.0,
                                     sl_pips: float = 4.0,
                                     max_trade_ratio: float = 0.35,
                                     sample_size: int = 500000,
                                     epochs: int = 35) -> Dict:
    """
    Phase 2B: 保守的利益パイプライン実行
    """
    print("=" * 60)
    print("    Phase 2B: CONSERVATIVE PROFITABLE パイプライン")
    print(f"    設定: TP={tp_pips}pips, SL={sl_pips}pips, 最大TRADE={max_trade_ratio:.1%}")
    print("    目標: 高品質シグナルのみを厳選")
    print("=" * 60)
    
    # トレーナー初期化
    trainer = ScalpingTrainer(data_path, use_binary_classification=True)
    
    # データ読み込み・特徴量生成
    if sample_size:
        trainer.ohlcv_data = load_sample_data(data_path, sample_size)
    else:
        tick_data = trainer.loader.load_tick_data_auto(data_path)
        trainer.ohlcv_data = trainer.loader.tick_to_ohlcv_1min(tick_data)
    
    trainer.features_data = trainer.feature_engineer.create_all_features(
        trainer.ohlcv_data, include_advanced=True, include_lags=True
    )
    
    # Phase 2B保守的ラベル生成
    print(f"Phase 2B 保守的ラベル生成...")
    conservative_labeler = ScalpingLabeler(
        profit_pips=tp_pips,
        loss_pips=sl_pips,
        lookforward_ticks=80,
        use_or_conditions=False
    )
    
    trainer.labels_data = conservative_labeler.create_conservative_but_profitable_labels(
        trainer.features_data, tp_pips=tp_pips, sl_pips=sl_pips, max_trade_ratio=max_trade_ratio
    )
    
    # データクリーニング・分割
    complete_mask = ~(trainer.features_data.isna().any(axis=1) | trainer.labels_data.isna())
    trainer.features_data = trainer.features_data[complete_mask]
    trainer.labels_data = trainer.labels_data[complete_mask]
    
    train_features, train_labels, val_features, val_labels, test_features, test_labels = trainer.split_data_timeseries()
    
    # モデル学習・評価
    train_results = trainer.train_model(train_features, train_labels, val_features, val_labels, epochs=epochs)
    eval_results = trainer.evaluate_model(test_features, test_labels)
    
    # 結果分析
    print("\n" + "=" * 50)
    print("   Phase 2B CONSERVATIVE 結果分析")
    print("=" * 50)
    
    trade_win_rate = eval_results.get('trade_win_rate', 0)
    profit_per_trade = eval_results.get('expected_profit_per_trade', 0)
    accuracy = eval_results.get('accuracy', 0)
    
    print(f"🛡️ 保守的指標:")
    print(f"  TRADE勝率: {trade_win_rate:.1%}")
    print(f"  利益/トレード: {profit_per_trade:+.2f}pips")
    print(f"  全体精度: {accuracy:.1%}")
    
    trainer.save_results()
    
    return {
        'phase': '2B_conservative',
        'approach': 'conservative_profitable',
        'params': {'tp_pips': tp_pips, 'sl_pips': sl_pips, 'max_trade_ratio': max_trade_ratio},
        'eval_results': eval_results,
        'train_results': train_results
    }

def run_phase2b_comparison_suite(data_path: str, sample_size: int = 500000) -> Dict:
    """
    Phase 2B: 全アプローチ比較実行
    """
    print("🚀" * 20)
    print("    Phase 2B: 全アプローチ比較実行開始")
    print("🚀" * 20)
    
    results = {}
    best_approach = None
    best_profit = -999
    
    approaches = [
        {
            'name': 'realistic',
            'func': run_phase2b_realistic_pipeline,
            'params': {'tp_pips': 6.0, 'sl_pips': 4.0}
        },
        {
            'name': 'momentum',
            'func': run_phase2b_momentum_pipeline,
            'params': {'body_ratio': 0.7, 'min_body_pips': 4.0}
        },
        {
            'name': 'conservative',
            'func': run_phase2b_conservative_pipeline,
            'params': {'tp_pips': 5.0, 'sl_pips': 4.0, 'max_trade_ratio': 0.30}
        }
    ]
    
    # 各アプローチを実行
    for approach in approaches:
        print(f"\n{'='*60}")
        print(f"🔄 {approach['name'].upper()} アプローチ実行中...")
        print(f"{'='*60}")
        
        try:
            result = approach['func'](
                data_path=data_path,
                sample_size=sample_size,
                epochs=30,  # 比較用に軽量化
                **approach['params']
            )
            
            results[approach['name']] = result
            profit = result['eval_results'].get('expected_profit_per_trade', -999)
            
            if profit > best_profit:
                best_profit = profit
                best_approach = approach['name']
                
        except Exception as e:
            print(f"❌ {approach['name']} エラー: {e}")
            results[approach['name']] = {'error': str(e)}
            continue
    
    # 結果比較表示
    print("\n" + "🏆" * 20)
    print("    Phase 2B 最終比較結果")
    print("🏆" * 20)
    
    comparison_table = []
    for name, result in results.items():
        if 'error' not in result:
            eval_res = result['eval_results']
            comparison_table.append({
                'アプローチ': name.upper(),
                'TRADE勝率': f"{eval_res.get('trade_win_rate', 0):.1%}",
                '利益/トレード': f"{eval_res.get('expected_profit_per_trade', 0):+.2f}pips",
                '全体精度': f"{eval_res.get('accuracy', 0):.1%}",
                'TRADEシグナル数': eval_res.get('trade_signals', 0)
            })
    
    # 表形式で表示
    if comparison_table:
        print(f"{'アプローチ':<15} {'勝率':<8} {'利益':<12} {'精度':<8} {'シグナル数':<10}")
        print("-" * 65)
        for row in comparison_table:
            print(f"{row['アプローチ']:<15} {row['TRADE勝率']:<8} {row['利益/トレード']:<12} {row['全体精度']:<8} {row['TRADEシグナル数']:<10}")
    
    # 最優秀アプローチ発表
    if best_approach:
        print(f"\n🥇 最優秀アプローチ: {best_approach.upper()}")
        print(f"   利益: {best_profit:+.2f}pips/トレード")
        
        # 詳細分析
        best_result = results[best_approach]
        best_eval = best_result['eval_results']
        
        print(f"\n📊 {best_approach.upper()} 詳細分析:")
        print(f"   勝率: {best_eval.get('trade_win_rate', 0):.1%}")
        print(f"   精度: {best_eval.get('accuracy', 0):.1%}")
        print(f"   TRADEシグナル: {best_eval.get('trade_signals', 0)}")
        print(f"   NO_TRADEシグナル: {best_eval.get('no_trade_signals', 0)}")
        
        # Phase 1 → Phase 2B 改善幅
        phase1_profit = -1.10  # Phase 1の結果
        improvement = best_profit - phase1_profit
        print(f"\n📈 Phase 1 → Phase 2B 改善:")
        print(f"   Phase 1: -1.10pips → Phase 2B: {best_profit:+.2f}pips")
        print(f"   改善幅: {improvement:+.2f}pips/トレード")
        
        if best_profit > 0:
            print("   ✅ 利益化達成！")
        else:
            print("   ⚠️ まだ損失。さらなる改善が必要")
    
    return {
        'phase': '2B_comparison',
        'results': results,
        'best_approach': best_approach,
        'best_profit': best_profit,
        'comparison_table': comparison_table
    }

def _generate_phase2a_recommendations(eval_results: Dict, approach: str) -> List[str]:
    """
    Phase 2A結果に基づく推奨事項生成
    """
    recommendations = []
    
    win_rate = eval_results.get('trade_win_rate', 0)
    profit = eval_results.get('expected_profit_per_trade', 0)
    accuracy = eval_results.get('accuracy', 0)
    
    if win_rate < 0.60:
        if approach == "profit_focused":
            recommendations.append("勝率が低い → ultra_conservativeアプローチを試す")
        else:
            recommendations.append("勝率が低い → 利確目標をさらに上げる（15pips等）")
    
    if profit < 2.0:
        recommendations.append("利益が低い → 損切り許容をさらに狭める（2pips等）")
    
    if accuracy < 0.55:
        recommendations.append("全体精度が低い → 特徴量エンジニアリング強化を検討")
        recommendations.append("または → より長いlookforward_ticks（100-120）を試す")
    
    if len(recommendations) == 0:
        recommendations.append("✅ 良好な結果です！実運用テストに進めます")
    
    return recommendations
def run_phase2d_chatgpt_improved_pipeline(data_path: str,
                                         approach: str = "chatgpt_improved",
                                         tp_pips: float = 5.0,
                                         sl_pips: float = 3.0,
                                         sample_size: int = 500000,
                                         epochs: int = 35) -> Dict:
    """
    Phase 2D: ChatGPT改善版パイプライン実行
    - 強化特徴量エンジニアリング
    - lookforward_ticks=120
    - 最適化パラメータ
    """
    print("🚀" * 20)
    print("    Phase 2D: ChatGPT改善版パイプライン")
    print(f"    アプローチ: {approach}")
    print(f"    設定: TP={tp_pips}pips, SL={sl_pips}pips")
    print("    強化: 新特徴量 + lookforward=120")
    print("🚀" * 20)
    
    # トレーナー初期化
    trainer = ScalpingTrainer(data_path, use_binary_classification=True)
    
    # データ読み込み
    if sample_size:
        print(f"サンプルデータ読み込み: {sample_size:,} 行")
        trainer.ohlcv_data = load_sample_data(data_path, sample_size)
    else:
        print("全データ読み込み...")
        tick_data = trainer.loader.load_tick_data_auto(data_path)
        trainer.ohlcv_data = trainer.loader.tick_to_ohlcv_1min(tick_data)
    
    print(f"1分足データ: {len(trainer.ohlcv_data)} 本")
    
    # ChatGPT強化特徴量生成
    print("ChatGPT強化特徴量生成...")
    trainer.features_data = trainer.feature_engineer.create_all_features_enhanced(trainer.ohlcv_data)
    
    print(f"強化特徴量数: {len(trainer.features_data.columns)} 列")
    
    # ChatGPT改善ラベル生成
    print(f"ChatGPT改善ラベル生成 (lookforward=120)...")
    
    chatgpt_labeler = ScalpingLabeler(
        profit_pips=tp_pips,
        loss_pips=sl_pips,
        lookforward_ticks=120,  # ChatGPT提案
        use_or_conditions=False
    )
    
    if approach == "chatgpt_improved":
        trainer.labels_data = chatgpt_labeler.create_chatgpt_improved_labels(
            trainer.features_data, tp_pips=tp_pips, sl_pips=sl_pips
        )
    elif approach == "parameter_optimized":
        trainer.labels_data = chatgpt_labeler.create_parameter_optimized_labels(
            trainer.features_data, tp_pips=tp_pips, sl_pips=sl_pips
        )
    else:
        raise ValueError(f"未知のアプローチ: {approach}")
    
    # データクリーニング
    print("データクリーニング...")
    complete_mask = ~(trainer.features_data.isna().any(axis=1) | trainer.labels_data.isna())
    trainer.features_data = trainer.features_data[complete_mask]
    trainer.labels_data = trainer.labels_data[complete_mask]
    
    print(f"完全データ: {len(trainer.features_data)} 行")
    
    # ラベル分布確認
    label_dist = dict(zip(*np.unique(trainer.labels_data, return_counts=True)))
    total = len(trainer.labels_data)
    trade_ratio = label_dist.get(1, 0) / total
    
    print(f"\nChatGPT改善ラベル分布:")
    print(f"  TRADE: {label_dist.get(1, 0):,} ({trade_ratio:.1%})")
    print(f"  NO_TRADE: {label_dist.get(0, 0):,} ({(1-trade_ratio):.1%})")
    
    # データ分割
    train_features, train_labels, val_features, val_labels, test_features, test_labels = trainer.split_data_timeseries()
    
    # ChatGPT改善モデル学習
    print("ChatGPT改善モデル学習...")
    
    # より高度なモデル構成（特徴量が増えたため）
    n_features = len(train_features.columns)
    trainer.model = ScalpingCNNLSTM(
        sequence_length=trainer.sequence_length,
        n_features=n_features,
        n_classes=2,
        cnn_filters=[20, 40],     # 特徴量増加に対応
        kernel_sizes=[3, 5],
        lstm_units=28,            # やや増加
        dropout_rate=0.4,
        learning_rate=0.001
    )
    
    # 学習実行
    train_results = trainer.train_model(
        train_features, train_labels,
        val_features, val_labels,
        epochs=epochs,
        batch_size=64
    )
    
    # 評価
    eval_results = trainer.evaluate_model(test_features, test_labels)
    
    # ChatGPT改善結果分析
    print("\n" + "🚀" * 20)
    print("    ChatGPT改善結果分析")
    print("🚀" * 20)
    
    trade_win_rate = eval_results.get('trade_win_rate', 0)
    profit_per_trade = eval_results.get('expected_profit_per_trade', 0)
    accuracy = eval_results.get('accuracy', 0)
    
    print(f"📊 ChatGPT改善指標:")
    print(f"  TRADE勝率: {trade_win_rate:.1%} (目標: 50%+)")
    print(f"  利益/トレード: {profit_per_trade:+.2f}pips (目標: +0.5pips)")
    print(f"  全体精度: {accuracy:.1%}")
    print(f"  特徴量数: {n_features}")
    
    # 改善判定
    chatgpt_success = (trade_win_rate >= 0.50) and (profit_per_trade > 0.0)
    baseline_improvement = profit_per_trade > -1.29  # Phase 2B REALISTICとの比較
    
    if chatgpt_success:
        print("🎉 ChatGPT改善成功！利益化達成！")
    elif baseline_improvement:
        print("📈 ベースライン改善！さらなる最適化で利益化可能")
    else:
        print("⚠️ 追加改善が必要")
    
    # 改善履歴比較
    print(f"\n📈 全Phase比較:")
    print(f"  Phase 1: 勝率40.8%, 利益-1.10pips")
    print(f"  Phase 2A: 勝率25.0%, 利益-3.00pips")
    print(f"  Phase 2B: 勝率39.2%, 利益-1.29pips")
    print(f"  Phase 2D: 勝率{trade_win_rate:.1%}, 利益{profit_per_trade:+.2f}pips")
    
    improvement_2b = profit_per_trade - (-1.29)
    print(f"  Phase 2B→2D改善: {improvement_2b:+.2f}pips")
    
    trainer.save_results()
    
    return {
        'phase': '2D',
        'approach': approach,
        'eval_results': eval_results,
        'train_results': train_results,
        'chatgpt_success': chatgpt_success,
        'baseline_improvement': baseline_improvement,
        'feature_count': n_features,
        'data_info': {
            'trade_ratio': trade_ratio,
            'label_distribution': label_dist
        }
    }

def run_phase2d_grid_search(data_path: str, sample_size: int = 500000) -> Dict:
    """
    ChatGPT提案のグリッドサーチ実行
    tp_pips = [4, 5], sl_pips = [3], max_trade_ratio = [0.25, 0.3, 0.35, 0.4]
    """
    print("🔍" * 20)
    print("    Phase 2D: ChatGPTグリッドサーチ")
    print("🔍" * 20)
    
    # ChatGPT提案のパラメータ範囲
    tp_options = [4.0, 5.0]
    sl_options = [3.0]
    trade_ratio_options = [0.25, 0.30, 0.35, 0.40]
    
    results = {}
    best_result = None
    best_profit = -999
    
    total_combinations = len(tp_options) * len(sl_options) * len(trade_ratio_options)
    current_combination = 0
    
    print(f"グリッドサーチ開始: {total_combinations} 組み合わせ")
    
    for tp_pips in tp_options:
        for sl_pips in sl_options:
            for max_trade_ratio in trade_ratio_options:
                current_combination += 1
                
                print(f"\n{'='*50}")
                print(f"組み合わせ {current_combination}/{total_combinations}")
                print(f"TP={tp_pips}pips, SL={sl_pips}pips, 最大TRADE={max_trade_ratio:.0%}")
                print(f"{'='*50}")
                
                try:
                    # パラメータ最適化ラベルでテスト
                    result = run_phase2d_chatgpt_improved_pipeline(
                        data_path=data_path,
                        approach="parameter_optimized",
                        tp_pips=tp_pips,
                        sl_pips=sl_pips,
                        sample_size=sample_size,
                        epochs=25  # グリッドサーチ用に軽量化
                    )
                    
                    param_key = f"TP{tp_pips}_SL{sl_pips}_TR{max_trade_ratio:.2f}"
                    results[param_key] = result
                    
                    profit = result['eval_results'].get('expected_profit_per_trade', -999)
                    win_rate = result['eval_results'].get('trade_win_rate', 0)
                    
                    print(f"結果: 勝率{win_rate:.1%}, 利益{profit:+.2f}pips")
                    
                    if profit > best_profit:
                        best_profit = profit
                        best_result = result
                        best_params = (tp_pips, sl_pips, max_trade_ratio)
                        print(f"🏆 新ベスト: {profit:+.2f}pips")
                    
                except Exception as e:
                    print(f"❌ エラー: {e}")
                    continue
    
    # グリッドサーチ結果表示
    print("\n" + "🏆" * 30)
    print("    ChatGPTグリッドサーチ結果")
    print("🏆" * 30)
    
    # 結果テーブル作成
    print(f"{'パラメータ':<20} {'勝率':<8} {'利益':<10} {'精度':<8}")
    print("-" * 50)
    
    for param_key, result in results.items():
        if 'eval_results' in result:
            eval_res = result['eval_results']
            win_rate = eval_res.get('trade_win_rate', 0)
            profit = eval_res.get('expected_profit_per_trade', 0)
            accuracy = eval_res.get('accuracy', 0)
            
            print(f"{param_key:<20} {win_rate:.1%}    {profit:+.2f}pips  {accuracy:.1%}")
    
    # 最優秀結果
    if best_result:
        print(f"\n🥇 最優秀パラメータ:")
        print(f"   TP={best_params[0]}pips, SL={best_params[1]}pips, 最大TRADE={best_params[2]:.0%}")
        print(f"   利益: {best_profit:+.2f}pips/トレード")
        print(f"   勝率: {best_result['eval_results'].get('trade_win_rate', 0):.1%}")
        
        if best_profit > 0:
            print("   ✅ 利益化達成！")
        else:
            print("   ⚠️ さらなる改善が必要")
    
    return {
        'phase': '2D_grid_search',
        'results': results,
        'best_result': best_result,
        'best_profit': best_profit,
        'best_params': best_params if best_result else None,
        'total_combinations': total_combinations
    }

def run_phase2d_ensemble_pipeline(data_path: str, sample_size: int = 500000) -> Dict:
    """
    ChatGPT提案のアンサンブル手法実行
    realistic + conservative 両方がTRADEと予測した場合のみ実行
    """
    print("🤝" * 20)
    print("    Phase 2D: ChatGPTアンサンブル")
    print("    戦略: realistic AND conservative 合意")
    print("🤝" * 20)
    
    # トレーナー初期化
    trainer = ScalpingTrainer(data_path, use_binary_classification=True)
    
    # データ読み込み・特徴量生成
    if sample_size:
        trainer.ohlcv_data = load_sample_data(data_path, sample_size)
    else:
        tick_data = trainer.loader.load_tick_data_auto(data_path)
        trainer.ohlcv_data = trainer.loader.tick_to_ohlcv_1min(tick_data)
    
    trainer.features_data = trainer.feature_engineer.create_all_features_enhanced(trainer.ohlcv_data)
    
    # 複数のラベラーでラベル生成
    print("アンサンブル用ラベル生成...")
    
    # ラベラー1: realistic (最適化パラメータ)
    realistic_labeler = ScalpingLabeler(
        profit_pips=4.0, loss_pips=3.0, lookforward_ticks=120, use_or_conditions=False
    )
    realistic_labels = realistic_labeler.create_parameter_optimized_labels(
        trainer.features_data, tp_pips=4.0, sl_pips=3.0, max_trade_ratio=0.30
    )
    
    # ラベラー2: conservative
    conservative_labeler = ScalpingLabeler(
        profit_pips=5.0, loss_pips=3.0, lookforward_ticks=120, use_or_conditions=False
    )
    conservative_labels = conservative_labeler.create_chatgpt_improved_labels(
        trainer.features_data, tp_pips=5.0, sl_pips=3.0
    )
    
    # アンサンブルラベル生成: 両方がTRADEと判定した場合のみTRADE
    ensemble_labels = np.zeros(len(realistic_labels), dtype=int)
    
    realistic_trade = (realistic_labels == 1).sum()
    conservative_trade = (conservative_labels == 1).sum()
    ensemble_trade = 0
    
    for i in range(len(realistic_labels)):
        if realistic_labels.iloc[i] == 1 and conservative_labels.iloc[i] == 1:
            ensemble_labels[i] = 1
            ensemble_trade += 1
    
    trainer.labels_data = pd.Series(ensemble_labels, index=trainer.features_data.index, name='ensemble_label')
    
    print(f"アンサンブルラベル統計:")
    print(f"  realistic TRADE: {realistic_trade:,}")
    print(f"  conservative TRADE: {conservative_trade:,}")
    print(f"  ensemble TRADE: {ensemble_trade:,} ({ensemble_trade/len(ensemble_labels):.1%})")
    
    # データクリーニング・分割
    complete_mask = ~(trainer.features_data.isna().any(axis=1) | trainer.labels_data.isna())
    trainer.features_data = trainer.features_data[complete_mask]
    trainer.labels_data = trainer.labels_data[complete_mask]
    
    train_features, train_labels, val_features, val_labels, test_features, test_labels = trainer.split_data_timeseries()
    
    # アンサンブルモデル学習
    train_results = trainer.train_model(
        train_features, train_labels,
        val_features, val_labels,
        epochs=30,
        batch_size=64
    )
    
    # 評価
    eval_results = trainer.evaluate_model(test_features, test_labels)
    
    # アンサンブル結果分析
    print("\n" + "🤝" * 20)
    print("    アンサンブル結果分析")
    print("🤝" * 20)
    
    trade_win_rate = eval_results.get('trade_win_rate', 0)
    profit_per_trade = eval_results.get('expected_profit_per_trade', 0)
    accuracy = eval_results.get('accuracy', 0)
    
    print(f"📊 アンサンブル指標:")
    print(f"  TRADE勝率: {trade_win_rate:.1%}")
    print(f"  利益/トレード: {profit_per_trade:+.2f}pips")
    print(f"  全体精度: {accuracy:.1%}")
    print(f"  シグナル厳選効果: {ensemble_trade/len(ensemble_labels):.1%}")
    
    ensemble_success = (trade_win_rate >= 0.60) and (profit_per_trade > 0.0)
    
    if ensemble_success:
        print("🎉 アンサンブル成功！高精度・高利益達成！")
    else:
        print("🔧 アンサンブルでも改善が必要")
    
    trainer.save_results()
    
    return {
        'phase': '2D_ensemble',
        'approach': 'realistic_conservative_ensemble',
        'eval_results': eval_results,
        'train_results': train_results,
        'ensemble_success': ensemble_success,
        'ensemble_stats': {
            'realistic_trade': realistic_trade,
            'conservative_trade': conservative_trade,
            'ensemble_trade': ensemble_trade,
            'ensemble_ratio': ensemble_trade/len(ensemble_labels)
        }
    }

class ParameterOptimizedTrainer:
    """パラメータ別最適化トレーナー"""
    
    def __init__(self, data_path: str, base_output_dir: str = "results"):
        self.data_path = data_path
        self.base_output_dir = base_output_dir
        os.makedirs(base_output_dir, exist_ok=True)
        
    def train_single_parameter_set(self, 
                                  tp_pips: float,
                                  sl_pips: float, 
                                  trade_threshold: float,
                                  sample_size: int = 500000,
                                  epochs: int = 30) -> Dict:
        """
        単一パラメータセットでの完全学習・評価
        """
        param_id = f"TP{tp_pips}_SL{sl_pips}_TR{trade_threshold:.2f}"
        print(f"\n{'='*60}")
        print(f"パラメータ別学習開始: {param_id}")
        print(f"TP={tp_pips}pips, SL={sl_pips}pips, TRADE閾値={trade_threshold:.0%}")
        print(f"{'='*60}")
        
        # 出力ディレクトリ作成
        param_dir = os.path.join(self.base_output_dir, param_id)
        os.makedirs(param_dir, exist_ok=True)
        
        try:
            # 1. パラメータ専用トレーナー初期化
            trainer = ScalpingTrainer(
                self.data_path,
                profit_pips=tp_pips,    # 🔧 パラメータに合わせて設定
                loss_pips=sl_pips,      # 🔧 パラメータに合わせて設定
                use_binary_classification=True
            )
            
            # 2. データ読み込み・特徴量生成
            if sample_size:
                trainer.ohlcv_data = load_sample_data(self.data_path, sample_size)
            else:
                tick_data = trainer.loader.load_tick_data_auto(self.data_path)
                trainer.ohlcv_data = trainer.loader.tick_to_ohlcv_1min(tick_data)
            
            # ChatGPT強化特徴量使用
            trainer.features_data = trainer.feature_engineer.create_all_features_enhanced(trainer.ohlcv_data)
            
            # 3. パラメータ専用ラベル生成
            print(f"パラメータ専用ラベル生成...")
            param_labeler = ScalpingLabeler(
                profit_pips=tp_pips,
                loss_pips=sl_pips,
                lookforward_ticks=120,  # ChatGPT提案
                use_or_conditions=False
            )
            
            trainer.labels_data = param_labeler.create_parameter_optimized_labels(
                trainer.features_data,
                tp_pips=tp_pips,
                sl_pips=sl_pips,
                max_trade_ratio=trade_threshold
            )
            
            # 4. データクリーニング
            complete_mask = ~(trainer.features_data.isna().any(axis=1) | trainer.labels_data.isna())
            trainer.features_data = trainer.features_data[complete_mask]
            trainer.labels_data = trainer.labels_data[complete_mask]
            
            # ラベル分布確認
            label_dist = dict(zip(*np.unique(trainer.labels_data, return_counts=True)))
            total = len(trainer.labels_data)
            trade_ratio = label_dist.get(1, 0) / total
            
            print(f"パラメータ専用ラベル分布:")
            print(f"  TRADE: {label_dist.get(1, 0):,} ({trade_ratio:.1%})")
            print(f"  NO_TRADE: {label_dist.get(0, 0):,} ({(1-trade_ratio):.1%})")
            
            if trade_ratio < 0.05:
                print(f"⚠️ 警告: TRADEシグナルが5%未満。学習困難の可能性")
                return {'error': 'insufficient_trade_signals', 'trade_ratio': trade_ratio}
            
            # 5. データ分割
            train_features, train_labels, val_features, val_labels, test_features, test_labels = trainer.split_data_timeseries()
            
            # 6. パラメータ専用モデル学習
            print(f"パラメータ専用モデル学習...")
            
            # モデル設定（パラメータに応じて調整）
            n_features = len(train_features.columns)
            trainer.model = ScalpingCNNLSTM(
                sequence_length=trainer.sequence_length,
                n_features=n_features,
                n_classes=2,
                cnn_filters=[16, 32],
                kernel_sizes=[3, 5],
                lstm_units=24,
                dropout_rate=0.4,
                learning_rate=0.001
            )
            
            # 学習実行
            train_results = trainer.train_model(
                train_features, train_labels,
                val_features, val_labels,
                epochs=epochs,
                batch_size=64
            )
            
            # 7. 専用モデルで評価
            eval_results = trainer.evaluate_model(test_features, test_labels)
            
            # 8. 結果保存
            model_path = os.path.join(param_dir, f"model_{param_id}.h5")
            trainer.model.save_model(model_path)
            
            # 結果JSON保存
            results = {
                'parameters': {
                    'tp_pips': tp_pips,
                    'sl_pips': sl_pips,
                    'trade_threshold': trade_threshold,
                    'param_id': param_id
                },
                'data_info': {
                    'total_samples': total,
                    'trade_ratio': trade_ratio,
                    'label_distribution': label_dist
                },
                'train_results': train_results,
                'eval_results': eval_results,
                'model_path': model_path,
                'timestamp': datetime.now().isoformat()
            }
            
            results_path = os.path.join(param_dir, f"results_{param_id}.json")
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # 結果表示
            trade_win_rate = eval_results.get('trade_win_rate', 0)
            profit_per_trade = eval_results.get('expected_profit_per_trade', 0)
            accuracy = eval_results.get('accuracy', 0)
            
            print(f"\n📊 {param_id} 結果:")
            print(f"  TRADE勝率: {trade_win_rate:.1%}")
            print(f"  利益/トレード: {profit_per_trade:+.2f}pips")
            print(f"  全体精度: {accuracy:.1%}")
            print(f"  モデル保存: {model_path}")
            print(f"  結果保存: {results_path}")
            
            return results
            
        except Exception as e:
            error_result = {
                'parameters': {'tp_pips': tp_pips, 'sl_pips': sl_pips, 'trade_threshold': trade_threshold},
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
            error_path = os.path.join(param_dir, f"error_{param_id}.json")
            with open(error_path, 'w') as f:
                json.dump(error_result, f, indent=2)
            
            print(f"❌ {param_id} エラー: {e}")
            return error_result

def run_phase2e_proper_grid_search(data_path: str, 
                                  sample_size: int = 500000,
                                  epochs: int = 25) -> Dict:
    """
    Phase 2E: 正しいグリッドサーチ実行
    各パラメータセットごとにモデルを個別学習・評価
    """
    print("🔧" * 30)
    print("    Phase 2E: 正しいグリッドサーチ")
    print("    各パラメータでモデル個別学習")
    print("🔧" * 30)
    
    # パラメータ組み合わせ（ChatGPT提案）
    tp_options = [4.0, 5.0]
    sl_options = [3.0]
    trade_threshold_options = [0.25, 0.30, 0.35]  # 0.40は除外（TRADEが多すぎる傾向）
    
    total_combinations = len(tp_options) * len(sl_options) * len(trade_threshold_options)
    print(f"学習予定: {total_combinations} パラメータセット")
    
    # 最適化トレーナー初期化
    optimizer = ParameterOptimizedTrainer(data_path, "phase2e_results")
    
    all_results = {}
    best_result = None
    best_profit = -999
    best_param_id = None
    
    current_combination = 0
    
    for tp_pips in tp_options:
        for sl_pips in sl_options:
            for trade_threshold in trade_threshold_options:
                current_combination += 1
                
                print(f"\n🔄 進捗: {current_combination}/{total_combinations}")
                
                # パラメータセット別学習・評価
                result = optimizer.train_single_parameter_set(
                    tp_pips=tp_pips,
                    sl_pips=sl_pips,
                    trade_threshold=trade_threshold,
                    sample_size=sample_size,
                    epochs=epochs
                )
                
                param_id = f"TP{tp_pips}_SL{sl_pips}_TR{trade_threshold:.2f}"
                all_results[param_id] = result
                
                # 最良結果更新
                if 'eval_results' in result:
                    profit = result['eval_results'].get('expected_profit_per_trade', -999)
                    
                    if profit > best_profit:
                        best_profit = profit
                        best_result = result
                        best_param_id = param_id
                        print(f"🏆 新ベスト: {param_id} = {profit:+.2f}pips")
    
    # 最終結果表示
    print("\n" + "🏆" * 40)
    print("    Phase 2E 正しいグリッドサーチ結果")
    print("🏆" * 40)
    
    # 結果テーブル
    print(f"{'パラメータ':<20} {'勝率':<8} {'利益':<12} {'精度':<8} {'状態':<10}")
    print("-" * 70)
    
    successful_results = 0
    
    for param_id, result in all_results.items():
        if 'eval_results' in result:
            eval_res = result['eval_results']
            win_rate = eval_res.get('trade_win_rate', 0)
            profit = eval_res.get('expected_profit_per_trade', 0)
            accuracy = eval_res.get('accuracy', 0)
            status = "成功"
            successful_results += 1
        else:
            win_rate = 0
            profit = 0
            accuracy = 0
            status = "失敗"
        
        print(f"{param_id:<20} {win_rate:.1%}   {profit:+.2f}pips   {accuracy:.1%}   {status:<10}")
    
    # 最優秀結果
    if best_result and best_profit > -999:
        print(f"\n🥇 最優秀パラメータ: {best_param_id}")
        print(f"   利益: {best_profit:+.2f}pips/トレード")
        print(f"   勝率: {best_result['eval_results'].get('trade_win_rate', 0):.1%}")
        print(f"   精度: {best_result['eval_results'].get('accuracy', 0):.1%}")
        
        if best_profit > 0:
            print("   🎉 利益化達成！")
        else:
            print("   ⚠️ まだ損失だが、正しい学習が実行された")
    else:
        print("\n❌ 全パラメータセットで学習失敗")
    
    print(f"\n📊 学習成功率: {successful_results}/{total_combinations} ({successful_results/total_combinations:.1%})")
    
    # 統合結果保存
    summary_result = {
        'phase': '2E_proper_grid_search',
        'total_combinations': total_combinations,
        'successful_results': successful_results,
        'best_param_id': best_param_id,
        'best_profit': best_profit,
        'all_results': all_results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('phase2e_results/summary_results.json', 'w') as f:
        json.dump(summary_result, f, indent=2, default=str)
    
    return summary_result

# メイン実行部分を更新
if __name__ == "__main__":
    import sys
    
    data_path = "data/usdjpy_ticks.csv" if len(sys.argv) < 2 else sys.argv[1]
    
    print("🔧" * 35)
    print("    USDJPY スキャルピングEA Phase 2E")
    print("    正しいパラメータ別学習・評価")
    print("🔧" * 35)
    print("問題修正: 各パラメータセットでモデル個別学習")
    print("ChatGPT指摘: モデルとラベルの整合性確保")
    print()
    
    try:
        # Phase 2E: 正しいグリッドサーチ実行
        print("🚀 Phase 2E 正しいグリッドサーチ実行開始...")
        
        results = run_phase2e_proper_grid_search(
            data_path=data_path,
            sample_size=500000,
            epochs=25
        )
        
        print(f"\n🏁 Phase 2E 完了")
        
        if results['best_profit'] > 0:
            print(f"🎉 利益化達成！最良パラメータ: {results['best_param_id']}")
            print(f"💰 利益: {results['best_profit']:+.2f}pips/トレード")
        else:
            print(f"📈 最良結果: {results['best_param_id']} = {results['best_profit']:+.2f}pips")
            print(f"🔧 さらなる改善戦略が必要")
        
    except Exception as e:
        print(f"Phase 2E エラー: {e}")
        import traceback
        traceback.print_exc()

# 個別テスト用関数
def test_chatgpt_improved_only():
    """ChatGPT改善版のみテスト"""
    return run_phase2d_chatgpt_improved_pipeline(
        data_path="data/usdjpy_ticks.csv",
        approach="chatgpt_improved",
        sample_size=200000,
        epochs=20
    )

def test_parameter_optimized_only():
    """パラメータ最適化のみテスト"""
    return run_phase2d_chatgpt_improved_pipeline(
        data_path="data/usdjpy_ticks.csv",
        approach="parameter_optimized",
        sample_size=200000,
        epochs=20
    )