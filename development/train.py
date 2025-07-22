"""
USDJPY スキャルピングEA用 学習パイプライン - 完全修正版
🔧 修正内容:
1. 2値分類完全対応
2. ラベル変換バグ修正
3. クラス重み適正化
4. 予測処理正常化
5. 全ての高度機能保持（妥協なし）
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
    """スキャルピングモデル学習管理クラス（完全修正版）"""
    
    def __init__(self, 
                 data_path: str,
                 sequence_length: int = 30,
                 profit_pips: float = 6.0,
                 loss_pips: float = 6.0,
                 lookforward_ticks: int = 100,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 use_binary_classification: bool = True):  # 🔧 デフォルトを2値分類に変更
        """
        完全修正版初期化
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
        
        # コンポーネント初期化
        self.utils = USDJPYUtils()
        self.loader = TickDataLoader()
        self.feature_engineer = FeatureEngineer()
        self.labeler = ScalpingLabeler(
            profit_pips, loss_pips, lookforward_ticks,
            use_or_conditions=True
        )
        
        # データ格納
        self.ohlcv_data = None
        self.features_data = None
        self.labels_data = None
        self.model = None
        
        # 結果格納
        self.train_results = {}
        
        print(f"🔧 修正版学習パイプライン初期化完了")
        print(f"利確:{profit_pips}pips, 損切:{loss_pips}pips, 前方参照:{lookforward_ticks}ティック")
        print(f"分類タイプ: {'2値分類(NO_TRADE/TRADE)' if use_binary_classification else '3値分類'}")
        print(f"データ分割: Train:{train_ratio*100:.1f}%, Val:{val_ratio*100:.1f}%, Test:{test_ratio*100:.1f}%")
    
    def load_and_prepare_data(self, sample_size: int = None) -> Dict:
        """
        データ読み込みと前処理（完全修正版）
        """
        print("=== データ読み込み・前処理開始（修正版） ===")
        
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
        self.features_data = self.feature_engineer.create_all_features_enhanced(self.ohlcv_data)
        print(f"特徴量数: {len(self.features_data.columns)}")
        
        # 3. 🔧 修正版ラベル生成（2値分類専用）
        print("🔧 修正版ラベル生成...")
        
        strict_labeler = ScalpingLabeler(
            profit_pips=self.profit_pips,
            loss_pips=self.loss_pips,
            lookforward_ticks=self.lookforward_ticks,
            use_or_conditions=False  # 厳格なAND条件
        )
        
        if self.use_binary_classification:
            # 🔧 修正: 確実に2値分類ラベルを生成
            self.labels_data = strict_labeler.create_binary_labels_strict(self.features_data)
            n_classes = 2
            print("✅ 2値分類ラベル生成: NO_TRADE vs TRADE")
        else:
            # 3値分類（既存機能保持）
            self.labels_data = strict_labeler.create_labels_vectorized(self.features_data)
            n_classes = 3
            print("✅ 3値分類ラベル生成: NO_TRADE vs BUY vs SELL")
        
        # ラベル分布確認
        if self.labels_data is not None and len(self.labels_data) > 0:
            label_dist = dict(zip(*np.unique(self.labels_data, return_counts=True)))
            total = len(self.labels_data)
            
            print(f"🔧 修正版ラベル分布:")
            if self.use_binary_classification:
                label_names = {0: 'NO_TRADE', 1: 'TRADE'}
                trade_ratio = label_dist.get(1, 0) / total
                print(f"  NO_TRADE: {label_dist.get(0, 0):,} ({(1-trade_ratio):.1%})")
                print(f"  TRADE: {label_dist.get(1, 0):,} ({trade_ratio:.1%})")
                
                # バランス評価
                if 0.05 <= trade_ratio <= 0.35:
                    print("  ✅ 理想的なTRADE比率です")
                elif trade_ratio > 0.35:
                    print("  ⚠️ TRADEがやや多めです")
                else:
                    print("  ⚠️ TRADEが少なめです")
            else:
                label_names = {0: 'NO_TRADE', 1: 'BUY', 2: 'SELL'}
                for label_val, count in label_dist.items():
                    percentage = count / total * 100
                    print(f"  {label_names.get(label_val, f'CLASS_{label_val}')}: {count:,} ({percentage:.2f}%)")
        
        # 4. データクリーニング
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
            'classification_type': '2値分類' if self.use_binary_classification else '3値分類'
        }

    def split_data_timeseries(self) -> Tuple:
        """
        時系列順でのデータ分割（修正版）
        """
        print("=== 時系列データ分割（修正版） ===")
        
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
        
        # 🔧 修正版: 各セットのラベル分布確認
        for name, labels in [('Train', train_labels), ('Val', val_labels), ('Test', test_labels)]:
            if self.use_binary_classification:
                trade_ratio = np.mean(labels == 1)
                no_trade_ratio = np.mean(labels == 0)
                print(f"{name} ラベル分布: NO_TRADE:{no_trade_ratio:.1%}, TRADE:{trade_ratio:.1%}")
            else:
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
        モデル学習実行（完全修正版）
        """
        print("=== モデル学習開始（修正版） ===")
        
        # 🔧 修正: 分類タイプに応じてクラス数を動的設定
        n_classes = 2 if self.use_binary_classification else 3
        print(f"分類タイプ: {n_classes}クラス ({'NO_TRADE/TRADE' if n_classes == 2 else 'NO_TRADE/BUY/SELL'})")
        
        # モデル初期化（修正版）
        n_features = len(train_features.columns)
        self.model = ScalpingCNNLSTM(
            sequence_length=self.sequence_length,
            n_features=n_features,
            n_classes=n_classes,  # 🔧 修正: 動的に設定
            cnn_filters=[16, 32],
            kernel_sizes=[3, 5],
            lstm_units=32,
            dropout_rate=0.5,
            learning_rate=0.001
        )
        
        # シーケンスデータ準備（修正版）
        print("🔧 修正版シーケンス準備...")
        X_train, y_train_cat, y_train_raw = self._prepare_sequences_fixed(train_features, train_labels)
        X_val, y_val_cat, y_val_raw = self._prepare_sequences_fixed(val_features, val_labels)
        
        # 🔧 修正版クラス重み計算
        class_weights = self._calculate_class_weights_fixed(y_train_raw)
        
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
        
        print("✅ 修正版学習完了")
        return self.train_results
    
    def _prepare_sequences_fixed(self, features_df: pd.DataFrame, labels: pd.Series) -> Tuple:
        """
        🔧 修正版シーケンス準備（2値分類完全対応）
        """
        print(f"🔧 修正版シーケンス準備: {len(features_df)} 行")
        
        # 数値型列のみ選択
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        features_numeric = features_df[numeric_columns].copy()
        
        # 欠損値処理
        features_clean = features_numeric.fillna(method='ffill').fillna(method='bfill')
        features_clean = features_clean.replace([np.inf, -np.inf], np.nan)
        features_clean = features_clean.fillna(0)
        
        # 特徴量スケーリング
        if not hasattr(self.model, 'scaler') or self.model.scaler is None:
            from sklearn.preprocessing import RobustScaler
            self.model.scaler = RobustScaler()
            scaled_features = self.model.scaler.fit_transform(features_clean)
        else:
            scaled_features = self.model.scaler.transform(features_clean)
        
        # シーケンス作成
        sequences = []
        sequence_labels = []
        
        for i in range(self.sequence_length, len(scaled_features)):
            seq = scaled_features[i-self.sequence_length:i]
            sequences.append(seq)
            sequence_labels.append(labels.iloc[i])
        
        X = np.array(sequences)
        y = np.array(sequence_labels)
        
        print(f"シーケンス作成完了: {X.shape}")
        
        # 🔧 修正版ラベル検証・変換
        unique_labels = np.unique(y)
        print(f"🔧 ラベル検証: {unique_labels}")
        
        if self.use_binary_classification:
            # 2値分類の場合：0と1のみを許可
            if not all(label in [0, 1] for label in unique_labels):
                print(f"⚠️ 2値分類ラベル修正が必要: {unique_labels}")
                # 🔧 修正: より適切な変換ロジック
                y_fixed = np.zeros_like(y)
                for i, label in enumerate(y):
                    if label >= 1:  # 1以上はTRADE
                        y_fixed[i] = 1
                    else:  # 0はNO_TRADE
                        y_fixed[i] = 0
                y = y_fixed
                unique_labels = np.unique(y)
                print(f"✅ 修正後ラベル: {unique_labels}")
            
            expected_classes = 2
        else:
            # 3値分類の場合：0,1,2を許可
            expected_classes = 3
        
        # ラベル分布確認
        label_dist = dict(zip(*np.unique(y, return_counts=True)))
        print(f"最終ラベル分布: {label_dist}")
        
        # 🔧 修正版 One-hot エンコーディング
        from tensorflow.keras.utils import to_categorical
        y_categorical = to_categorical(y, num_classes=expected_classes)
        
        print(f"✅ カテゴリカル変換完了: {y_categorical.shape}")
        
        return X, y_categorical, y
    
    def _calculate_class_weights_fixed(self, y: np.array) -> Dict:
        """
        🔧 修正版クラス重み計算（適正化）
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        # 基本的なbalanced重み
        unique_classes = np.unique(y)
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=y
        )
        
        class_weight_dict = dict(zip(unique_classes, class_weights))
        
        # 🔧 修正: 過度な強化を防止
        if self.use_binary_classification:
            label_counts = np.bincount(y, minlength=2)
            total_samples = len(y)
            trade_ratio = label_counts[1] / total_samples
            
            print(f"🔧 クラス重み分析:")
            print(f"  NO_TRADE: {label_counts[0]} ({(1-trade_ratio):.1%})")
            print(f"  TRADE: {label_counts[1]} ({trade_ratio:.1%})")
            
            # 🔧 修正: 適度な調整（過度な強化を防止）
            if trade_ratio < 0.2:  # 20%未満の場合のみ軽度調整
                enhancement_factor = 1.2  # 1.5 → 1.2に軽減
                class_weight_dict[1] *= enhancement_factor
                print(f"  軽度TRADE強化: {enhancement_factor}倍")
            
            # 🔧 修正: 重みの上限設定（極端な偏りを防止）
            max_weight = 3.0
            for class_idx in class_weight_dict:
                if class_weight_dict[class_idx] > max_weight:
                    print(f"  重み上限適用: クラス{class_idx} {class_weight_dict[class_idx]:.2f} → {max_weight}")
                    class_weight_dict[class_idx] = max_weight
        
        print(f"✅ 最終クラス重み: {class_weight_dict}")
        self.model.class_weights = class_weight_dict
        return class_weight_dict
    
    def evaluate_model(self, 
                      test_features: pd.DataFrame, 
                      test_labels: pd.Series) -> Dict:
        """
        モデル評価（完全修正版）
        """
        print("=== モデル評価開始（修正版） ===")
        
        if self.model is None:
            raise ValueError("モデルが学習されていません")
        
        # 🔧 修正版シーケンス準備
        X_test, y_test_cat, y_test_raw = self._prepare_sequences_fixed(test_features, test_labels)
        
        # 🔧 修正版予測実行
        pred_proba, pred_class = self._predict_fixed(X_test)
        
        # 基本指標
        accuracy = np.mean(pred_class == y_test_raw)
        f1 = f1_score(y_test_raw, pred_class, average='weighted')
        
        print(f"🔧 修正版テストデータ評価:")
        print(f"  精度: {accuracy:.3f}")
        print(f"  F1スコア: {f1:.3f}")
        
        # 🔧 修正版予測分布確認
        pred_dist = dict(zip(*np.unique(pred_class, return_counts=True)))
        print(f"  予測分布: {pred_dist}")
        
        # 分類タイプに応じたラベル名設定
        if self.use_binary_classification:
            label_names = ['NO_TRADE', 'TRADE']
            expected_classes = 2
        else:
            label_names = ['NO_TRADE', 'BUY', 'SELL']
            expected_classes = 3
        
        # 実際のクラス数チェック
        actual_classes = len(np.unique(y_test_raw))
        if actual_classes != expected_classes:
            print(f"⚠️ クラス数不整合: 期待{expected_classes}、実際{actual_classes}")
            # 実際のクラス数に合わせて調整
            label_names = [f'CLASS_{i}' for i in range(actual_classes)]
        
        # 詳細レポート
        try:
            class_report = classification_report(
                y_test_raw, pred_class, 
                target_names=label_names,
                labels=list(range(len(label_names))),
                zero_division=0
            )
            print(f"\n✅ 分類レポート:")
            print(class_report)
        except Exception as e:
            print(f"⚠️ 分類レポート生成エラー: {e}")
            class_report = f"精度: {accuracy:.3f}, F1: {f1:.3f}"
        
        # 混同行列
        try:
            conf_matrix = confusion_matrix(
                y_test_raw, pred_class,
                labels=list(range(len(label_names)))
            )
            print(f"\n混同行列:")
            print(conf_matrix)
        except Exception as e:
            print(f"⚠️ 混同行列生成エラー: {e}")
            conf_matrix = np.array([[0]])
        
        # 🔧 修正版トレード指標計算
        if self.use_binary_classification:
            trade_metrics = self._calculate_binary_trading_metrics_fixed(
                y_test_raw, pred_class, pred_proba
            )
        else:
            trade_metrics = self._calculate_trading_metrics_fixed(
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
            'prediction_distribution': pred_dist,
            **trade_metrics
        }
        
        return evaluation_results
    
    def _predict_fixed(self, X: np.array) -> Tuple[np.array, np.array]:
        """🔧 修正版予測実行"""
        if self.model is None or self.model.model is None:
            raise ValueError("モデルが構築されていません")
        
        # Kerasモデルから予測
        keras_model = self.model.keras_model if hasattr(self.model, 'keras_model') else self.model.model
        predictions_proba = keras_model.predict(X, verbose=0)
        
        # マルチ出力対応
        if isinstance(predictions_proba, list):
            main_predictions = predictions_proba[0]
        else:
            main_predictions = predictions_proba
        
        predictions_class = np.argmax(main_predictions, axis=1)
        
        print(f"🔧 予測実行完了:")
        print(f"  入力shape: {X.shape}")
        print(f"  出力shape: {main_predictions.shape}")
        print(f"  予測分布: {dict(zip(*np.unique(predictions_class, return_counts=True)))}")
        
        return main_predictions, predictions_class
    
    def _calculate_binary_trading_metrics_fixed(self, 
                                               y_true: np.array, 
                                               y_pred: np.array, 
                                               pred_proba: np.array) -> Dict:
        """
        🔧 修正版2値分類用トレード指標計算
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
        
        # 期待利益計算
        total_signals = metrics['trade_signals']
        if total_signals > 0:
            correct_trades = metrics['trade_signals'] * metrics['trade_win_rate']
            wrong_trades = total_signals - correct_trades
            
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
        
        print(f"\n🔧 修正版2値分類トレード指標:")
        print(f"  TRADE勝率: {metrics['trade_win_rate']:.3f} ({metrics['trade_signals']} シグナル)")
        print(f"  NO_TRADE精度: {metrics['no_trade_accuracy']:.3f} ({metrics['no_trade_signals']} シグナル)")
        print(f"  期待利益: {metrics['expected_profit_pips']:.1f} pips")
        print(f"  1トレード当たり期待利益: {metrics['expected_profit_per_trade']:.2f} pips")
        print(f"  高信頼度精度: {metrics['high_confidence_accuracy']:.3f} ({metrics['high_confidence_signals']} シグナル)")
        
        return metrics
    
    def _calculate_trading_metrics_fixed(self, 
                                        y_true: np.array, 
                                        y_pred: np.array, 
                                        pred_proba: np.array) -> Dict:
        """
        🔧 修正版3値分類用トレード指標計算
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
        
        # 期待利益計算
        total_signals = metrics['buy_signals'] + metrics['sell_signals']
        if total_signals > 0:
            total_wins = (metrics['buy_signals'] * metrics['buy_win_rate'] + 
                         metrics['sell_signals'] * metrics['sell_win_rate'])
            total_losses = total_signals - total_wins
            
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
        
        print(f"\n🔧 修正版3値分類トレード指標:")
        print(f"  BUY勝率: {metrics['buy_win_rate']:.3f} ({metrics['buy_signals']} シグナル)")
        print(f"  SELL勝率: {metrics['sell_win_rate']:.3f} ({metrics['sell_signals']} シグナル)")
        print(f"  期待利益: {metrics['expected_profit_pips']:.1f} pips")
        print(f"  1トレード当たり期待利益: {metrics['expected_profit_per_trade']:.2f} pips")
        print(f"  高信頼度精度: {metrics['high_confidence_accuracy']:.3f} ({metrics['high_confidence_signals']} シグナル)")
        
        return metrics
    
    def save_results(self, output_dir: str = "training_results"):
        """学習結果保存（修正版）"""
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
            'use_binary_classification': self.use_binary_classification,
            'timestamp': timestamp,
            'version': 'fixed_train_py'
        }
        
        with open(f"{output_dir}/config_{timestamp}.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        # 🔧 修正版モデル保存
        if self.model:
            model_path = f"{output_dir}/fixed_model_{timestamp}.h5"
            try:
                if hasattr(self.model, 'keras_model'):
                    self.model.keras_model.save(model_path)
                elif hasattr(self.model, 'model'):
                    self.model.model.save(model_path)
                else:
                    self.model.save(model_path)
                print(f"✅ 修正版モデル保存完了: {model_path}")
            except Exception as e:
                print(f"⚠️ モデル保存エラー: {e}")
        
        print(f"✅ 修正版結果保存完了: {output_dir}")


# =======================================
# 完全修正版学習パイプライン関数群
# =======================================

def run_fixed_training_pipeline(data_path: str = "data/usdjpy_ticks.csv",
                                sample_size: int = 500000,
                                epochs: int = 50,
                                batch_size: int = 64) -> Dict:
    """
    🔧 完全修正版学習パイプライン実行
    """
    print("🚀" * 30)
    print("    完全修正版学習パイプライン")
    print("    バグ修正・機能完全保持")
    print("🚀" * 30)
    
    # 修正版トレーナー初期化
    trainer = ScalpingTrainer(
        data_path=data_path,
        use_binary_classification=True  # 2値分類固定
    )
    
    try:
        # データ準備
        print("📊 STEP 1: データ準備...")
        data_info = trainer.load_and_prepare_data(sample_size)
        
        # データ分割
        print("📊 STEP 2: データ分割...")
        train_features, train_labels, val_features, val_labels, test_features, test_labels = trainer.split_data_timeseries()
        
        # モデル学習
        print("🧠 STEP 3: モデル学習...")
        train_results = trainer.train_model(
            train_features, train_labels,
            val_features, val_labels,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # モデル評価
        print("📊 STEP 4: モデル評価...")
        eval_results = trainer.evaluate_model(test_features, test_labels)
        
        # 結果保存
        print("💾 STEP 5: 結果保存...")
        trainer.save_results("fixed_training_results")
        
        # 結果統合
        all_results = {
            'version': 'fixed_train_py',
            'data_info': data_info,
            'train_results': train_results,
            'eval_results': eval_results,
            'success': True,
            'timestamp': datetime.now().isoformat()
        }
        
        # 成功判定
        trade_pred_ratio = eval_results.get('prediction_distribution', {}).get(1, 0) / eval_results.get('test_samples', 1)
        trade_win_rate = eval_results.get('trade_win_rate', 0)
        avg_profit = eval_results.get('expected_profit_per_trade', 0)
        
        print(f"\n🎯 完全修正版結果:")
        print(f"  予測TRADE比率: {trade_pred_ratio:.1%}")
        print(f"  TRADE勝率: {trade_win_rate:.1%}")
        print(f"  平均利益: {avg_profit:+.2f} pips")
        
        if 0.1 <= trade_pred_ratio <= 0.6 and trade_win_rate >= 0.35:
            print("✅ 修正成功！バランスの取れたモデル構築完了")
            all_results['fix_success'] = True
        else:
            print("⚠️ 部分的改善。さらなる調整が推奨されます")
            all_results['fix_success'] = False
        
        return all_results
        
    except Exception as e:
        print(f"❌ 修正版学習エラー: {e}")
        import traceback
        traceback.print_exc()
        return {
            'version': 'fixed_train_py',
            'success': False,
            'error': str(e)
        }

def test_fixed_model_quick(model_path: str = None) -> Dict:
    """
    🔧 修正版モデルクイックテスト
    """
    print("🧪 修正版モデルクイックテスト...")
    
    try:
        if model_path is None:
            # 最新の修正版モデルを検索
            import glob
            fixed_models = glob.glob("fixed_training_results/fixed_model_*.h5")
            if fixed_models:
                model_path = max(fixed_models, key=os.path.getctime)
                print(f"📁 最新修正版モデル使用: {model_path}")
            else:
                print("❌ 修正版モデルが見つかりません")
                return {'error': 'No fixed model found'}
        
        # unified_backtest.pyでクイックテスト
        from unified_backtest import UnifiedBacktestSystem
        
        system = UnifiedBacktestSystem(model_path)
        
        # 短期間でのクイックテスト
        results = system.run_single_test(
            data_path="data/usdjpy_ticks.csv",
            start_date="2025-07-10",
            end_date="2025-07-16",
            all_data=True,
            tp_pips=4.0,
            sl_pips=6.0,
            confidence_threshold=0.6,
            mode='tick-precise-fixed'
        )
        
        if results and 'error' not in results:
            trade_count = results['total_trades']
            win_rate = results['win_rate']
            avg_pips = results['avg_pips_per_trade']
            
            print(f"🧪 クイックテスト結果:")
            print(f"  取引数: {trade_count}")
            print(f"  勝率: {win_rate:.1%}")
            print(f"  平均収益: {avg_pips:+.2f} pips")
            
            if 5 <= trade_count <= 500 and win_rate >= 0.35:
                print("✅ 修正版モデル正常動作確認")
                return {
                    'success': True,
                    'trade_count': trade_count,
                    'win_rate': win_rate,
                    'avg_pips': avg_pips,
                    'model_path': model_path
                }
            else:
                print("⚠️ パフォーマンス要改善")
                return {
                    'success': False,
                    'reason': 'performance_issues',
                    'details': results
                }
        else:
            print(f"❌ テスト失敗: {results.get('error', 'Unknown error') if results else 'No results'}")
            return {'error': 'test_failed', 'details': results}
            
    except Exception as e:
        print(f"❌ クイックテストエラー: {e}")
        return {'error': str(e)}

# メイン実行部分（デバッグ用）
if __name__ == "__main__":
    print("🔧 完全修正版train.py - バグ修正・機能完全保持")
    print("=" * 60)
    
    # 修正版パイプライン実行
    results = run_fixed_training_pipeline(
        sample_size=100000,  # テスト用に軽量化
        epochs=20
    )
    
    if results.get('success'):
        print(f"\n✅ 修正版学習完了")
        
        # クイックテスト実行
        test_results = test_fixed_model_quick()
        
        if test_results.get('success'):
            print(f"🎉 完全修正成功！実用レベル達成")
        else:
            print(f"📊 学習成功、性能調整継続中")
    else:
        print(f"\n❌ 修正版学習失敗: {results.get('error')}")
    
    print("\n💡 次のステップ:")
    print("1. model_retrain_script.py でこの修正版を使用")
    print("2. より大きなサンプルサイズでの再学習")
    print("3. unified_backtest.py での精密評価")