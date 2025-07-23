import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, List
import logging
import os
from datetime import datetime

from .model import ScalpingModel
from .utils import time_series_split, save_model_metadata, memory_usage_mb

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    スキャルピングモデル訓練管理クラス
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config['model']
        self.data_config = config['data']
        self.labels_config = config['labels']
        
        self.model_wrapper = ScalpingModel(config)
        self.training_history = None
        self.model_path = None
        
    def prepare_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        時系列データの分割（未来リーク防止）
        """
        logger.info("データ分割開始")
        
        # 時系列順で分割
        n_samples = len(X)
        train_slice, val_slice, test_slice = time_series_split(
            n_samples, 
            self.data_config['validation_split'],
            self.data_config['test_split']
        )
        
        X_train = X[train_slice]
        X_val = X[val_slice]
        X_test = X[test_slice]
        y_train = y[train_slice]
        y_val = y[val_slice]
        y_test = y[test_slice]
        
        logger.info(f"データ分割完了:")
        logger.info(f"  訓練: {X_train.shape[0]:,} サンプル")
        logger.info(f"  検証: {X_val.shape[0]:,} サンプル")
        logger.info(f"  テスト: {X_test.shape[0]:,} サンプル")
        
        # ラベル分布確認
        for split_name, y_split in [("訓練", y_train), ("検証", y_val), ("テスト", y_test)]:
            unique, counts = np.unique(y_split, return_counts=True)
            dist = dict(zip(unique, counts))
            logger.info(f"  {split_name}ラベル分布: {dist}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def calculate_class_weights(self, y_train: np.ndarray) -> Dict[int, float]:
        """
        クラス重みを計算（不均衡対策）
        """
        unique_classes = np.unique(y_train)
        class_weights_array = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=y_train
        )
        
        class_weights = dict(zip(unique_classes, class_weights_array))
        
        logger.info(f"クラス重み: {class_weights}")
        return class_weights
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   feature_names: List[str] = None) -> str:
        """
        モデル訓練実行
        """
        logger.info("モデル訓練開始")
        
        # モデル作成
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model_wrapper.create_model(input_shape)
        
        # モデルコンパイル
        self.model_wrapper.compile_model()
        
        # クラス重み計算
        class_weights = self.calculate_class_weights(y_train)
        
        # モデル保存パス
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        architecture = self.model_config['architecture']
        self.model_path = os.path.join(
            self.data_config['output_dir'],
            f"scalping_model_{architecture}_{timestamp}.h5"
        )
        
        # コールバック設定
        callbacks = self.model_wrapper.get_callbacks(self.model_path)
        
        # 訓練実行
        logger.info(f"訓練開始 - エポック数: {self.model_config['epochs']}")
        logger.info(f"バッチサイズ: {self.model_config['batch_size']}")
        logger.info(f"メモリ使用量: {memory_usage_mb():.1f}MB")
        
        history = self.model_wrapper.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.model_config['epochs'],
            batch_size=self.model_config['batch_size'],
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
            shuffle=False  # 時系列データなのでシャッフル禁止
        )
        
        self.training_history = history.history
        
        # メタデータ保存
        save_model_metadata(
            self.model_path,
            self.config,
            self.training_history,
            feature_names
        )
        
        logger.info(f"訓練完了 - モデル保存: {self.model_path}")
        return self.model_path
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        モデル評価
        """
        logger.info("モデル評価開始")
        
        if self.model_wrapper.model is None:
            raise ValueError("モデルが未訓練です")
        
        # 予測実行
        y_pred_proba = self.model_wrapper.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # 基本メトリクス（複数の戻り値に対応）
        eval_results = self.model_wrapper.model.evaluate(X_test, y_test, verbose=0)
        if isinstance(eval_results, list):
            test_loss = eval_results[0]
            test_accuracy = eval_results[1]  # 最初のaccuracyメトリクス
        else:
            test_loss = eval_results
            test_accuracy = np.mean(y_test == y_pred)  # 手動計算
        
        # 分類レポート
        class_names = self.labels_config['class_names']
        report = classification_report(
            y_test, y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        # 混同行列
        cm = confusion_matrix(y_test, y_pred)
        
        # F1スコア（各クラス別）
        f1_scores = {
            class_names[i]: f1_score(y_test, y_pred, labels=[i], average='macro', zero_division=0)
            for i in range(len(class_names))
        }
        
        # 閾値別評価
        threshold_results = self._evaluate_with_thresholds(X_test, y_test, y_pred_proba)
        
        evaluation_results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'f1_scores': f1_scores,
            'threshold_evaluation': threshold_results,
            'predictions': {
                'y_true': y_test.tolist(),
                'y_pred': y_pred.tolist(),
                'y_pred_proba': y_pred_proba.tolist()
            }
        }
        
        logger.info(f"評価完了 - テスト精度: {test_accuracy:.4f}")
        
        return evaluation_results
    
    def _evaluate_with_thresholds(self, X_test: np.ndarray, y_test: np.ndarray, 
                                 y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """
        様々な信頼度閾値での評価
        """
        thresholds = self.config['evaluation']['prediction_thresholds']
        threshold_results = {}
        
        for threshold in thresholds:
            # 閾値適用予測
            pred_filtered, confidences, _ = self.model_wrapper.predict_with_confidence(
                X_test, threshold
            )
            
            # 高信頼度サンプルのみで評価
            high_conf_mask = confidences >= threshold
            if high_conf_mask.sum() == 0:
                threshold_results[str(threshold)] = {
                    'num_predictions': 0,
                    'accuracy': 0.0,
                    'f1_score': 0.0,
                    'coverage': 0.0
                }
                continue
            
            y_test_filtered = y_test[high_conf_mask]
            y_pred_filtered = pred_filtered[high_conf_mask]
            
            accuracy = np.mean(y_test_filtered == y_pred_filtered)
            f1 = f1_score(y_test_filtered, y_pred_filtered, average='weighted', zero_division=0)
            coverage = high_conf_mask.sum() / len(y_test)
            
            threshold_results[str(threshold)] = {
                'num_predictions': int(high_conf_mask.sum()),
                'accuracy': float(accuracy),
                'f1_score': float(f1),
                'coverage': float(coverage)
            }
        
        return threshold_results
    
    def plot_training_history(self, save_path: str = None):
        """
        訓練履歴の可視化
        """
        if self.training_history is None:
            logger.warning("訓練履歴がありません")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16)
        
        # 損失
        axes[0, 0].plot(self.training_history['loss'], label='Training Loss')
        axes[0, 0].plot(self.training_history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 精度
        axes[0, 1].plot(self.training_history['accuracy'], label='Training Accuracy')
        axes[0, 1].plot(self.training_history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 学習率（もし記録されていれば）
        if 'lr' in self.training_history:
            axes[1, 0].plot(self.training_history['lr'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)
        else:
            axes[1, 0].text(0.5, 0.5, 'Learning Rate\nNot Available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # Precision/Recall
        if 'precision' in self.training_history:
            axes[1, 1].plot(self.training_history['precision'], label='Precision')
            axes[1, 1].plot(self.training_history['recall'], label='Recall')
            axes[1, 1].set_title('Precision & Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'Precision/Recall\nNot Available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"訓練履歴グラフ保存: {save_path}")
        
        plt.close()  # メモリリーク防止
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: str = None):
        """
        混同行列の可視化
        """
        plt.figure(figsize=(8, 6))
        
        class_names = self.labels_config['class_names']
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"混同行列保存: {save_path}")
        
        plt.close()  # メモリリーク防止
    
    def plot_threshold_analysis(self, threshold_results: Dict[str, Any], save_path: str = None):
        """
        閾値分析結果の可視化
        """
        thresholds = list(threshold_results.keys())
        accuracies = [threshold_results[t]['accuracy'] for t in thresholds]
        f1_scores = [threshold_results[t]['f1_score'] for t in thresholds]
        coverages = [threshold_results[t]['coverage'] for t in thresholds]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 精度とF1スコア
        ax1.plot(thresholds, accuracies, marker='o', label='Accuracy')
        ax1.plot(thresholds, f1_scores, marker='s', label='F1 Score')
        ax1.set_title('Performance vs Confidence Threshold')
        ax1.set_xlabel('Confidence Threshold')
        ax1.set_ylabel('Score')
        ax1.legend()
        ax1.grid(True)
        ax1.set_ylim(0, 1)
        
        # カバレッジ
        ax2.plot(thresholds, coverages, marker='D', color='green', label='Coverage')
        ax2.set_title('Coverage vs Confidence Threshold')
        ax2.set_xlabel('Confidence Threshold')
        ax2.set_ylabel('Coverage Ratio')
        ax2.legend()
        ax2.grid(True)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"閾値分析グラフ保存: {save_path}")
        
        plt.close()  # メモリリーク防止
    
    def save_evaluation_report(self, evaluation_results: Dict[str, Any], report_path: str):
        """
        評価結果をJSONで保存
        """
        import json
        
        # NumPy配列をリストに変換
        serializable_results = {}
        for key, value in evaluation_results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            else:
                serializable_results[key] = value
        
        # メタデータ追加
        serializable_results['evaluation_timestamp'] = datetime.now().isoformat()
        serializable_results['model_path'] = self.model_path
        serializable_results['config'] = self.config
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"評価レポート保存: {report_path}")
    
    def generate_trading_signals(self, X: np.ndarray, timestamps: List, 
                                min_confidence: float = None) -> pd.DataFrame:
        """
        取引シグナル生成（MT5連携用）
        """
        if min_confidence is None:
            min_confidence = self.config['evaluation']['min_confidence']
        
        # 予測実行
        predictions, confidences, probabilities = self.model_wrapper.predict_with_confidence(
            X, min_confidence
        )
        
        # シグナルデータフレーム作成
        signals = []
        class_names = self.labels_config['class_names']
        
        for i, (pred, conf, proba, timestamp) in enumerate(zip(predictions, confidences, probabilities, timestamps)):
            signal = {
                'timestamp': timestamp,
                'signal': class_names[pred],
                'confidence': float(conf),
                'buy_probability': float(proba[0]),
                'sell_probability': float(proba[1]),
                'no_trade_probability': float(proba[2]),
                'trade_recommended': conf >= min_confidence and pred != self.labels_config['no_trade_class']
            }
            signals.append(signal)
        
        signals_df = pd.DataFrame(signals)
        
        # 統計情報
        total_signals = len(signals_df)
        trade_signals = signals_df['trade_recommended'].sum()
        
        logger.info(f"シグナル生成完了:")
        logger.info(f"  総シグナル数: {total_signals:,}")
        logger.info(f"  取引推奨: {trade_signals:,} ({trade_signals/total_signals*100:.1f}%)")
        
        return signals_df