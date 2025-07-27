import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import json
from datetime import datetime

from .utils import calculate_class_weights, save_class_weights, plot_threshold_analysis, memory_usage_check
from .model import ForexModelBuilder

class ModelTrainer:
    """モデル訓練クラス"""
    
    def __init__(self, config, log_dir):
        self.config = config
        self.log_dir = log_dir
        self.model_config = config['model']
        self.logger = logging.getLogger(__name__)
        
        # メモリ使用量チェック
        memory_usage_check()
        
    def train_model(self, X, y, feature_names):
        """
        モデルを訓練
        
        Args:
            X: 特徴量配列 (samples, timesteps, features)
            y: ラベル配列 (samples,)
            feature_names: 特徴量名リスト
            
        Returns:
            dict: 訓練結果
        """
        self.logger.info(f"Starting model training with data shape: X={X.shape}, y={y.shape}")
        
        # データ分割
        X_train, X_val, y_train, y_val = self._split_data(X, y)
        
        # クラス重み計算
        class_weights = calculate_class_weights(y_train)
        save_class_weights(class_weights, self.log_dir)
        
        # モデル構築
        model_builder = ForexModelBuilder(self.config)
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = model_builder.build_model(input_shape)
        
        # コールバック作成
        callbacks = model_builder.create_callbacks(self.log_dir)
        
        # 訓練実行
        history = self._train_model(model, X_train, y_train, X_val, y_val, class_weights, callbacks)
        
        # モデル保存
        model_path = self._save_model(model)
        
        # 評価実行
        metrics = self._evaluate_model(model, X_val, y_val, feature_names)
        
        # 訓練履歴プロット
        self._plot_training_history(history)
        
        # 結果まとめ
        results = {
            'model': model,
            'model_path': model_path,
            'history': history,
            'metrics': metrics,
            'class_weights': class_weights,
            'feature_names': feature_names,
            'input_shape': input_shape
        }
        
        self.logger.info("Model training completed successfully")
        return results
    
    def _split_data(self, X, y):
        """データ分割（時系列順維持）"""
        validation_split = self.model_config.get('validation_split', 0.2)
        
        # 時系列データなので末尾を検証用に使用
        split_idx = int(len(X) * (1 - validation_split))
        
        X_train = X[:split_idx]
        X_val = X[split_idx:]
        y_train = y[:split_idx]
        y_val = y[split_idx:]
        
        self.logger.info(f"Data split - Train: {X_train.shape[0]}, Validation: {X_val.shape[0]}")
        
        # データ分布確認
        train_unique, train_counts = np.unique(y_train, return_counts=True)
        val_unique, val_counts = np.unique(y_val, return_counts=True)
        
        self.logger.info("Training set distribution:")
        for label, count in zip(train_unique, train_counts):
            self.logger.info(f"  Class {label}: {count} ({count/len(y_train)*100:.1f}%)")
        
        self.logger.info("Validation set distribution:")
        for label, count in zip(val_unique, val_counts):
            self.logger.info(f"  Class {label}: {count} ({count/len(y_val)*100:.1f}%)")
        
        return X_train, X_val, y_train, y_val
    
    def _train_model(self, model, X_train, y_train, X_val, y_val, class_weights, callbacks):
        """モデル訓練実行"""
        batch_size = self.model_config.get('batch_size', 512)
        epochs = self.model_config.get('epochs', 100)
        
        self.logger.info(f"Training parameters - Batch size: {batch_size}, Epochs: {epochs}")
        
        # メモリ効率のためのジェネレーター使用を検討
        if len(X_train) > 100000:  # 大容量データの場合
            self.logger.info("Using data generator for large dataset")
            history = self._train_with_generator(model, X_train, y_train, X_val, y_val, 
                                               class_weights, callbacks, batch_size, epochs)
        else:
            # 通常の訓練
            try:
                history = model.fit(
                    X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_val, y_val),
                    class_weight=class_weights,
                    callbacks=callbacks,
                    verbose=1,
                    shuffle=False  # 時系列データなのでシャッフルしない
                )
            except tf.errors.ResourceExhaustedError:
                self.logger.warning("GPU memory exhausted, reducing batch size")
                batch_size = batch_size // 2
                history = model.fit(
                    X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_val, y_val),
                    class_weight=class_weights,
                    callbacks=callbacks,
                    verbose=1,
                    shuffle=False
                )
        
        return history
    
    def _train_with_generator(self, model, X_train, y_train, X_val, y_val, 
                            class_weights, callbacks, batch_size, epochs):
        """大容量データ用ジェネレーター訓練"""
        
        def data_generator(X, y, batch_size, class_weights):
            """データジェネレーター"""
            indices = np.arange(len(X))
            while True:
                for start_idx in range(0, len(X), batch_size):
                    end_idx = min(start_idx + batch_size, len(X))
                    batch_indices = indices[start_idx:end_idx]
                    
                    X_batch = X[batch_indices]
                    y_batch = y[batch_indices]
                    
                    # サンプル重み計算
                    sample_weights = np.array([class_weights[label] for label in y_batch])
                    
                    yield X_batch, y_batch, sample_weights
        
        # ジェネレーター作成
        train_gen = data_generator(X_train, y_train, batch_size, class_weights)
        val_gen = data_generator(X_val, y_val, batch_size, {0: 1.0, 1: 1.0, 2: 1.0})
        
        steps_per_epoch = len(X_train) // batch_size
        validation_steps = len(X_val) // batch_size
        
        # 訓練実行
        history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def _save_model(self, model):
        """モデル保存"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        model_filename = f"forex_model_{timestamp}.h5"
        model_path = os.path.join('models', model_filename)
        
        # modelsディレクトリ作成
        os.makedirs('models', exist_ok=True)
        
        # モデル保存
        model.save(model_path)
        
        # ログディレクトリにもコピー保存
        log_model_path = os.path.join(self.log_dir, 'model.h5')
        model.save(log_model_path)
        
        self.logger.info(f"Model saved to: {model_path}")
        return model_path
    
    def _evaluate_model(self, model, X_val, y_val, feature_names):
        """モデル評価"""
        self.logger.info("Evaluating model...")
        
        # 予測実行
        y_pred_proba = model.predict(X_val, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # 基本メトリクス
        accuracy = np.mean(y_pred == y_val)
        f1_macro = f1_score(y_val, y_pred, average='macro')
        f1_per_class = f1_score(y_val, y_pred, average=None)
        
        # 分類レポート
        target_names = ['BUY', 'SELL', 'NO_TRADE']
        class_report = classification_report(y_val, y_pred, target_names=target_names, output_dict=True)
        
        # 混同行列
        cm = confusion_matrix(y_val, y_pred)
        
        # 閾値分析
        threshold_results = plot_threshold_analysis(y_val, y_pred_proba, self.log_dir)
        
        # 予測分布分析
        pred_distribution = self._analyze_prediction_distribution(y_val, y_pred, y_pred_proba)
        
        # 混同行列プロット
        self._plot_confusion_matrix(cm, target_names)
        
        # 分類レポート保存
        self._save_classification_report(class_report)
        
        # メトリクス辞書作成
        metrics = {
            'val_accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_buy': float(f1_per_class[0]),
            'f1_sell': float(f1_per_class[1]),
            'f1_no_trade': float(f1_per_class[2]),
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'prediction_distribution': pred_distribution,
            'buy_ratio': float(np.sum(y_pred == 0) / len(y_pred)),
            'sell_ratio': float(np.sum(y_pred == 1) / len(y_pred)),
            'no_trade_ratio': float(np.sum(y_pred == 2) / len(y_pred))
        }
        
        # 評価結果ログ出力
        self.logger.info(f"Validation Accuracy: {accuracy:.4f}")
        self.logger.info(f"Macro F1 Score: {f1_macro:.4f}")
        self.logger.info(f"F1 Scores - BUY: {f1_per_class[0]:.4f}, SELL: {f1_per_class[1]:.4f}, NO_TRADE: {f1_per_class[2]:.4f}")
        
        return metrics
    
    def _analyze_prediction_distribution(self, y_true, y_pred, y_pred_proba):
        """予測分布分析"""
        analysis = {}
        
        # クラス別信頼度分析
        for class_idx, class_name in enumerate(['BUY', 'SELL', 'NO_TRADE']):
            class_mask = y_pred == class_idx
            if np.sum(class_mask) > 0:
                class_confidences = y_pred_proba[class_mask, class_idx]
                analysis[f'{class_name}_confidence_mean'] = float(np.mean(class_confidences))
                analysis[f'{class_name}_confidence_std'] = float(np.std(class_confidences))
                analysis[f'{class_name}_confidence_min'] = float(np.min(class_confidences))
                analysis[f'{class_name}_confidence_max'] = float(np.max(class_confidences))
        
        # 全体的な信頼度分布
        max_confidences = np.max(y_pred_proba, axis=1)
        analysis['overall_confidence_mean'] = float(np.mean(max_confidences))
        analysis['overall_confidence_std'] = float(np.std(max_confidences))
        
        # 低信頼度予測の割合
        low_confidence_threshold = 0.5
        low_confidence_ratio = np.sum(max_confidences < low_confidence_threshold) / len(max_confidences)
        analysis['low_confidence_ratio'] = float(low_confidence_ratio)
        
        return analysis
    
    def _plot_confusion_matrix(self, cm, target_names):
        """混同行列プロット"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Confusion matrix plot saved")
    
    def _save_classification_report(self, class_report):
        """分類レポート保存"""
        # JSON形式で保存
        report_path = os.path.join(self.log_dir, 'classification_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(class_report, f, indent=2, ensure_ascii=False)
        
        # テキスト形式でも保存
        target_names = ['BUY', 'SELL', 'NO_TRADE']
        from sklearn.metrics import classification_report
        
        # 再度生成（テキスト形式用）
        # この部分は実際の予測結果が必要なので、呼び出し元で実装
        
        self.logger.info("Classification report saved")
    
    def _plot_training_history(self, history):
        """訓練履歴プロット"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(history.history['loss'], label='Training Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning Rate (if available)
        if 'lr' in history.history:
            axes[1, 0].plot(history.history['lr'], label='Learning Rate')
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Learning Rate\nNot Available', 
                          ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # Additional metrics
        if 'sparse_categorical_accuracy' in history.history:
            axes[1, 1].plot(history.history['sparse_categorical_accuracy'], 
                          label='Training Sparse Accuracy')
            axes[1, 1].plot(history.history['val_sparse_categorical_accuracy'], 
                          label='Validation Sparse Accuracy')
            axes[1, 1].set_title('Sparse Categorical Accuracy')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Additional Metrics\nNot Available', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 履歴データをCSVでも保存
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(os.path.join(self.log_dir, 'training_history_data.csv'), index=False)
        
        self.logger.info("Training history plots saved")