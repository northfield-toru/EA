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
import tensorflow as tf

from .model import ScalpingModel
from .utils import time_series_split, save_model_metadata, memory_usage_mb

logger = logging.getLogger(__name__)

class F1ScoreEarlyStopping(tf.keras.callbacks.Callback):
    """
    ChatGPT推奨: BUY クラスF1スコア監視のEarly Stopping
    SELL偏重問題の根本解決
    """
    
    def __init__(self, monitor_class='BUY', patience=7, min_delta=0.01, restore_best_weights=True, **kwargs):
        super().__init__(**kwargs)
        self.monitor_class = monitor_class
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.wait = 0
        self.best_f1 = 0
        self.best_weights = None
        
        logger.info(f"F1監視Early Stopping初期化: {monitor_class}クラス, patience={patience}")
    
    def on_train_begin(self, logs=None):
        self.wait = 0
        self.best_f1 = 0
        self.best_weights = None
    
    def on_epoch_end(self, epoch, logs=None):
        # 検証データでBUY F1スコア計算
        if hasattr(self.model, 'validation_data') and self.model.validation_data:
            val_x, val_y = self.validation_data[0], self.validation_data[1]
        else:
            # validation_dataが直接アクセスできない場合
            return
        
        val_pred = self.model.predict(val_x, verbose=0)
        val_pred_classes = np.argmax(val_pred, axis=1)
        val_true = val_y
        
        # BUY クラス（class=0）のF1スコア計算
        buy_mask_true = (val_true == 0)
        buy_mask_pred = (val_pred_classes == 0)
        
        if buy_mask_true.sum() > 0 and buy_mask_pred.sum() > 0:
            # Precision = TP / (TP + FP)
            precision = np.sum(buy_mask_true & buy_mask_pred) / buy_mask_pred.sum()
            # Recall = TP / (TP + FN)
            recall = np.sum(buy_mask_true & buy_mask_pred) / buy_mask_true.sum()
            
            if precision + recall > 0:
                f1_buy = 2 * (precision * recall) / (precision + recall)
            else:
                f1_buy = 0
        else:
            f1_buy = 0
        
        # ログ出力
        print(f"\nEpoch {epoch+1} - BUY F1: {f1_buy:.4f} (Best: {self.best_f1:.4f})")
        
        # 改善チェック
        if f1_buy > self.best_f1 + self.min_delta:
            self.best_f1 = f1_buy
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            print(f"✅ BUY F1改善: {f1_buy:.4f}")
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"\n🛑 BUY F1が{self.patience}エポック改善せず。Early Stopping実行。")
                if self.restore_best_weights and self.best_weights is not None:
                    self.model.set_weights(self.best_weights)
                    print("最良重みを復元しました。")
                self.model.stop_training = True

class ModelTrainer:
    """
    スキャルピングモデル訓練管理クラス（ChatGPT推奨強化版）
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config['model']
        self.data_config = config['data']
        self.labels_config = config['labels']
        
        self.model_wrapper = ScalpingModel(config)
        self.training_history = None
        self.model_path = None
        
        # ChatGPT推奨: SELL偏重検出フラグ
        self.sell_bias_detected = False
        
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
        
        # ラベル分布確認とSELL偏重検出
        for split_name, y_split in [("訓練", y_train), ("検証", y_val), ("テスト", y_test)]:
            unique, counts = np.unique(y_split, return_counts=True)
            dist = dict(zip(unique, counts))
            logger.info(f"  {split_name}ラベル分布: {dist}")
            
            # SELL偏重検出（訓練データで判定）
            if split_name == "訓練":
                total = len(y_split)
                sell_ratio = dist.get(1, 0) / total  # SELL=1
                buy_ratio = dist.get(0, 0) / total   # BUY=0
                
                if sell_ratio > 0.7:
                    self.sell_bias_detected = True
                    logger.warning(f"🚨 深刻なSELL偏重検出: SELL={sell_ratio:.1%}, BUY={buy_ratio:.1%}")
                elif sell_ratio > 0.6:
                    self.sell_bias_detected = True
                    logger.warning(f"⚠️ SELL偏重検出: SELL={sell_ratio:.1%}, BUY={buy_ratio:.1%}")
                else:
                    logger.info(f"✅ クラス分布良好: SELL={sell_ratio:.1%}, BUY={buy_ratio:.1%}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def calculate_enhanced_class_weights(self, y_train: np.ndarray) -> Dict[int, float]:
        """
        ChatGPT推奨の強化版クラス重み計算
        SELL偏重を完全に補正
        """
        unique_classes = np.unique(y_train)
        
        # 基本的な不均衡補正
        base_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
        
        # ラベル分布確認
        buy_count = np.sum(y_train == self.labels_config['buy_class'])
        sell_count = np.sum(y_train == self.labels_config['sell_class'])
        no_trade_count = np.sum(y_train == self.labels_config['no_trade_class'])
        
        total = len(y_train)
        buy_ratio = buy_count / total
        sell_ratio = sell_count / total
        no_trade_ratio = no_trade_count / total
        
        logger.info(f"詳細ラベル分布:")
        logger.info(f"  BUY: {buy_count:,} ({buy_ratio:.1%})")
        logger.info(f"  SELL: {sell_count:,} ({sell_ratio:.1%})")
        logger.info(f"  NO_TRADE: {no_trade_count:,} ({no_trade_ratio:.1%})")
        
        # ChatGPT推奨: SELL偏重レベルに応じた動的補正
        if sell_ratio > 0.8:
            # 極度のSELL偏重（緊急対策）
            sell_penalty = 0.3      # SELL重みを大幅削減
            buy_boost = 5.0         # BUY重みを5倍に増強
            no_trade_adjust = 1.2   # NO_TRADE軽微増強
            
            logger.warning("🚨 極度SELL偏重 - 緊急対策適用")
            
        elif sell_ratio > 0.7:
            # 深刻なSELL偏重
            sell_penalty = 0.4      # SELL重みを大幅削減
            buy_boost = 4.0         # BUY重みを4倍に増強
            no_trade_adjust = 1.0
            
            logger.warning("🚨 深刻SELL偏重 - 強力対策適用")
            
        elif sell_ratio > 0.6:
            # 中程度のSELL偏重
            sell_penalty = 0.6      # SELL重みを削減
            buy_boost = 2.5         # BUY重みを2.5倍に増強
            no_trade_adjust = 1.0
            
            logger.warning("⚠️ 中程度SELL偏重 - 標準対策適用")
            
        else:
            # 正常範囲 - 軽微な調整のみ
            sell_penalty = 0.8
            buy_boost = 1.5
            no_trade_adjust = 1.0
            
            logger.info("✅ 正常範囲 - 軽微補正のみ")
        
        # 最終クラス重み計算
        enhanced_weights = {}
        for i, class_idx in enumerate(unique_classes):
            base_weight = base_weights[i]
            
            if class_idx == self.labels_config['buy_class']:
                enhanced_weights[class_idx] = float(base_weight * buy_boost)
            elif class_idx == self.labels_config['sell_class']:
                enhanced_weights[class_idx] = float(base_weight * sell_penalty)
            elif class_idx == self.labels_config['no_trade_class']:
                enhanced_weights[class_idx] = float(base_weight * no_trade_adjust)
            else:
                enhanced_weights[class_idx] = float(base_weight)
        
        logger.info(f"最終クラス重み:")
        for class_idx, weight in enhanced_weights.items():
            class_name = self.labels_config['class_names'][class_idx]
            logger.info(f"  {class_name}: {weight:.3f}")
        
        return enhanced_weights
    
    def calculate_class_weights(self, y_train: np.ndarray) -> Dict[int, float]:
        """
        標準クラス重み計算（後方互換用）
        新しいenhanced版を使用することを推奨
        """
        if self.sell_bias_detected:
            logger.info("SELL偏重検出のため強化版クラス重みを使用")
            return self.calculate_enhanced_class_weights(y_train)
        else:
            # 標準的な balanced 重み
            unique_classes = np.unique(y_train)
            class_weights_array = compute_class_weight(
                'balanced',
                classes=unique_classes,
                y=y_train
            )
            
            class_weights = dict(zip(unique_classes, class_weights_array))
            logger.info(f"標準クラス重み: {class_weights}")
            return class_weights
    
    def get_enhanced_callbacks(self, model_path: str) -> List[tf.keras.callbacks.Callback]:
        """
        ChatGPT推奨の強化版コールバック
        F1スコア監視Early Stopping含む
        """
        callbacks_list = []
        
        # ChatGPT最重要推奨: BUY F1スコア監視Early Stopping
        if self.sell_bias_detected:
            f1_callback = F1ScoreEarlyStopping(
                monitor_class='BUY',
                patience=10,
                min_delta=0.005,  # より敏感に
                restore_best_weights=True
            )
            callbacks_list.append(f1_callback)
            logger.info("✅ BUY F1監視Early Stopping追加（SELL偏重対策）")
        
        # 標準Early Stopping（バックアップ）
        standard_early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',  # ChatGPT推奨: accuracyではなくloss監視
            patience=15,         # より長い忍耐
            restore_best_weights=True,
            min_delta=0.001,
            verbose=1
        )
        callbacks_list.append(standard_early_stopping)
        
        # Model Checkpoint
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks_list.append(checkpoint)
        
        # Learning Rate Reduction
        lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,          # より緩やかな削減
            patience=7,          # より長い忍耐
            min_lr=1e-8,
            verbose=1
        )
        callbacks_list.append(lr_reduction)
        
        # CSV Logger
        csv_logger = tf.keras.callbacks.CSVLogger(
            model_path.replace('.h5', '_training_log.csv'),
            append=True
        )
        callbacks_list.append(csv_logger)
        
        # ChatGPT推奨: クラス分布監視
        class_monitor = ClassDistributionMonitor(log_frequency=5)
        callbacks_list.append(class_monitor)
        
        logger.info(f"コールバック設定完了: {len(callbacks_list)}個")
        return callbacks_list
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   feature_names: List[str] = None) -> str:
        """
        モデル訓練実行（ChatGPT推奨強化版）
        """
        logger.info("🚀 強化版モデル訓練開始")
        
        # モデル作成
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model_wrapper.create_model(input_shape)
        
        # モデルコンパイル
        self.model_wrapper.compile_model()
        
        # 強化版クラス重み計算（ChatGPT推奨）
        class_weights = self.calculate_enhanced_class_weights(y_train)
        
        # モデル保存パス
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        architecture = self.model_config['architecture']
        self.model_path = os.path.join(
            self.data_config['output_dir'],
            f"scalping_model_{architecture}_{timestamp}.h5"
        )
        
        # 強化版コールバック設定
        callbacks = self.get_enhanced_callbacks(self.model_path)
        
        # 訓練実行
        logger.info(f"訓練パラメータ:")
        logger.info(f"  エポック数: {self.model_config['epochs']}")
        logger.info(f"  バッチサイズ: {self.model_config['batch_size']}")
        logger.info(f"  学習率: {self.model_config['learning_rate']}")
        logger.info(f"  メモリ使用量: {memory_usage_mb():.1f}MB")
        logger.info(f"  SELL偏重対策: {'有効' if self.sell_bias_detected else '無効'}")
        
        # ChatGPT最重要推奨: shuffle=True（時系列でもSELL偏重対策のため）
        shuffle_data = True if self.sell_bias_detected else False
        logger.info(f"  データシャッフル: {'有効' if shuffle_data else '無効'}（SELL偏重対策）")
        
        history = self.model_wrapper.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.model_config['epochs'],
            batch_size=self.model_config['batch_size'],
            class_weight=class_weights,  # ChatGPT推奨: 必須
            callbacks=callbacks,
            verbose=1,
            shuffle=shuffle_data  # ChatGPT推奨: SELL偏重時はTrue
        )
        
        # Callbackにvalidationデータをセットするためのハック
        for callback in callbacks:
            if hasattr(callback, 'validation_data'):
                callback.validation_data = (X_val, y_val)
        
        self.training_history = history.history
        
        # メタデータ保存（強化版情報も含む）
        metadata_info = {
            'sell_bias_detected': self.sell_bias_detected,
            'enhanced_class_weights': class_weights,
            'shuffle_enabled': shuffle_data
        }
        
        save_model_metadata(
            self.model_path,
            self.config,
            self.training_history,
            feature_names,
            scaling_params=getattr(self, 'scaling_params', None),
            enhanced_info=metadata_info
        )
        
        logger.info(f"✅ 訓練完了 - モデル保存: {self.model_path}")
        
        # 訓練結果サマリー
        final_epoch = len(self.training_history['loss'])
        final_acc = self.training_history['val_accuracy'][-1]
        logger.info(f"📊 訓練結果サマリー:")
        logger.info(f"  最終エポック: {final_epoch}")
        logger.info(f"  最終検証精度: {final_acc:.4f}")
        logger.info(f"  SELL偏重対策: {'適用済み' if self.sell_bias_detected else '適用なし'}")
        
        return self.model_path
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        モデル評価（ChatGPT推奨指標追加）
        """
        logger.info("📊 強化版モデル評価開始")
        
        if self.model_wrapper.model is None:
            raise ValueError("モデルが未訓練です")
        
        # 予測実行
        y_pred_proba = self.model_wrapper.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # 基本メトリクス
        eval_results = self.model_wrapper.model.evaluate(X_test, y_test, verbose=0)
        if isinstance(eval_results, list):
            test_loss = eval_results[0]
            test_accuracy = eval_results[1]
        else:
            test_loss = eval_results
            test_accuracy = np.mean(y_test == y_pred)
        
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
        
        # ChatGPT推奨: クラス別F1スコア（正確な計算）
        f1_scores = {}
        for i, class_name in enumerate(class_names):
            f1_class = f1_score(y_test, y_pred, labels=[i], average='macro', zero_division=0)
            f1_scores[class_name] = f1_class
        
        # ChatGPT推奨: SELL偏重診断
        sell_bias_analysis = self._analyze_sell_bias(y_pred, y_test)
        
        # 閾値別評価
        threshold_results = self._evaluate_with_thresholds(X_test, y_test, y_pred_proba)
        
        evaluation_results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'f1_scores': f1_scores,
            'sell_bias_analysis': sell_bias_analysis,  # 新規追加
            'threshold_evaluation': threshold_results,
            'predictions': {
                'y_true': y_test.tolist(),
                'y_pred': y_pred.tolist(),
                'y_pred_proba': y_pred_proba.tolist()
            }
        }
        
        # 評価結果ログ
        logger.info(f"📊 評価結果:")
        logger.info(f"  テスト精度: {test_accuracy:.4f}")
        logger.info(f"  テスト損失: {test_loss:.4f}")
        logger.info(f"  BUY F1: {f1_scores['BUY']:.4f}")
        logger.info(f"  SELL F1: {f1_scores['SELL']:.4f}")
        logger.info(f"  SELL偏重度: {sell_bias_analysis['sell_bias_severity']}")
        
        return evaluation_results
    
    def _analyze_sell_bias(self, y_pred: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        ChatGPT推奨: SELL偏重の詳細分析
        """
        total_predictions = len(y_pred)
        
        # 予測分布
        pred_counts = np.bincount(y_pred, minlength=3)
        buy_pred_ratio = pred_counts[0] / total_predictions
        sell_pred_ratio = pred_counts[1] / total_predictions
        no_trade_pred_ratio = pred_counts[2] / total_predictions
        
        # SELL偏重度判定
        if sell_pred_ratio > 0.8:
            severity = "極度"
            recommendation = "緊急対策必要"
        elif sell_pred_ratio > 0.7:
            severity = "深刻"
            recommendation = "強力対策必要"
        elif sell_pred_ratio > 0.6:
            severity = "中程度"
            recommendation = "標準対策推奨"
        else:
            severity = "正常"
            recommendation = "対策不要"
        
        # 実際のBUYをSELLと誤分類した率
        buy_mask = (y_test == 0)
        if buy_mask.sum() > 0:
            buy_to_sell_error = np.sum((y_test == 0) & (y_pred == 1)) / buy_mask.sum()
        else:
            buy_to_sell_error = 0
        
        analysis = {
            'sell_prediction_ratio': float(sell_pred_ratio),
            'buy_prediction_ratio': float(buy_pred_ratio),
            'no_trade_prediction_ratio': float(no_trade_pred_ratio),
            'sell_bias_severity': severity,
            'recommendation': recommendation,
            'buy_to_sell_error_rate': float(buy_to_sell_error),
            'prediction_counts': {
                'BUY': int(pred_counts[0]),
                'SELL': int(pred_counts[1]),
                'NO_TRADE': int(pred_counts[2])
            }
        }
        
        return analysis
    
    def _evaluate_with_thresholds(self, X_test: np.ndarray, y_test: np.ndarray, 
                                 y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """
        様々な信頼度閾値での評価（既存）
        """
        thresholds = self.config['evaluation']['prediction_thresholds']
        threshold_results = {}
        
        for threshold in thresholds:
            # 最大予測確率を取得
            max_confidences = np.max(y_pred_proba, axis=1)
            predicted_classes = np.argmax(y_pred_proba, axis=1)
            
            # 高信頼度サンプルのマスク
            high_conf_mask = max_confidences >= threshold
            
            if high_conf_mask.sum() == 0:
                threshold_results[str(threshold)] = {
                    'num_predictions': 0,
                    'accuracy': 0.0,
                    'f1_score': 0.0,
                    'coverage': 0.0
                }
                continue
            
            # 高信頼度サンプルでの評価
            y_test_filtered = y_test[high_conf_mask]
            y_pred_filtered = predicted_classes[high_conf_mask]
            
            # 低信頼度のものはNO_TRADEクラス(2)に変更
            y_pred_with_threshold = predicted_classes.copy()
            y_pred_with_threshold[~high_conf_mask] = self.labels_config['no_trade_class']
            
            # 全データでの評価（閾値適用後）
            accuracy = np.mean(y_test == y_pred_with_threshold)
            f1 = f1_score(y_test, y_pred_with_threshold, average='weighted', zero_division=0)
            coverage = high_conf_mask.sum() / len(y_test)
            
            threshold_results[str(threshold)] = {
                'num_predictions': int(high_conf_mask.sum()),
                'accuracy': float(accuracy),
                'f1_score': float(f1),
                'coverage': float(coverage),
                'high_conf_accuracy': float(np.mean(y_test_filtered == y_pred_filtered)) if len(y_test_filtered) > 0 else 0.0
            }
        
        return threshold_results
    
    # 以下、既存のplot系メソッドは変更なし
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
        閾値分析結果の可視化（文字化け修正版）
        """
        import matplotlib
        matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
        
        thresholds = list(threshold_results.keys())
        accuracies = [threshold_results[t]['accuracy'] for t in thresholds]
        f1_scores = [threshold_results[t]['f1_score'] for t in thresholds]
        coverages = [threshold_results[t]['coverage'] for t in thresholds]
        num_predictions = [threshold_results[t]['num_predictions'] for t in thresholds]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Threshold Analysis Results', fontsize=16, fontweight='bold')
        
        # 1. Accuracy and F1 Score
        ax1.plot(thresholds, accuracies, marker='o', linewidth=2, label='Accuracy', color='blue')
        ax1.plot(thresholds, f1_scores, marker='s', linewidth=2, label='F1 Score', color='orange')
        ax1.set_title('Performance vs Confidence Threshold')
        ax1.set_xlabel('Confidence Threshold')
        ax1.set_ylabel('Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # 2. Coverage
        ax2.plot(thresholds, coverages, marker='D', linewidth=2, color='green', label='Coverage')
        ax2.set_title('Coverage vs Confidence Threshold')
        ax2.set_xlabel('Confidence Threshold')
        ax2.set_ylabel('Coverage Ratio')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # 3. Trade Opportunities
        ax3.bar(thresholds, num_predictions, alpha=0.7, color='purple', edgecolor='black')
        ax3.set_title('Trade Opportunities vs Confidence Threshold')
        ax3.set_xlabel('Confidence Threshold')
        ax3.set_ylabel('Number of Trades')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Trade Rate (%)
        total_samples = max(num_predictions) if num_predictions else 1
        trade_rates = [count/total_samples*100 for count in num_predictions]
        
        ax4.bar(thresholds, trade_rates, alpha=0.7, color='red', edgecolor='black')
        ax4.set_title('Trade Rate vs Confidence Threshold')
        ax4.set_xlabel('Confidence Threshold')
        ax4.set_ylabel('Trade Rate (%)')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Threshold analysis plot saved: {save_path}")
        
        plt.close()
    
    def plot_order_distribution_by_class(self, evaluation_results: Dict[str, Any], save_path: str = None):
        """
        クラス別・閾値別のオーダー数分布を可視化（文字化け修正版）
        """
        import matplotlib
        matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
        
        threshold_results = evaluation_results['threshold_evaluation']
        y_pred_proba = np.array(evaluation_results['predictions']['y_pred_proba'])
        y_pred = np.array(evaluation_results['predictions']['y_pred'])
        
        thresholds = list(threshold_results.keys())
        class_names = self.labels_config['class_names']
        
        # 各閾値・各クラスでの予測数を計算
        distribution_data = []
        
        for threshold in thresholds:
            thresh_val = float(threshold)
            max_confidences = np.max(y_pred_proba, axis=1)
            high_conf_mask = max_confidences >= thresh_val
            
            high_conf_predictions = y_pred[high_conf_mask]
            
            for class_idx, class_name in enumerate(class_names):
                count = np.sum(high_conf_predictions == class_idx)
                distribution_data.append({
                    'threshold': threshold,
                    'class': class_name,
                    'count': count
                })
        
        df_dist = pd.DataFrame(distribution_data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Order Distribution by Class and Threshold', fontsize=16, fontweight='bold')
        
        # 1. Stacked Bar Chart
        pivot_data = df_dist.pivot(index='threshold', columns='class', values='count')
        pivot_data.plot(kind='bar', stacked=True, ax=ax1, 
                       color=['lightblue', 'lightcoral', 'lightgreen'])
        ax1.set_title('Stacked Order Distribution')
        ax1.set_xlabel('Confidence Threshold')
        ax1.set_ylabel('Number of Orders')
        ax1.legend(title='Class')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.tick_params(axis='x', rotation=0)
        
        # 2. Grouped Bar Chart
        pivot_data.plot(kind='bar', ax=ax2, 
                       color=['blue', 'red', 'green'], alpha=0.7)
        ax2.set_title('Grouped Order Distribution')
        ax2.set_xlabel('Confidence Threshold')
        ax2.set_ylabel('Number of Orders')
        ax2.legend(title='Class')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Order distribution plot saved: {save_path}")
        
        plt.close()
        
        return df_dist
    
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
        serializable_results['sell_bias_detected'] = self.sell_bias_detected
        
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

class ClassDistributionMonitor(tf.keras.callbacks.Callback):
    """
    ChatGPT推奨: 訓練中のクラス分布監視コールバック
    SELL偏重の早期発見
    """
    
    def __init__(self, log_frequency=5):
        super().__init__()
        self.log_frequency = log_frequency
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.log_frequency == 0:
            # 検証データで予測実行
            if hasattr(self.model, 'validation_data') and self.model.validation_data:
                val_x, val_y = self.validation_data[0], self.validation_data[1]
                predictions = self.model.predict(val_x, verbose=0)
                pred_classes = np.argmax(predictions, axis=1)
                
                # クラス分布計算
                unique, counts = np.unique(pred_classes, return_counts=True)
                total = len(pred_classes)
                
                distribution = {}
                class_names = ['BUY', 'SELL', 'NO_TRADE']
                for i, name in enumerate(class_names):
                    count = counts[unique == i][0] if i in unique else 0
                    distribution[name] = f"{count/total*100:.1f}%"
                
                print(f"\nEpoch {epoch+1} - 予測分布: {distribution}")
                
                # SELL偏重警告
                sell_ratio = counts[unique == 1][0] / total if 1 in unique else 0
                if sell_ratio > 0.8:
                    print(f"🚨 SELL偏重警告: {sell_ratio:.1%}")
                elif sell_ratio > 0.7:
                    print(f"⚠️ SELL偏重注意: {sell_ratio:.1%}")

# 使用方法の例
def create_enhanced_trainer(config):
    """
    強化版トレーナーの作成例
    """
    trainer = ModelTrainer(config)
    return trainer