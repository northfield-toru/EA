"""
USDJPY スキャルピングEA用 AIモデル定義 - 修正版
CNN + LSTM ハイブリッドアーキテクチャ
2値分類（NO_TRADE, TRADE）をデフォルト化
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, metrics
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, RobustScaler

class ScalpingCNNLSTM:
    """スキャルピング用CNN+LSTMモデル（2値分類デフォルト）"""
    
    def __init__(self, 
                 sequence_length: int = 30,
                 n_features: int = 82,         # 修正: 87 → 82
                 n_classes: int = 2,           # 修正: 3 → 2（デフォルト2値分類）
                 cnn_filters: list = [16, 32],
                 kernel_sizes: list = [3, 5],
                 lstm_units: int = 32,
                 dropout_rate: float = 0.5,
                 learning_rate: float = 0.001):
        """
        Args:
            sequence_length: 時系列長（デフォルト30分）
            n_features: 特徴量数（82）
            n_classes: クラス数（デフォルト2=NO_TRADE/TRADE）
            cnn_filters: CNNフィルタ数のリスト
            kernel_sizes: CNNカーネルサイズのリスト
            lstm_units: LSTMユニット数
            dropout_rate: ドロップアウト率
            learning_rate: 学習率
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_classes = n_classes
        self.cnn_filters = cnn_filters
        self.kernel_sizes = kernel_sizes
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model = None
        self.keras_model = None  # 追加：Kerasモデル直接参照用
        self.scaler = None
        self.class_weights = None
        self.history = None
        
        print(f"モデル初期化: seq_len={sequence_length}, features={n_features}, classes={n_classes}")
    
    def build_model(self) -> models.Model:
        """
        軽量CNN + LSTM モデルを構築
        """
        print("モデル構築開始...")
        
        # 入力層
        input_layer = layers.Input(
            shape=(self.sequence_length, self.n_features),
            name='input_layer'
        )
        
        # 軽量Conv1D層
        x = layers.Conv1D(
            filters=16,
            kernel_size=3,
            activation='relu',
            padding='same',
            name='conv1d_1'
        )(input_layer)
        
        x = layers.BatchNormalization(name='bn_conv1')(x)
        x = layers.MaxPooling1D(pool_size=2, name='pool1')(x)
        x = layers.Dropout(self.dropout_rate * 0.5, name='dropout_conv1')(x)
        
        # 第2 Conv1D層
        x = layers.Conv1D(
            filters=32,
            kernel_size=5,
            activation='relu',
            padding='same',
            name='conv1d_2'
        )(x)
        
        x = layers.BatchNormalization(name='bn_conv2')(x)
        x = layers.MaxPooling1D(pool_size=2, name='pool2')(x)
        x = layers.Dropout(self.dropout_rate * 0.5, name='dropout_conv2')(x)
        
        # LSTM層
        x = layers.LSTM(
            units=self.lstm_units,
            return_sequences=False,
            dropout=self.dropout_rate * 0.5,
            recurrent_dropout=self.dropout_rate * 0.5,
            name='lstm_layer'
        )(x)
        
        # Dense層
        x = layers.Dense(
            32,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            name='dense_1'
        )(x)
        
        x = layers.BatchNormalization(name='bn_dense1')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_dense1')(x)
        
        # 第2 Dense層
        x = layers.Dense(
            16,
            activation='relu',
            name='dense_2'
        )(x)
        
        x = layers.Dropout(self.dropout_rate * 0.5, name='dropout_dense2')(x)
        
        # 出力層（2値分類用）
        output_layer = layers.Dense(
            self.n_classes,
            activation='softmax',
            name='output_layer'
        )(x)
        
        # モデル作成
        model = models.Model(
            inputs=input_layer,
            outputs=output_layer,
            name='ScalpingCNNLSTM_Binary'
        )
        
        # コンパイル
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                metrics.Precision(name='precision'),
                metrics.Recall(name='recall')
            ]
        )
        
        self.model = model
        self.keras_model = model  # 追加：直接参照用
        
        print("モデル構築完了")
        print(f"パラメータ数: {model.count_params():,}")
        
        return model
    
    def prepare_sequences(self, features_df: pd.DataFrame, labels: Optional[pd.Series] = None) -> Tuple:
        """
        時系列シーケンスデータを準備（2値分類対応）
        """
        print(f"シーケンス準備開始: {len(features_df)} 行")
        
        # 数値型列のみ選択
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        features_numeric = features_df[numeric_columns].copy()
        
        print(f"数値特徴量: {len(numeric_columns)} 列")
        
        # 欠損値処理
        features_clean = features_numeric.fillna(method='ffill').fillna(method='bfill')
        
        # 無限値を除去
        features_clean = features_clean.replace([np.inf, -np.inf], np.nan)
        features_clean = features_clean.fillna(0)
        
        # 特徴量スケーリング
        if self.scaler is None:
            self.scaler = RobustScaler()
            scaled_features = self.scaler.fit_transform(features_clean)
        else:
            scaled_features = self.scaler.transform(features_clean)
        
        # シーケンス作成
        sequences = []
        sequence_labels = []
        
        for i in range(self.sequence_length, len(scaled_features)):
            seq = scaled_features[i-self.sequence_length:i]
            sequences.append(seq)
            
            if labels is not None:
                sequence_labels.append(labels.iloc[i])
        
        X = np.array(sequences)
        
        print(f"シーケンス作成完了: {X.shape}")
        
        if labels is not None:
            y = np.array(sequence_labels)
            
            # ラベル分布確認
            unique, counts = np.unique(y, return_counts=True)
            label_dist = dict(zip(unique, counts))
            print(f"ラベル分布: {label_dist}")
            
            # 2値分類のラベル検証
            if self.n_classes == 2:
                # 0と1のみを許可
                if not all(label in [0, 1] for label in unique):
                    print(f"⚠️ 警告: 2値分類なのに予期しないラベル値: {unique}")
                    # 2より大きい値は1に変換
                    y = np.where(y >= 1, 1, 0)
                    unique, counts = np.unique(y, return_counts=True)
                    label_dist = dict(zip(unique, counts))
                    print(f"修正後ラベル分布: {label_dist}")
            
            # One-hot エンコーディング
            y_categorical = to_categorical(y, num_classes=self.n_classes)
            
            return X, y_categorical, y
        
        return X
    
    def calculate_class_weights(self, y: np.array) -> Dict:
        """
        クラス重み計算（2値分類特化）
        """
        # 基本的なbalanced重み
        unique_classes = np.unique(y)
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=y
        )
        
        class_weight_dict = dict(zip(unique_classes, class_weights))
        
        # 2値分類の場合の特別処理
        if self.n_classes == 2:
            label_counts = np.bincount(y, minlength=2)
            total_samples = len(y)
            
            # NO_TRADE (0) vs TRADE (1) のバランス調整
            no_trade_count = label_counts[0]
            trade_count = label_counts[1]
            
            if trade_count > 0:
                # TRADE比率が30%未満の場合、TRADEクラスを強化
                trade_ratio = trade_count / total_samples
                if trade_ratio < 0.3:
                    enhancement_factor = 1.5
                    class_weight_dict[1] *= enhancement_factor
                    print(f"TRADE比率{trade_ratio:.1%}につき、TRADE重みを{enhancement_factor}倍強化")
        
        print(f"最終クラス重み: {class_weight_dict}")
        print(f"クラス分布: {dict(zip(range(len(np.bincount(y))), np.bincount(y)))}")
        
        self.class_weights = class_weight_dict
        return class_weight_dict
    
    def get_callbacks(self, 
                     model_save_path: str = 'best_model.h5',
                     patience: int = 5,
                     reduce_lr_patience: int = 3) -> list:
        """学習用コールバック設定"""
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            callbacks.ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=reduce_lr_patience,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks_list
    
    def train(self, 
              X_train: np.array, 
              y_train: np.array,
              X_val: np.array,
              y_val: np.array,
              epochs: int = 100,
              batch_size: int = 64,
              **kwargs) -> Dict:
        """
        モデル学習（2値分類対応）
        """
        if self.model is None:
            self.build_model()
        
        print("学習開始...")
        print(f"学習データ: {X_train.shape}, 検証データ: {X_val.shape}")
        
        # コールバック取得
        callbacks_list = self.get_callbacks(**kwargs)
        
        # 学習実行
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            class_weight=self.class_weights,
            verbose=1
        )
        
        self.history = history.history
        
        print("学習完了")
        return self.history
    
    def predict(self, X: np.array) -> Tuple[np.array, np.array]:
        """予測実行"""
        if self.model is None:
            raise ValueError("モデルが構築されていません")
        
        predictions_proba = self.model.predict(X, verbose=0)
        predictions_class = np.argmax(predictions_proba, axis=1)
        
        return predictions_proba, predictions_class
    
    def save(self, filepath: str):
        """モデル保存（keras_model使用）"""
        if self.keras_model is not None:
            self.keras_model.save(filepath)
            print(f"モデル保存完了: {filepath}")
        else:
            raise ValueError("保存するモデルがありません")
    
    def load_model(self, filepath: str):
        """モデル読み込み"""
        self.model = models.load_model(filepath)
        self.keras_model = self.model
        print(f"モデル読み込み完了: {filepath}")


def create_lightweight_model(sequence_length: int = 30, n_features: int = 82, n_classes: int = 2) -> ScalpingCNNLSTM:
    """
    軽量化モデル作成（2値分類デフォルト）
    """
    model = ScalpingCNNLSTM(
        sequence_length=sequence_length,
        n_features=n_features,
        n_classes=n_classes,  # 明示的に2値分類
        cnn_filters=[16, 32],
        kernel_sizes=[3, 5],
        lstm_units=32,
        dropout_rate=0.5,
        learning_rate=0.001
    )
    
    model.build_model()
    
    param_count = model.model.count_params()
    print(f"軽量モデル構築完了")
    print(f"  パラメータ数: {param_count:,}")
    print(f"  クラス数: {n_classes} ({'2値分類(NO_TRADE/TRADE)' if n_classes == 2 else '3値分類'})")
    
    return model


def create_sample_model(sequence_length: int = 30, n_features: int = 82) -> ScalpingCNNLSTM:
    """サンプルモデル作成（2値分類固定）"""
    return create_lightweight_model(sequence_length, n_features, n_classes=2)


class FocalLoss:
    """Focal Loss（不均衡データ対策）"""
    
    def __init__(self, alpha=1.0, gamma=2.0):
        self.alpha = alpha
        self.gamma = gamma
    
    def __call__(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        ce_loss = -y_true * tf.math.log(y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_loss = self.alpha * tf.pow(1 - p_t, self.gamma) * ce_loss
        
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=1))


if __name__ == "__main__":
    print("=== スキャルピングCNN+LSTMモデル テスト（2値分類版） ===")
    
    try:
        # 2値分類モデル作成
        model = create_sample_model()
        
        # モデルサマリー表示
        if hasattr(model.model, 'summary'):
            model.model.summary()
        
        # サンプルデータでテスト
        batch_size = 32
        sequence_length = 30
        n_features = 82
        
        # ダミーデータ作成
        X_sample = np.random.randn(batch_size, sequence_length, n_features)
        
        # 予測テスト
        pred_proba, pred_class = model.predict(X_sample)
        
        print(f"\n予測テスト完了:")
        print(f"入力形状: {X_sample.shape}")
        print(f"予測確率形状: {pred_proba.shape}")
        print(f"予測クラス形状: {pred_class.shape}")
        print(f"予測クラス例: {pred_class[:5]}")
        print(f"予測確率例: {pred_proba[:3]}")
        
        # クラス分布確認
        unique, counts = np.unique(pred_class, return_counts=True)
        print(f"予測クラス分布: {dict(zip(unique, counts))}")
        
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()