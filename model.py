"""
USDJPY スキャルピングEA用 AIモデル定義
CNN + LSTM ハイブリッドアーキテクチャ
3クラス分類（NO_TRADE, BUY, SELL）
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
    """スキャルピング用CNN+LSTMモデル"""
    
    def __init__(self, 
                 sequence_length: int = 30,
                 n_features: int = 87,
                 n_classes: int = 3,
                 cnn_filters: list = [16, 32],  # 軽量化: [32,64,128] → [16,32]
                 kernel_sizes: list = [3, 5],  # 軽量化: [3,5,7] → [3,5]
                 lstm_units: int = 32,         # 軽量化: 64 → 32
                 dropout_rate: float = 0.5,    # 強化: 0.3 → 0.5
                 learning_rate: float = 0.001):
        """
        Args:
            sequence_length: 時系列長（デフォルト30分）
            n_features: 特徴量数
            n_classes: クラス数（NO_TRADE, BUY, SELL）
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
        self.scaler = None
        self.class_weights = None
        self.history = None
        
        print(f"モデル初期化: seq_len={sequence_length}, features={n_features}, classes={n_classes}")
    
    def _build_multi_kernel_cnn(self, input_layer):
        """
        複数カーネルサイズのCNN層を構築
        Args:
            input_layer: 入力層
        Returns:
            Tensor: 結合されたCNN出力
        """
        cnn_outputs = []
        
        for kernel_size in self.kernel_sizes:
            # 各カーネルサイズでCNN処理
            x = input_layer
            
            for i, filters in enumerate(self.cnn_filters):
                x = layers.Conv1D(
                    filters=filters,
                    kernel_size=kernel_size,
                    activation='relu',
                    padding='same',
                    name=f'conv1d_k{kernel_size}_layer{i+1}'
                )(x)
                
                x = layers.BatchNormalization(
                    name=f'bn_k{kernel_size}_layer{i+1}'
                )(x)
                
                if i < len(self.cnn_filters) - 1:  # 最後以外はプーリング
                    x = layers.MaxPooling1D(
                        pool_size=2,
                        name=f'pool_k{kernel_size}_layer{i+1}'
                    )(x)
                
                x = layers.Dropout(
                    self.dropout_rate,
                    name=f'dropout_k{kernel_size}_layer{i+1}'
                )(x)
            
            cnn_outputs.append(x)
        
        # 複数カーネルの出力を結合
        if len(cnn_outputs) > 1:
            merged_cnn = layers.Concatenate(axis=-1, name='multi_kernel_concat')(cnn_outputs)
        else:
            merged_cnn = cnn_outputs[0]
        
        return merged_cnn
    
    def _build_attention_layer(self, lstm_output):
        """
        軽量Attention機構
        Args:
            lstm_output: LSTMの出力
        Returns:
            Tensor: Attention適用後の出力
        """
        # Self-Attention（簡易版）
        attention_weights = layers.Dense(
            self.lstm_units,
            activation='tanh',
            name='attention_dense'
        )(lstm_output)
        
        attention_weights = layers.Dense(
            1,
            activation='softmax',
            name='attention_weights'
        )(attention_weights)
        
        # 重み付き平均
        attended_output = layers.Multiply(
            name='attention_multiply'
        )([lstm_output, attention_weights])
        
        attended_output = layers.GlobalAveragePooling1D(
            name='attention_pooling'
        )(attended_output)
        
        return attended_output
    
    def build_model(self) -> models.Model:
        """
        CNN + LSTM + Attention ハイブリッドモデルを構築
        Returns:
            Model: 構築されたモデル
        """
        print("モデル構築開始...")
        
        # 入力層
        input_layer = layers.Input(
            shape=(self.sequence_length, self.n_features),
            name='input_layer'
        )
        
        # マルチカーネルCNN層
        cnn_output = self._build_multi_kernel_cnn(input_layer)
        
        # LSTM層（双方向）
        lstm_output = layers.Bidirectional(
            layers.LSTM(
                self.lstm_units,
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate
            ),
            name='bidirectional_lstm'
        )(cnn_output)
        
        # Attention機構
        attended_output = self._build_attention_layer(lstm_output)
        
        # 全結合層（軽量化）
        dense_output = layers.Dense(
            32,  # 軽量化: 128 → 32
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001),  # L2正則化追加
            name='dense_1'
        )(attended_output)
        
        dense_output = layers.BatchNormalization(name='bn_dense')(dense_output)
        dense_output = layers.Dropout(self.dropout_rate, name='dropout_dense')(dense_output)
        
        # 中間層削除で軽量化
        
        # 出力層（3クラス分類）
        output_layer = layers.Dense(
            self.n_classes,
            activation='softmax',
            name='output_layer'
        )(dense_output)
        
        # モデル作成
        model = models.Model(
            inputs=input_layer,
            outputs=output_layer,
            name='ScalpingCNNLSTM'
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
        
        print("モデル構築完了")
        print(f"パラメータ数: {model.count_params():,}")
        
        return model
    
    def prepare_sequences(self, features_df: pd.DataFrame, labels: Optional[pd.Series] = None) -> Tuple:
        """
        時系列シーケンスデータを準備
        Args:
            features_df: 特徴量DataFrame
            labels: ラベル（学習時のみ）
        Returns:
            tuple: (X, y) または X のみ
        """
        print(f"シーケンス準備開始: {len(features_df)} 行")
        
        # 数値型列のみ選択（文字列型列を除外）
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
            self.scaler = RobustScaler()  # 外れ値に強い
            scaled_features = self.scaler.fit_transform(features_clean)
        else:
            scaled_features = self.scaler.transform(features_clean)
        
        # シーケンス作成
        sequences = []
        sequence_labels = []
        
        for i in range(self.sequence_length, len(scaled_features)):
            # 過去30ステップの特徴量
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
            
            # One-hot エンコーディング
            y_categorical = to_categorical(y, num_classes=self.n_classes)
            
            return X, y_categorical, y
        
        return X
    
    def calculate_class_weights(self, y: np.array) -> Dict:
        """
        クラス重み計算（不均衡データ対策強化版）
        Args:
            y: ラベル配列
        Returns:
            dict: クラス重み
        """
        # 基本的なbalanced重み
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y),
            y=y
        )
        
        class_weight_dict = dict(zip(np.unique(y), class_weights))
        
        # 極端な不均衡の場合、追加調整
        label_counts = np.bincount(y)
        max_count = np.max(label_counts)
        min_count = np.min(label_counts[label_counts > 0])
        
        # 不均衡比が10:1以上の場合、少数クラスをさらに強化
        if max_count / min_count > 10:
            for class_idx, count in enumerate(label_counts):
                if count > 0 and count == min_count:
                    class_weight_dict[class_idx] *= 1.5  # 1.5倍強化
                    print(f"極端不均衡検出: クラス{class_idx}の重みを{class_weight_dict[class_idx]:.2f}に強化")
        
        print(f"最終クラス重み: {class_weight_dict}")
        print(f"クラス分布: {dict(zip(range(len(label_counts)), label_counts))}")
        
        self.class_weights = class_weight_dict
        return class_weight_dict
    
    def get_callbacks(self, 
                     model_save_path: str = 'best_model.h5',
                     patience: int = 5,        # 厳格化: 10 → 5
                     reduce_lr_patience: int = 3) -> list:  # 厳格化: 5 → 3
        """
        学習用コールバック設定
        Args:
            model_save_path: モデル保存パス
            patience: EarlyStopping待機回数
            reduce_lr_patience: 学習率削減待機回数
        Returns:
            list: コールバックリスト
        """
        callbacks_list = [
            # EarlyStopping
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # ModelCheckpoint
            callbacks.ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            
            # ReduceLROnPlateau
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
        モデル学習
        Args:
            X_train: 学習用特徴量
            y_train: 学習用ラベル（categorical）
            X_val: 検証用特徴量
            y_val: 検証用ラベル（categorical）
            epochs: エポック数
            batch_size: バッチサイズ
        Returns:
            dict: 学習履歴
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
        """
        予測実行
        Args:
            X: 入力データ
        Returns:
            tuple: (予測確率, 予測クラス)
        """
        if self.model is None:
            raise ValueError("モデルが構築されていません")
        
        predictions_proba = self.model.predict(X, verbose=0)
        predictions_class = np.argmax(predictions_proba, axis=1)
        
        return predictions_proba, predictions_class
    
    def save_model(self, filepath: str):
        """モデル保存"""
        if self.model is not None:
            self.model.save(filepath)
            print(f"モデル保存完了: {filepath}")
    
    def load_model(self, filepath: str):
        """モデル読み込み"""
        self.model = models.load_model(filepath)
        print(f"モデル読み込み完了: {filepath}")
    
    def get_model_summary(self):
        """モデルサマリー表示"""
        if self.model is not None:
            self.model.summary()
        else:
            print("モデルが構築されていません")


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


def create_light_model(sequence_length: int = 30, n_features: int = 87) -> ScalpingCNNLSTM:
    """
    軽量化スキャルピングモデル作成（過学習対策版）
    Args:
        sequence_length: 時系列長
        n_features: 特徴量数
    Returns:
        ScalpingCNNLSTM: 軽量化モデルインスタンス
    """
    model = ScalpingCNNLSTM(
        sequence_length=sequence_length,
        n_features=n_features,
        cnn_filters=[16, 32],      # 軽量化
        kernel_sizes=[3, 5],       # 軽量化
        lstm_units=32,             # 軽量化
        dropout_rate=0.5,          # 強化
        learning_rate=0.001
    )
    
    model.build_model()
    print(f"軽量化モデル作成完了 - パラメータ数: {model.model.count_params():,}")
    return model


def create_sample_model(sequence_length: int = 30, n_features: int = 87) -> ScalpingCNNLSTM:
    """
    サンプルモデル作成（軽量化版にリダイレクト）
    """
    return create_light_model(sequence_length, n_features)


if __name__ == "__main__":
    # モデルテスト
    print("=== スキャルピングCNN+LSTMモデル テスト ===")
    
    try:
        # サンプルモデル作成
        model = create_sample_model()
        
        # モデルサマリー表示
        model.get_model_summary()
        
        # サンプルデータでテスト
        batch_size = 32
        sequence_length = 30
        n_features = 87
        
        # ダミーデータ作成
        X_sample = np.random.randn(batch_size, sequence_length, n_features)
        
        # 予測テスト
        pred_proba, pred_class = model.predict(X_sample)
        
        print(f"\n予測テスト完了:")
        print(f"入力形状: {X_sample.shape}")
        print(f"予測確率形状: {pred_proba.shape}")
        print(f"予測クラス形状: {pred_class.shape}")
        print(f"予測クラス例: {pred_class[:5]}")
        
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()