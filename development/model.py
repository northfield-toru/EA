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


def create_lightweight_model(sequence_length: int = 30, n_features: int = 87, n_classes: int = 2) -> ScalpingCNNLSTM:
    """
    ChatGPT提案の軽量化モデル
    Args:
        sequence_length: 時系列長
        n_features: 特徴量数
        n_classes: クラス数（2=TRADE/NO_TRADE, 3=BUY/SELL/NO_TRADE）
    Returns:
        ScalpingCNNLSTM: 軽量モデル
    """
    model = ScalpingCNNLSTM(
        sequence_length=sequence_length,
        n_features=n_features,
        n_classes=n_classes,
        cnn_filters=[16, 32],      # 軽量化（元: [32,64,128]）
        kernel_sizes=[3, 5],       # 軽量化（元: [3,5,7]）
        lstm_units=32,             # 軽量化（元: 64）
        dropout_rate=0.5,          # 強化（元: 0.3）
        learning_rate=0.001
    )
    
    model.build_model()
    
    # パラメータ数表示
    param_count = model.model.count_params()
    print(f"軽量モデル構築完了")
    print(f"  パラメータ数: {param_count:,} （目標: 100K以下）")
    print(f"  クラス数: {n_classes}")
    print(f"  ドロップアウト率: {model.dropout_rate}")
    
    return model


def create_sample_model(sequence_length: int = 30, n_features: int = 87) -> ScalpingCNNLSTM:
    """軽量モデルにリダイレクト"""
    return create_lightweight_model(sequence_length, n_features, n_classes=2)

class LightweightScalpingModel:
    """Phase 2B用軽量スキャルピングモデル"""
    
    def __init__(self, 
                 sequence_length: int = 30,
                 n_features: int = 87,
                 n_classes: int = 2,
                 learning_rate: float = 0.001):
        """
        軽量版モデル初期化
        - パラメータ数を50K以下に抑制
        - 学習速度とメモリ効率を重視
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        
        self.model = None
        self.scaler = None
        self.class_weights = None
        
        print(f"軽量モデル初期化: seq={sequence_length}, features={n_features}, classes={n_classes}")
    
    def build_lightweight_model(self) -> models.Model:
        """
        軽量モデル構築
        - Conv1D + LSTM のシンプル構成
        - パラメータ数最小化
        """
        print("軽量モデル構築開始...")
        
        # 入力層
        input_layer = layers.Input(
            shape=(self.sequence_length, self.n_features),
            name='input_layer'
        )
        
        # 軽量Conv1D層
        x = layers.Conv1D(
            filters=16,           # 軽量化: 16フィルタのみ
            kernel_size=3,
            activation='relu',
            padding='same',
            name='conv1d_light'
        )(input_layer)
        
        x = layers.BatchNormalization(name='bn_conv')(x)
        x = layers.MaxPooling1D(pool_size=2, name='pool_conv')(x)
        x = layers.Dropout(0.3, name='dropout_conv')(x)
        
        # 軽量LSTM層
        x = layers.LSTM(
            units=24,             # 軽量化: 24ユニットのみ
            return_sequences=False,
            dropout=0.3,
            recurrent_dropout=0.3,
            name='lstm_light'
        )(x)
        
        # 軽量Dense層
        x = layers.Dense(
            16,                   # 軽量化: 16ユニット
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            name='dense_light'
        )(x)
        
        x = layers.BatchNormalization(name='bn_dense')(x)
        x = layers.Dropout(0.4, name='dropout_dense')(x)
        
        # 出力層
        output_layer = layers.Dense(
            self.n_classes,
            activation='softmax',
            name='output_layer'
        )(x)
        
        # モデル作成
        model = models.Model(
            inputs=input_layer,
            outputs=output_layer,
            name='LightweightScalpingModel'
        )
        
        # コンパイル
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        param_count = model.count_params()
        print(f"軽量モデル構築完了: パラメータ数 {param_count:,}")
        
        if param_count > 100000:
            print(f"⚠️ 警告: パラメータ数が100K超過。さらなる軽量化を検討してください")
        else:
            print(f"✅ 軽量化成功: 目標100K以下達成")
        
        return model
    
    def calculate_enhanced_class_weights(self, y: np.array, strategy: str = "balanced_enhanced") -> Dict:
        """
        Phase 2B用強化クラス重み計算
        Args:
            y: ラベル配列
            strategy: "balanced", "balanced_enhanced", "custom"
        Returns:
            dict: クラス重み
        """
        unique_classes, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        
        if strategy == "balanced":
            # 標準的なbalanced重み
            class_weights = compute_class_weight('balanced', classes=unique_classes, y=y)
            
        elif strategy == "balanced_enhanced":
            # Phase 2B強化版: 少数クラスをさらに重視
            base_weights = compute_class_weight('balanced', classes=unique_classes, y=y)
            
            # 不均衡比に応じた追加強化
            majority_count = np.max(counts)
            minority_count = np.min(counts)
            imbalance_ratio = majority_count / minority_count
            
            # 強化係数計算
            if imbalance_ratio > 10:
                enhancement_factor = 1.5
            elif imbalance_ratio > 5:
                enhancement_factor = 1.3
            else:
                enhancement_factor = 1.1
            
            # 少数クラス（通常TRADE=1）を強化
            class_weights = base_weights.copy()
            minority_class_idx = np.argmin(counts)
            minority_class = unique_classes[minority_class_idx]
            
            for i, class_val in enumerate(unique_classes):
                if class_val == minority_class:
                    class_weights[i] *= enhancement_factor
                    
        elif strategy == "custom":
            # カスタム重み: TRADE勝率を考慮
            class_weights = np.ones(len(unique_classes))
            
            for i, class_val in enumerate(unique_classes):
                class_count = counts[i]
                class_ratio = class_count / total_samples
                
                if class_val == 1:  # TRADEクラス
                    # TRADEクラスの重みを動的調整
                    if class_ratio < 0.2:  # 20%未満なら強化
                        class_weights[i] = 3.0
                    elif class_ratio < 0.3:  # 30%未満なら中程度強化
                        class_weights[i] = 2.0
                    else:
                        class_weights[i] = 1.5
                else:  # NO_TRADEクラス
                    class_weights[i] = 1.0
        
        # 辞書形式に変換
        class_weight_dict = dict(zip(unique_classes, class_weights))
        
        print(f"Phase 2B クラス重み計算完了 ({strategy}):")
        for class_val, weight in class_weight_dict.items():
            class_name = 'NO_TRADE' if class_val == 0 else 'TRADE'
            count = counts[np.where(unique_classes == class_val)[0][0]]
            ratio = count / total_samples
            print(f"  {class_name}: 重み={weight:.2f}, サンプル数={count:,} ({ratio:.1%})")
        
        self.class_weights = class_weight_dict
        return class_weight_dict
    
    def prepare_sequences(self, features_df: pd.DataFrame, labels: Optional[pd.Series] = None) -> Tuple:
        """
        軽量版シーケンス準備
        """
        print(f"軽量版シーケンス準備: {len(features_df)} 行")
        
        # 数値型列のみ選択
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        features_numeric = features_df[numeric_columns].copy()
        
        # 欠損値・無限値処理
        features_clean = features_numeric.fillna(method='ffill').fillna(0)
        features_clean = features_clean.replace([np.inf, -np.inf], 0)
        
        # スケーリング（軽量版: StandardScaler使用）
        if self.scaler is None:
            self.scaler = StandardScaler()
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
        
        if labels is not None:
            y = np.array(sequence_labels)
            y_categorical = to_categorical(y, num_classes=self.n_classes)
            
            print(f"軽量版シーケンス完了: X={X.shape}, y={y_categorical.shape}")
            return X, y_categorical, y
        
        print(f"軽量版シーケンス完了: X={X.shape}")
        return X
    
    def train_lightweight(self, 
                         X_train: np.array, 
                         y_train: np.array,
                         X_val: np.array,
                         y_val: np.array,
                         epochs: int = 50,
                         batch_size: int = 64,
                         class_weight_strategy: str = "balanced_enhanced") -> Dict:
        """
        軽量モデル学習
        """
        if self.model is None:
            self.build_lightweight_model()
        
        print("軽量モデル学習開始...")
        
        # 強化クラス重み計算
        y_train_raw = np.argmax(y_train, axis=1)
        class_weights = self.calculate_enhanced_class_weights(y_train_raw, class_weight_strategy)
        
        # コールバック設定（軽量版）
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=7,  # やや長めに設定
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # 学習実行
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            class_weight=class_weights,
            verbose=1
        )
        
        print("軽量モデル学習完了")
        return history.history
    
    def predict_lightweight(self, X: np.array) -> Tuple[np.array, np.array]:
        """軽量モデル予測"""
        if self.model is None:
            raise ValueError("モデルが構築されていません")
        
        predictions_proba = self.model.predict(X, verbose=0)
        predictions_class = np.argmax(predictions_proba, axis=1)
        
        return predictions_proba, predictions_class

def create_phase2b_lightweight_model(sequence_length: int = 30, 
                                    n_features: int = 87, 
                                    n_classes: int = 2) -> LightweightScalpingModel:
    """
    Phase 2B用軽量モデル作成
    - パラメータ数50K以下
    - 高速学習・推論
    """
    model = LightweightScalpingModel(
        sequence_length=sequence_length,
        n_features=n_features,
        n_classes=n_classes,
        learning_rate=0.001
    )
    
    model.build_lightweight_model()
    
    param_count = model.model.count_params()
    print(f"Phase 2B軽量モデル作成完了: {param_count:,}パラメータ")
    
    return model

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