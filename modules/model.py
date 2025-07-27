import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import logging
from typing import Tuple

class ForexModelBuilder:
    """FXトレーディングモデル構築クラス"""
    
    def __init__(self, config):
        self.config = config
        self.model_config = config['model']
        self.logger = logging.getLogger(__name__)
        
    def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """
        モデルを構築
        
        Args:
            input_shape: 入力形状 (timesteps, features)
            
        Returns:
            keras.Model: 構築されたモデル
        """
        model_type = self.model_config['type'].upper()  # 大文字に統一
        
        self.logger.info(f"Building {model_type} model with input shape: {input_shape}")
        
        if model_type == 'CONV1D':
            model = self._build_conv1d_model(input_shape)
        elif model_type == 'LSTM':
            model = self._build_lstm_model(input_shape)
        elif model_type == 'HYBRID':
            model = self._build_hybrid_model(input_shape)
        elif model_type == 'TRANSFORMER':
            model = self._build_transformer_model(input_shape)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Supported types: Conv1D, LSTM, Hybrid, Transformer")
        
        # モデルコンパイル
        self._compile_model(model)
        
        # モデルサマリーをログ出力
        model.summary(print_fn=self.logger.info)
        
        return model
    
    def _build_conv1d_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Conv1Dモデル構築"""
        inputs = keras.Input(shape=input_shape)
        
        x = inputs
        
        # Conv1Dレイヤー
        filters = self.model_config.get('conv_filters', [32, 64, 128])
        kernel_size = self.model_config.get('conv_kernel_size', 3)
        dropout_rate = self.model_config.get('dropout_rate', 0.2)
        
        for i, filter_count in enumerate(filters):
            x = layers.Conv1D(
                filters=filter_count,
                kernel_size=kernel_size,
                activation='relu',
                padding='same',
                name=f'conv1d_{i+1}'
            )(x)
            x = layers.BatchNormalization(name=f'bn_conv_{i+1}')(x)
            x = layers.Dropout(dropout_rate, name=f'dropout_conv_{i+1}')(x)
            
            # プーリング（最後以外）
            if i < len(filters) - 1:
                x = layers.MaxPooling1D(pool_size=2, name=f'pool_{i+1}')(x)
        
        # Global pooling
        x = layers.GlobalMaxPooling1D(name='global_pool')(x)
        
        # Dense layers
        x = layers.Dense(128, activation='relu', name='dense_1')(x)
        x = layers.BatchNormalization(name='bn_dense_1')(x)
        x = layers.Dropout(dropout_rate, name='dropout_dense_1')(x)
        
        x = layers.Dense(64, activation='relu', name='dense_2')(x)
        x = layers.BatchNormalization(name='bn_dense_2')(x)
        x = layers.Dropout(dropout_rate, name='dropout_dense_2')(x)
        
        # 出力層
        outputs = layers.Dense(3, activation='softmax', name='output')(x)
        
        model = keras.Model(inputs, outputs, name='Conv1D_Model')
        return model
    
    def _build_lstm_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """LSTMモデル構築"""
        inputs = keras.Input(shape=input_shape)
        
        x = inputs
        
        # LSTMレイヤー
        lstm_units = self.model_config.get('lstm_units', [64, 32])
        dropout_rate = self.model_config.get('dropout_rate', 0.2)
        
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1
            
            x = layers.LSTM(
                units=units,
                return_sequences=return_sequences,
                dropout=dropout_rate,
                recurrent_dropout=dropout_rate,
                name=f'lstm_{i+1}'
            )(x)
            
            if return_sequences:
                x = layers.BatchNormalization(name=f'bn_lstm_{i+1}')(x)
        
        # Dense layers
        x = layers.Dense(128, activation='relu', name='dense_1')(x)
        x = layers.BatchNormalization(name='bn_dense_1')(x)
        x = layers.Dropout(dropout_rate, name='dropout_dense_1')(x)
        
        x = layers.Dense(64, activation='relu', name='dense_2')(x)
        x = layers.BatchNormalization(name='bn_dense_2')(x)
        x = layers.Dropout(dropout_rate, name='dropout_dense_2')(x)
        
        # 出力層
        outputs = layers.Dense(3, activation='softmax', name='output')(x)
        
        model = keras.Model(inputs, outputs, name='LSTM_Model')
        return model
    
    def _build_hybrid_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Hybrid（Conv1D + LSTM）モデル構築"""
        inputs = keras.Input(shape=input_shape)
        
        # Conv1D部分
        conv_filters = self.model_config.get('conv_filters', [32, 64])
        kernel_size = self.model_config.get('conv_kernel_size', 3)
        dropout_rate = self.model_config.get('dropout_rate', 0.2)
        
        x = inputs
        
        for i, filter_count in enumerate(conv_filters):
            x = layers.Conv1D(
                filters=filter_count,
                kernel_size=kernel_size,
                activation='relu',
                padding='same',
                name=f'conv1d_{i+1}'
            )(x)
            x = layers.BatchNormalization(name=f'bn_conv_{i+1}')(x)
            x = layers.Dropout(dropout_rate * 0.5, name=f'dropout_conv_{i+1}')(x)
        
        # LSTM部分
        lstm_units = self.model_config.get('lstm_units', [64, 32])
        
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1
            
            x = layers.LSTM(
                units=units,
                return_sequences=return_sequences,
                dropout=dropout_rate * 0.5,
                recurrent_dropout=dropout_rate * 0.5,
                name=f'lstm_{i+1}'
            )(x)
            
            if return_sequences:
                x = layers.BatchNormalization(name=f'bn_lstm_{i+1}')(x)
        
        # Dense layers
        x = layers.Dense(128, activation='relu', name='dense_1')(x)
        x = layers.BatchNormalization(name='bn_dense_1')(x)
        x = layers.Dropout(dropout_rate, name='dropout_dense_1')(x)
        
        x = layers.Dense(64, activation='relu', name='dense_2')(x)
        x = layers.BatchNormalization(name='bn_dense_2')(x)
        x = layers.Dropout(dropout_rate, name='dropout_dense_2')(x)
        
        # 出力層
        outputs = layers.Dense(3, activation='softmax', name='output')(x)
        
        model = keras.Model(inputs, outputs, name='Hybrid_Model')
        return model
    
    def _build_transformer_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Transformerモデル構築"""
        inputs = keras.Input(shape=input_shape)
        
        # パラメータ
        d_model = 128
        num_heads = 8
        ff_dim = 256
        dropout_rate = self.model_config.get('dropout_rate', 0.2)
        
        # Position Encoding
        x = self._add_position_encoding(inputs, d_model)
        
        # Multi-Head Attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
            name='multihead_attention'
        )(x, x)
        
        # Add & Norm 1
        x = layers.Add(name='add_1')([x, attention_output])
        x = layers.LayerNormalization(epsilon=1e-6, name='layer_norm_1')(x)
        
        # Feed Forward Network
        ffn_output = self._create_ffn(x, ff_dim, dropout_rate)
        
        # Add & Norm 2
        x = layers.Add(name='add_2')([x, ffn_output])
        x = layers.LayerNormalization(epsilon=1e-6, name='layer_norm_2')(x)
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
        
        # Dense layers
        x = layers.Dense(128, activation='relu', name='dense_1')(x)
        x = layers.BatchNormalization(name='bn_dense_1')(x)
        x = layers.Dropout(dropout_rate, name='dropout_dense_1')(x)
        
        x = layers.Dense(64, activation='relu', name='dense_2')(x)
        x = layers.BatchNormalization(name='bn_dense_2')(x)
        x = layers.Dropout(dropout_rate, name='dropout_dense_2')(x)
        
        # 出力層
        outputs = layers.Dense(3, activation='softmax', name='output')(x)
        
        model = keras.Model(inputs, outputs, name='Transformer_Model')
        return model
    
    def _add_position_encoding(self, inputs, d_model):
        """Position Encoding追加"""
        # 入力を指定次元にプロジェクション
        x = layers.Dense(d_model, name='input_projection')(inputs)
        
        # Position encodingは簡易版を使用
        # 実際のTransformerの実装ではsinusoidal encodingを使用
        seq_len = tf.shape(x)[1]
        pos_encoding = layers.Embedding(
            input_dim=1000,  # 最大シーケンス長
            output_dim=d_model,
            name='position_embedding'
        )(tf.range(seq_len))
        
        return x + pos_encoding
    
    def _create_ffn(self, x, ff_dim, dropout_rate):
        """Feed Forward Network作成"""
        ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(tf.shape(x)[-1])
        ], name='ffn')
        
        return ffn(x)
    
    def _compile_model(self, model: keras.Model):
        """モデルコンパイル（完全版）"""
        learning_rate = self.model_config.get('learning_rate', 0.001)
        label_smoothing = self.model_config.get('label_smoothing', 0.0)
        loss_type = self.model_config.get('loss_type', 'sparse_categorical_crossentropy')
        
        # オプティマイザー
        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        # 損失関数選択（完全版）
        if loss_type == 'focal':
            # Focal Loss
            alpha = self.model_config.get('focal_alpha', 1.0)
            gamma = self.model_config.get('focal_gamma', 2.0)
            loss = self._create_focal_loss(alpha, gamma)
            self.logger.info(f"Using Focal Loss: alpha={alpha}, gamma={gamma}")
            
        elif loss_type == 'label_smoothing' or label_smoothing > 0:
            # Label Smoothing Loss
            smoothing = max(label_smoothing, 0.1)
            loss = self._create_label_smoothing_loss(smoothing)
            self.logger.info(f"Using Label Smoothing Loss: smoothing={smoothing}")
            
        else:
            # 標準損失関数
            loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
            self.logger.info("Using standard SparseCategoricalCrossentropy")
        
        # メトリクス
        metrics = [
            'accuracy',
            keras.metrics.SparseCategoricalAccuracy(name='sparse_categorical_accuracy'),
            keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top_2_accuracy')
        ]
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        self.logger.info(f"Model compiled with learning_rate: {learning_rate}, loss_type: {loss_type}")
    
    def _create_label_smoothing_loss(self, smoothing):
        """Label Smoothing Loss作成（完全版）"""
        
        class LabelSmoothingLoss(tf.keras.losses.Loss):
            def __init__(self, smoothing=0.1, **kwargs):
                super().__init__(**kwargs)
                self.smoothing = smoothing
            
            def call(self, y_true, y_pred):
                # y_trueの形状を正規化
                y_true = tf.cast(tf.squeeze(y_true), tf.int32)
                
                # クラス数を取得
                num_classes = tf.shape(y_pred)[-1]
                
                # one-hot encoding
                y_true_onehot = tf.one_hot(y_true, num_classes, dtype=tf.float32)
                
                # label smoothing適用
                smooth_factor = self.smoothing / tf.cast(num_classes, tf.float32)
                y_true_smooth = y_true_onehot * (1.0 - self.smoothing) + smooth_factor
                
                # categorical crossentropy計算
                y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
                loss = -tf.reduce_sum(y_true_smooth * tf.math.log(y_pred), axis=-1)
                
                return loss
            
            def get_config(self):
                config = super().get_config()
                config.update({'smoothing': self.smoothing})
                return config
        
        return LabelSmoothingLoss(smoothing=smoothing)
    
    def _create_focal_loss(self, alpha, gamma):
        """Focal Loss作成（完全版）"""
        
        class SparseFocalLoss(tf.keras.losses.Loss):
            def __init__(self, alpha=1.0, gamma=2.0, **kwargs):
                super().__init__(**kwargs)
                self.alpha = alpha
                self.gamma = gamma
            
            def call(self, y_true, y_pred):
                # y_trueの形状を調整
                y_true = tf.cast(tf.squeeze(y_true), tf.int32)
                
                # 予測確率をクリップ
                y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
                
                # 正解クラスの予測確率を取得
                batch_size = tf.shape(y_true)[0]
                indices = tf.stack([tf.range(batch_size), y_true], axis=1)
                pt = tf.gather_nd(y_pred, indices)
                
                # Focal weight計算
                focal_weight = self.alpha * tf.pow(1.0 - pt, self.gamma)
                
                # Cross entropy計算
                ce_loss = -tf.math.log(pt)
                
                # Focal loss
                focal_loss = focal_weight * ce_loss
                
                return focal_loss
            
            def get_config(self):
                config = super().get_config()
                config.update({
                    'alpha': self.alpha,
                    'gamma': self.gamma
                })
                return config
        
        return SparseFocalLoss(alpha=alpha, gamma=gamma)
    
    def create_callbacks(self, log_dir: str) -> list:
        """コールバック作成"""
        callbacks = []
        
        # Early Stopping
        patience = self.model_config.get('early_stopping_patience', 10)
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Model Checkpoint
        checkpoint_path = f"{log_dir}/best_model.h5"
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(model_checkpoint)
        
        # Reduce Learning Rate
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # CSV Logger
        csv_logger = keras.callbacks.CSVLogger(
            f"{log_dir}/training_history.csv",
            append=False
        )
        callbacks.append(csv_logger)
        
        # TensorBoard（オプション）
        tensorboard = keras.callbacks.TensorBoard(
            log_dir=f"{log_dir}/tensorboard",
            histogram_freq=1,
            write_graph=True,
            write_images=False
        )
        callbacks.append(tensorboard)
        
        self.logger.info(f"Created {len(callbacks)} callbacks")
        return callbacks
    
    def get_model_info(self, model: keras.Model) -> dict:
        """モデル情報取得"""
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        info = {
            'model_type': self.model_config['type'],
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'non_trainable_parameters': int(non_trainable_params),
            'input_shape': model.input_shape,
            'output_shape': model.output_shape,
            'layers_count': len(model.layers)
        }
        
        return info