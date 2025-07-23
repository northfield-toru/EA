import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import to_categorical
import numpy as np
from typing import Dict, Any, Tuple, List
import logging
from .utils import set_random_seed

logger = logging.getLogger(__name__)

class ScalpingModel:
    """
    USDJPYスキャルピング用AIモデル
    Conv1D + LSTM + Dense のハイブリッド構成
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config['model']
        self.system_config = config['system']
        
        # 再現性確保
        set_random_seed(self.system_config['random_seed'])
        
        # GPU設定
        if self.system_config.get('gpu_memory_growth', True):
            self._setup_gpu()
        
        # 混合精度設定
        if self.system_config.get('mixed_precision', True):
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            logger.info("混合精度トレーニング有効化")
        
        self.model = None
        self.feature_count = None
        
    def _setup_gpu(self):
        """GPU設定"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
            except RuntimeError as e:
                logger.warning(f"GPU setup failed: {e}")
        else:
            logger.info("No GPU detected, using CPU")
    
    def build_conv1d_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        Conv1D ベースモデル構築
        時系列パターン認識に特化
        """
        logger.info(f"Conv1D モデル構築開始: input_shape={input_shape}")
        
        # 入力層
        inputs = layers.Input(shape=input_shape, name='tick_sequences')
        
        x = inputs
        
        # Conv1D ブロック
        conv_filters = self.model_config['conv_filters']
        kernel_size = self.model_config['conv_kernel_size']
        pool_size = self.model_config['pool_size']
        
        for i, filters in enumerate(conv_filters):
            # Convolution
            x = layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation='relu',
                padding='same',
                name=f'conv1d_{i+1}'
            )(x)
            
            # Batch Normalization
            x = layers.BatchNormalization(name=f'bn_conv_{i+1}')(x)
            
            # Max Pooling
            if i < len(conv_filters) - 1:  # 最後以外でプーリング
                x = layers.MaxPooling1D(pool_size=pool_size, name=f'pool_{i+1}')(x)
            
            # Dropout
            x = layers.Dropout(self.model_config['dropout_rate'], name=f'dropout_conv_{i+1}')(x)
        
        # Global Average Pooling で次元削減
        x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
        
        # Dense ブロック
        dense_units = self.model_config['dense_units']
        for i, units in enumerate(dense_units):
            x = layers.Dense(units, activation='relu', name=f'dense_{i+1}')(x)
            x = layers.BatchNormalization(name=f'bn_dense_{i+1}')(x)
            x = layers.Dropout(self.model_config['dropout_rate'], name=f'dropout_dense_{i+1}')(x)
        
        # 出力層
        outputs = layers.Dense(
            self.model_config['num_classes'],
            activation='softmax',
            name='classification_output',
            dtype='float32'  # 混合精度対応
        )(x)
        
        model = models.Model(inputs=inputs, outputs=outputs, name='ScalpingConv1D')
        
        logger.info(f"Conv1D モデル構築完了: パラメータ数={model.count_params():,}")
        return model
    
    def build_lstm_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        LSTM ベースモデル構築
        長期依存関係の学習に特化
        """
        logger.info(f"LSTM モデル構築開始: input_shape={input_shape}")
        
        inputs = layers.Input(shape=input_shape, name='tick_sequences')
        
        # LSTM ブロック
        x = layers.LSTM(
            units=128,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.2,
            name='lstm_1'
        )(inputs)
        x = layers.BatchNormalization(name='bn_lstm_1')(x)
        
        x = layers.LSTM(
            units=64,
            return_sequences=False,
            dropout=0.2,
            recurrent_dropout=0.2,
            name='lstm_2'
        )(x)
        x = layers.BatchNormalization(name='bn_lstm_2')(x)
        
        # Dense ブロック
        dense_units = self.model_config['dense_units']
        for i, units in enumerate(dense_units):
            x = layers.Dense(units, activation='relu', name=f'dense_{i+1}')(x)
            x = layers.BatchNormalization(name=f'bn_dense_{i+1}')(x)
            x = layers.Dropout(self.model_config['dropout_rate'], name=f'dropout_dense_{i+1}')(x)
        
        # 出力層
        outputs = layers.Dense(
            self.model_config['num_classes'],
            activation='softmax',
            name='classification_output',
            dtype='float32'
        )(x)
        
        model = models.Model(inputs=inputs, outputs=outputs, name='ScalpingLSTM')
        
        logger.info(f"LSTM モデル構築完了: パラメータ数={model.count_params():,}")
        return model
    
    def build_hybrid_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        Conv1D + LSTM ハイブリッドモデル
        短期パターン + 長期依存性を両方活用
        """
        logger.info(f"ハイブリッドモデル構築開始: input_shape={input_shape}")
        
        inputs = layers.Input(shape=input_shape, name='tick_sequences')
        
        # Conv1D 分岐
        conv_branch = inputs
        for i, filters in enumerate([64, 128]):
            conv_branch = layers.Conv1D(
                filters=filters,
                kernel_size=3,
                activation='relu',
                padding='same',
                name=f'conv_branch_{i+1}'
            )(conv_branch)
            conv_branch = layers.BatchNormalization(name=f'bn_conv_branch_{i+1}')(conv_branch)
            if i == 0:
                conv_branch = layers.MaxPooling1D(pool_size=2, name=f'pool_conv_branch_{i+1}')(conv_branch)
        
        # LSTM 分岐
        lstm_branch = inputs
        lstm_branch = layers.LSTM(
            units=64,
            return_sequences=True,
            dropout=0.2,
            name='lstm_branch'
        )(lstm_branch)
        
        # 分岐を結合
        merged = layers.Concatenate(axis=-1, name='merge_branches')([conv_branch, lstm_branch])
        
        # 最終LSTM層
        x = layers.LSTM(
            units=128,
            return_sequences=False,
            dropout=0.2,
            name='final_lstm'
        )(merged)
        x = layers.BatchNormalization(name='bn_final')(x)
        
        # Dense ブロック
        for i, units in enumerate(self.model_config['dense_units']):
            x = layers.Dense(units, activation='relu', name=f'dense_{i+1}')(x)
            x = layers.BatchNormalization(name=f'bn_dense_{i+1}')(x)
            x = layers.Dropout(self.model_config['dropout_rate'], name=f'dropout_dense_{i+1}')(x)
        
        # 出力層
        outputs = layers.Dense(
            self.model_config['num_classes'],
            activation='softmax',
            name='classification_output',
            dtype='float32'
        )(x)
        
        model = models.Model(inputs=inputs, outputs=outputs, name='ScalpingHybrid')
        
        logger.info(f"ハイブリッドモデル構築完了: パラメータ数={model.count_params():,}")
        return model
    
    def build_transformer_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        Transformer ベースモデル
        アテンション機構による高度な時系列解析
        """
        logger.info(f"Transformer モデル構築開始: input_shape={input_shape}")
        
        def transformer_block(inputs, head_size, num_heads, ff_dim, dropout=0.1):
            # Multi-head self-attention
            attention = layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=head_size,
                dropout=dropout
            )(inputs, inputs)
            
            # Add & Norm
            attention = layers.Dropout(dropout)(attention)
            attention = layers.LayerNormalization(epsilon=1e-6)(inputs + attention)
            
            # Feed forward
            ff = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(attention)
            ff = layers.Dropout(dropout)(ff)
            ff = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(ff)
            
            # Add & Norm
            return layers.LayerNormalization(epsilon=1e-6)(attention + ff)
        
        inputs = layers.Input(shape=input_shape, name='tick_sequences')
        
        # Positional encoding
        x = inputs
        
        # Transformer blocks
        for i in range(2):
            x = transformer_block(
                x,
                head_size=32,
                num_heads=4,
                ff_dim=64,
                dropout=0.1
            )
        
        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense ブロック
        for i, units in enumerate(self.model_config['dense_units']):
            x = layers.Dense(units, activation='relu', name=f'dense_{i+1}')(x)
            x = layers.Dropout(self.model_config['dropout_rate'], name=f'dropout_dense_{i+1}')(x)
        
        # 出力層
        outputs = layers.Dense(
            self.model_config['num_classes'],
            activation='softmax',
            name='classification_output',
            dtype='float32'
        )(x)
        
        model = models.Model(inputs=inputs, outputs=outputs, name='ScalpingTransformer')
        
        logger.info(f"Transformer モデル構築完了: パラメータ数={model.count_params():,}")
        return model
    
    def create_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        設定に基づいてモデルを作成
        """
        architecture = self.model_config.get('architecture', 'conv1d')
        
        if architecture == 'conv1d':
            self.model = self.build_conv1d_model(input_shape)
        elif architecture == 'lstm':
            self.model = self.build_lstm_model(input_shape)
        elif architecture == 'hybrid':
            self.model = self.build_hybrid_model(input_shape)
        elif architecture == 'transformer':
            self.model = self.build_transformer_model(input_shape)
        else:
            raise ValueError(f"未対応のアーキテクチャ: {architecture}")
        
        # モデル設定
        self.feature_count = input_shape[-1]
        
        return self.model
    
    def compile_model(self):
        """
        モデルのコンパイル
        """
        # オプティマイザ
        optimizer = optimizers.Adam(
            learning_rate=self.model_config['learning_rate'],
            clipnorm=1.0  # 勾配クリッピング
        )
        
        # 損失関数
        loss = 'sparse_categorical_crossentropy'
        
        # メトリクス（整数ラベル用に修正）
        metrics = [
            'sparse_categorical_accuracy',  # デフォルトのaccuracyを削除
            tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),  # 明示的に指定
        ]
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        logger.info("モデルコンパイル完了")
    
    def get_callbacks(self, model_path: str) -> List[tf.keras.callbacks.Callback]:
        """
        トレーニングコールバックを取得
        """
        callbacks_list = [
            # Early Stopping
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=self.model_config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            
            # Model Checkpoint
            callbacks.ModelCheckpoint(
                filepath=model_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # Learning Rate Reduction
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.model_config['reduce_lr_patience'],
                min_lr=1e-7,
                verbose=1
            ),
            
            # CSV Logger
            callbacks.CSVLogger(
                model_path.replace('.h5', '_training_log.csv'),
                append=True
            )
        ]
        
        return callbacks_list
    
    def predict_with_confidence(self, X: np.ndarray, threshold: float = 0.6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        信頼度を考慮した予測
        """
        if self.model is None:
            raise ValueError("モデルが未構築です")
        
        # 予測実行
        predictions = self.model.predict(X, verbose=0)
        
        # 最大確率とクラス
        max_probs = np.max(predictions, axis=1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # 信頼度フィルタリング
        high_confidence_mask = max_probs >= threshold
        
        # 低信頼度のものはNO_TRADEに変更
        filtered_predictions = predicted_classes.copy()
        filtered_predictions[~high_confidence_mask] = self.config['labels']['no_trade_class']
        
        return filtered_predictions, max_probs, predictions