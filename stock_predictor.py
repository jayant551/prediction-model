import numpy as np
import pandas as pd
from typing import cast
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import warnings
from typing import Dict, Any
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# Deep Learning

import tensorflow as tf 
from tensorflow.keras.models import Sequential, Model # type: ignore
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input, Attention, MultiHeadAttention, LayerNormalization # type: ignore
from tensorflow.keras.optimizers import Adam, RMSprop # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint # type: ignore
from tensorflow.keras.regularizers import l2 # pyright: ignore[reportMissingImports]

# Additional Libraries
import joblib
import json
import datetime as dt
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration class for model parameters"""
    sequence_length: int = 60
    test_size: float = 0.2
    validation_size: float = 0.1
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    dropout_rate: float = 0.2
    l2_reg: float = 0.01

class AdvancedFeatureEngineering:
    """Advanced feature engineering for stock prediction"""
    
    @staticmethod
    def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        df = data.copy()
        
        # Moving Averages
        for window in [5, 10, 20, 50, 100, 200]:
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'EMA_{window}'] = df['Close'].ewm(span=window).mean()
        
        # Price-based indicators
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Open_Close_Ratio'] = df['Open'] / df['Close']
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
        
        # Volume indicators
        df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_10']
        df['Price_Volume'] = df['Close'] * df['Volume']
        
        # Volatility indicators
        df['Volatility_10'] = df['Close'].pct_change().rolling(window=10).std()
        df['Volatility_30'] = df['Close'].pct_change().rolling(window=30).std()
        
        # RSI
        df['RSI_14'] = AdvancedFeatureEngineering.calculate_rsi(df['Close'], 14)
        df['RSI_30'] = AdvancedFeatureEngineering.calculate_rsi(df['Close'], 30)
        
        # MACD
        macd_data = AdvancedFeatureEngineering.calculate_macd(df['Close'])
        df = pd.concat([df, macd_data], axis=1)
        
        # Bollinger Bands
        bb_data = AdvancedFeatureEngineering.calculate_bollinger_bands(df['Close'])
        df = pd.concat([df, bb_data], axis=1)
        
        # Stochastic Oscillator
        stoch_data = AdvancedFeatureEngineering.calculate_stochastic(df)
        df = pd.concat([df, stoch_data], axis=1)
        
        # Williams %R
        df['Williams_R'] = AdvancedFeatureEngineering.calculate_williams_r(df)
        
        # Average True Range (ATR)
        df['ATR'] = AdvancedFeatureEngineering.calculate_atr(df)
        
        # Money Flow Index
        df['MFI'] = AdvancedFeatureEngineering.calculate_mfi(df)
        
        return df
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        delta_numeric = cast(pd.Series, delta.astype(float))
        gain = (delta_numeric.where(delta_numeric > 0, 0)).rolling(window=window).mean()
        loss = (-delta_numeric.where(delta_numeric < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD indicators"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'MACD': macd_line,
            'MACD_Signal': signal_line,
            'MACD_Histogram': histogram
        })
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        bb_width = upper_band - lower_band
        bb_position = (prices - lower_band) / bb_width
        
        return pd.DataFrame({
            'BB_Upper': upper_band,
            'BB_Middle': rolling_mean,
            'BB_Lower': lower_band,
            'BB_Width': bb_width,
            'BB_Position': bb_position
        })
    
    @staticmethod
    def calculate_stochastic(data: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
        """Calculate Stochastic Oscillator"""
        low_min = data['Low'].rolling(window=k_window).min()
        high_max = data['High'].rolling(window=k_window).max()
        k_percent = 100 * ((data['Close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_window).mean()
        
        return pd.DataFrame({
            'Stoch_K': k_percent,
            'Stoch_D': d_percent
        })
    
    @staticmethod
    def calculate_williams_r(data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        high_max = data['High'].rolling(window=window).max()
        low_min = data['Low'].rolling(window=window).min()
        return -100 * ((high_max - data['Close']) / (high_max - low_min))
    
    @staticmethod
    def calculate_atr(data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        high_low_series = pd.Series(high_low)
        high_close_series = pd.Series(high_close)
        low_close_series = pd.Series(low_close)

        true_range = pd.concat([high_low_series, high_close_series, low_close_series], axis=1).max(axis=1)
        return true_range.rolling(window=window).mean()
    
    @staticmethod
    def calculate_mfi(data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Money Flow Index"""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        money_flow = typical_price * data['Volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=window).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=window).sum()
        
        mfi_ratio = positive_flow / negative_flow
        return 100 - (100 / (1 + mfi_ratio))

class MultiModelPredictor:
    """Multi-model ensemble predictor with various ML algorithms"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.scalers = {}
        self.models = {}
        self.feature_importance = {}
        self.model_weights = {}
        
    def create_lstm_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Create advanced LSTM model"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape, 
                 kernel_regularizer=l2(self.config.l2_reg)),
            Dropout(self.config.dropout_rate),
            
            LSTM(64, return_sequences=True, kernel_regularizer=l2(self.config.l2_reg)),
            Dropout(self.config.dropout_rate),
            
            LSTM(32, kernel_regularizer=l2(self.config.l2_reg)),
            Dropout(self.config.dropout_rate),
            
            Dense(50, activation='relu', kernel_regularizer=l2(self.config.l2_reg)),
            Dropout(self.config.dropout_rate),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_gru_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Create GRU model"""
        model = Sequential([
            GRU(128, return_sequences=True, input_shape=input_shape,
                kernel_regularizer=l2(self.config.l2_reg)),
            Dropout(self.config.dropout_rate),
            
            GRU(64, return_sequences=True, kernel_regularizer=l2(self.config.l2_reg)),
            Dropout(self.config.dropout_rate),
            
            GRU(32, kernel_regularizer=l2(self.config.l2_reg)),
            Dropout(self.config.dropout_rate),
            
            Dense(50, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_cnn_lstm_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Create CNN-LSTM hybrid model"""
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            
            LSTM(50, return_sequences=True),
            Dropout(self.config.dropout_rate),
            
            LSTM(25),
            Dropout(self.config.dropout_rate),
            
            Dense(50, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_transformer_model(self, input_shape: Tuple[int, int]) -> Model:
        """Create Transformer-based model"""
        inputs = Input(shape=input_shape)
        
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=8, 
            key_dim=64
        )(inputs, inputs)
        
        attention_output = Dropout(0.1)(attention_output)
        attention_output = LayerNormalization()(inputs + attention_output)
        
        # Feed-forward network
        ffn_output = Dense(128, activation='relu')(attention_output)
        ffn_output = Dense(input_shape[-1])(ffn_output)
        ffn_output = Dropout(0.1)(ffn_output)
        ffn_output = LayerNormalization()(attention_output + ffn_output)
        
        # Global average pooling and output
        pooled = tf.keras.layers.GlobalAveragePooling1D()(ffn_output)
        outputs = Dense(1)(pooled)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model

class FullScaleStockPredictor:
    """Main predictor class with full-scale capabilities"""
    
    def __init__(self, symbols: List[str], config: Optional[ModelConfig] = None):
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.config = config or ModelConfig()
        self.data = {}
        self.processed_data = {}
        self.models = {}
        self.scalers = {}
        self.predictions = {}
        self.evaluation_metrics = {}
        self.feature_engineering = AdvancedFeatureEngineering()
        
        logger.info(f"Initialized FullScaleStockPredictor for {len(self.symbols)} symbols")
    
    def load_data(self, period: str = '5y', interval: str = '1d') -> Dict[str, pd.DataFrame]:
        """Load data for all symbols with comprehensive error handling"""
        logger.info(f"Loading data for {len(self.symbols)} symbols...")
        
        for symbol in self.symbols:
            try:
                stock = yf.Ticker(symbol)
                data = stock.history(period=period, interval=interval)
                
                if data.empty:
                    logger.warning(f"No data found for {symbol}")
                    continue
                
                # Add technical indicators
                data = self.feature_engineering.add_technical_indicators(data)
                
                # Add market context features
                data = self._add_market_context(data, symbol)
                
                self.data[symbol] = data
                logger.info(f"Loaded {len(data)} records for {symbol}")
                
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {e}")
                continue
        
        return self.data
    
    def _add_market_context(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add market context features"""
        df = data.copy()

        if not isinstance(df.index, pd.DatetimeIndex):
             df.index = pd.to_datetime(df.index)
        
        # Time-based features
        df['DayOfWeek'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['IsMonthEnd'] = df.index.is_month_end.astype(int)
        df['IsQuarterEnd'] = df.index.is_quarter_end.astype(int)
        
        # Seasonal features
        df['DayOfYear'] = df.index.dayofyear
        df['WeekOfYear'] = df.index.isocalendar().week
        
        # Market session features
        df['IsMonday'] = (df['DayOfWeek'] == 0).astype(int)
        df['IsFriday'] = (df['DayOfWeek'] == 4).astype(int)
        
        return df
    
    def prepare_features(self, target_column: str = 'Close') -> Dict[str, np.ndarray]:
        """Prepare features for all symbols"""
        logger.info("Preparing features for all symbols...")
        
        for symbol in self.symbols:
            if symbol not in self.data:
                continue
                
            try:
                # Select numerical features only
                numerical_cols = self.data[symbol].select_dtypes(include=[np.number]).columns
                feature_data = self.data[symbol][numerical_cols].fillna(method='ffill').fillna(method='bfill')
                
                # Remove infinite values
                feature_data = feature_data.replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
                
                # Store processed data
                self.processed_data[symbol] = feature_data
                logger.info(f"Prepared {len(numerical_cols)} features for {symbol}")
                
            except Exception as e:
                logger.error(f"Error preparing features for {symbol}: {e}")
                continue
        
        return self.processed_data
    
    def create_sequences(self, symbol: str, target_col: str = 'Close') -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction"""
        if symbol not in self.processed_data:
            raise ValueError(f"No processed data found for {symbol}")
        
        data = self.processed_data[symbol]
        
        # Scale the data
        scaler = RobustScaler()  # More robust to outliers
        scaled_data = scaler.fit_transform(data)
        self.scalers[symbol] = scaler
        
        # Create sequences
        X, y = [], []
        target_idx = data.columns.get_loc(target_col)
        
        for i in range(self.config.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.config.sequence_length:i])
            y.append(scaled_data[i, target_idx])
        
        return np.array(X), np.array(y)
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Split data with proper time series methodology"""
        # Calculate split points
        total_size = len(X)
        train_size = int(total_size * (1 - self.config.test_size - self.config.validation_size))
        val_size = int(total_size * self.config.validation_size)
        
        # Split data chronologically
        X_train = X[:train_size]
        X_val = X[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        
        y_train = y[:train_size]
        y_val = y[train_size:train_size + val_size]
        y_test = y[train_size + val_size:]
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_ensemble_models(self, symbol: str) -> Dict[str, Any]:
        """Train ensemble of models for a symbol"""
        logger.info(f"Training ensemble models for {symbol}...")
        
        # Prepare data
        X, y = self.create_sequences(symbol)
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        models = {}
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        # Callbacks for deep learning models
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
        ]
        multi_model = MultiModelPredictor(self.config)
        # 1. LSTM Model
        try:
            
            lstm_model = multi_model.create_lstm_model(input_shape)
            
            history = lstm_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                callbacks=callbacks,
                verbose=0
            )
            
            models['LSTM'] = {
                'model': lstm_model,
                'history': history,
                'type': 'deep_learning'
            }
            logger.info(f"LSTM model trained for {symbol}")
            
        except Exception as e:
            logger.error(f"Error training LSTM for {symbol}: {e}")
        
        # 2. GRU Model
        try:
            gru_model = multi_model.create_gru_model(input_shape)
            
            history = gru_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                callbacks=callbacks,
                verbose=0
            )
            
            models['GRU'] = {
                'model': gru_model,
                'history': history,
                'type': 'deep_learning'
            }
            logger.info(f"GRU model trained for {symbol}")
            
        except Exception as e:
            logger.error(f"Error training GRU for {symbol}: {e}")
        
        # 3. CNN-LSTM Model
        try:
            cnn_lstm_model = multi_model.create_cnn_lstm_model(input_shape)
            
            history = cnn_lstm_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                callbacks=callbacks,
                verbose=0
            )
            
            models['CNN_LSTM'] = {
                'model': cnn_lstm_model,
                'history': history,
                'type': 'deep_learning'
            }
            logger.info(f"CNN-LSTM model trained for {symbol}")
            
        except Exception as e:
            logger.error(f"Error training CNN-LSTM for {symbol}: {e}")
        
        # 4. Random Forest (for comparison)
        
            # Reshape for traditional ML
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        X_val_2d = X_val.reshape(X_val.shape[0], -1)
        X_test_2d = X_test.reshape(X_test.shape[0], -1)
        try:
            rf_model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train_2d, y_train)
            
            models['RandomForest'] = {
                'model': rf_model,
                'X_train': X_train_2d,
                'X_val': X_val_2d,
                'X_test': X_test_2d,
                'type': 'traditional_ml'
            }
            logger.info(f"Random Forest model trained for {symbol}")
            
        except Exception as e:
            logger.error(f"Error training Random Forest for {symbol}: {e}")
        
        # 5. Gradient Boosting
        try:
            gb_model = GradientBoostingRegressor(
                n_estimators=100,
                random_state=42
            )
            gb_model.fit(X_train_2d, y_train)
            
            models['GradientBoosting'] = {
                'model': gb_model,
                'X_train': X_train_2d,
                'X_val': X_val_2d,
                'X_test': X_test_2d,
                'type': 'traditional_ml'
            }
            logger.info(f"Gradient Boosting model trained for {symbol}")
            
        except Exception as e:
            logger.error(f"Error training Gradient Boosting for {symbol}: {e}")
        
        # Store data for predictions
        models['data'] = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
        
        self.models[symbol] = models
        logger.info(f"Ensemble training completed for {symbol}: {len(models)-1} models trained")
        
        return models
    
    def make_ensemble_predictions(self, symbol: str) -> Dict[str, np.ndarray]:
        """Make predictions using ensemble of models"""
        if symbol not in self.models:
            raise ValueError(f"No trained models found for {symbol}")
        
        models = self.models[symbol]
        data = models['data']
        predictions = {}
        
        # Get predictions from each model
        for model_name, model_info in models.items():
            if model_name == 'data':
                continue
                
            try:
                pred_train, pred_val, pred_test = None, None, None
                if model_info['type'] == 'deep_learning':
                    pred_train = model_info['model'].predict(data['X_train'])
                    pred_val = model_info['model'].predict(data['X_val'])
                    pred_test = model_info['model'].predict(data['X_test'])
                    
                elif model_info['type'] == 'traditional_ml':
                    pred_train = model_info['model'].predict(model_info['X_train'])
                    pred_val = model_info['model'].predict(model_info['X_val'])
                    pred_test = model_info['model'].predict(model_info['X_test'])
                    
                    # Reshape to match deep learning output
                    pred_train = pred_train.reshape(-1, 1)
                    pred_val = pred_val.reshape(-1, 1)
                    pred_test = pred_test.reshape(-1, 1)
                
                predictions[model_name] = {
                    'train': pred_train,
                    'val': pred_val,
                    'test': pred_test
                }
                
            except Exception as e:
                logger.error(f"Error making predictions with {model_name} for {symbol}: {e}")
                continue
        
        # Create ensemble predictions (simple average)
        if predictions:
            ensemble_pred = {}
            for split in ['train', 'val', 'test']:
                split_predictions = [pred[split] for pred in predictions.values()]
                ensemble_pred[split] = np.mean(split_predictions, axis=0)
            
            predictions['Ensemble'] = ensemble_pred
        
        self.predictions[symbol] = predictions
        return predictions
    
    def evaluate_models(self, symbol: str) -> Dict[str, Dict[str, float]]:
        """Evaluate all models for a symbol"""
        if symbol not in self.predictions or symbol not in self.models:
            raise ValueError(f"No predictions or models found for {symbol}")
        
        predictions = self.predictions[symbol]
        actual_data = self.models[symbol]['data']
        
        # Inverse transform actual values
        scaler = self.scalers[symbol]
        target_idx = list(self.processed_data[symbol].columns).index('Close')
        
        # Create dummy arrays for inverse transformation
        def inverse_transform_target(values):
            dummy = np.zeros((len(values), scaler.scale_.shape[0]))
            dummy[:, target_idx] = values.flatten()
            return scaler.inverse_transform(dummy)[:, target_idx]
        
        y_test_actual = inverse_transform_target(actual_data['y_test'])
        
        metrics = {}
        
        for model_name, model_predictions in predictions.items():
            try:
                # Inverse transform predictions
                pred_test = inverse_transform_target(model_predictions['test'])
                
                # Calculate metrics
                mse = mean_squared_error(y_test_actual, pred_test)
                mae = mean_absolute_error(y_test_actual, pred_test)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test_actual, pred_test)
                
                # Calculate directional accuracy
                actual_direction = np.diff(y_test_actual) > 0
                pred_direction = np.diff(pred_test) > 0
                directional_accuracy = np.mean(actual_direction == pred_direction) * 100
                
                # Calculate percentage accuracy
                percentage_error = np.abs((y_test_actual - pred_test) / y_test_actual) * 100
                accuracy_5pct = np.mean(percentage_error <= 5) * 100
                accuracy_10pct = np.mean(percentage_error <= 10) * 100
                
                metrics[model_name] = {
                    'MSE': mse,
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2,
                    'Directional_Accuracy': directional_accuracy,
                    'Accuracy_5pct': accuracy_5pct,
                    'Accuracy_10pct': accuracy_10pct,
                    'Mean_Percentage_Error': np.mean(percentage_error)
                }
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name} for {symbol}: {e}")
                continue
        
        self.evaluation_metrics[symbol] = metrics
        return metrics
    
    def predict_future(self, symbol: str, days: int = 30, model_name: str = 'Ensemble') -> pd.DataFrame:
        """Predict future prices"""
        if symbol not in self.models:
            raise ValueError(f"No trained models found for {symbol}")
        
        # Get the last sequence from processed data
        scaler = self.scalers[symbol]
        data = self.processed_data[symbol]
        last_sequence = scaler.transform(data.iloc[-self.config.sequence_length:].values)
        
        future_predictions = []
        current_sequence = last_sequence.copy()
        
        # Get the appropriate model
        if model_name == 'Ensemble':
            models_to_use = [m for name, m in self.models[symbol].items() 
                           if name != 'data' and m.get('type') == 'deep_learning']
        else:
            models_to_use = [self.models[symbol][model_name]]
        
        for day in range(days):
            day_predictions = []
            
            for model_info in models_to_use:
                if model_info['type'] == 'deep_learning':
                    # Reshape for prediction
                    sequence_input = current_sequence.reshape(1, self.config.sequence_length, -1)
                    pred = model_info['model'].predict(sequence_input, verbose=0)[0, 0]
                    day_predictions.append(pred)
            
            # Average predictions
            next_pred = np.mean(day_predictions) if day_predictions else 0
            future_predictions.append(next_pred)
            
            # Update sequence
            new_row = current_sequence[-1].copy()
            new_row[list(data.columns).index('Close')] = next_pred
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        # Inverse transform predictions
        target_idx = list(data.columns).index('Close')
        dummy_array = np.zeros((len(future_predictions), scaler.scale_.shape[0]))
        dummy_array[:, target_idx] = future_predictions
        future_prices = scaler.inverse_transform(dummy_array)[:, target_idx]
        
        # Create future dates
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq='D')
        
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': future_prices,
            'Symbol': symbol
        })
        
        return future_df
    
    def create_comprehensive_visualization(self, symbol: str, save_plots: bool = False):
        """Create comprehensive visualization dashboard"""
        if symbol not in self.predictions or symbol not in self.evaluation_metrics:
            logger.error(f"No predictions or metrics found for {symbol}")
            return
        
        # Set up the plot style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 24))
        
        # Get data
        predictions = self.predictions[symbol]
        metrics = self.evaluation_metrics[symbol]
        actual_data = self.models[symbol]['data']
        scaler = self.scalers[symbol]
        target_idx = list(self.processed_data[symbol].columns).index('Close')
        
        # Inverse transform function
        def inverse_transform_target(values):
            dummy = np.zeros((len(values), scaler.scale_.shape[0]))
            dummy[:, target_idx] = values.flatten()
            return scaler.inverse_transform(dummy)[:, target_idx]
        
        # Prepare data for plotting
        y_train_actual = inverse_transform_target(actual_data['y_train'])
        y_val_actual = inverse_transform_target(actual_data['y_val'])
        y_test_actual = inverse_transform_target(actual_data['y_test'])
        
        # Create date ranges
        data_dates = self.processed_data[symbol].index[self.config.sequence_length:]
        train_end = len(y_train_actual)
        val_end = train_end + len(y_val_actual)
        
        train_dates = data_dates[:train_end]
        val_dates = data_dates[train_end:val_end]
        test_dates = data_dates[val_end:]
        
        # Plot 1: Main Price Predictions
        plt.subplot(4, 3, 1)
        plt.plot(train_dates, y_train_actual, label='Actual (Train)', alpha=0.7, linewidth=1)
        plt.plot(val_dates, y_val_actual, label='Actual (Val)', alpha=0.7, linewidth=1)
        plt.plot(test_dates, y_test_actual, label='Actual (Test)', alpha=0.8, linewidth=2)
        
        # Plot ensemble predictions
        if 'Ensemble' in predictions:
            ensemble_train = inverse_transform_target(predictions['Ensemble']['train'])
            ensemble_val = inverse_transform_target(predictions['Ensemble']['val'])
            ensemble_test = inverse_transform_target(predictions['Ensemble']['test'])
            
            plt.plot(train_dates, ensemble_train, label='Ensemble (Train)', alpha=0.7, linestyle='--')
            plt.plot(val_dates, ensemble_val, label='Ensemble (Val)', alpha=0.7, linestyle='--')
            plt.plot(test_dates, ensemble_test, label='Ensemble (Test)', alpha=0.8, linestyle='--', linewidth=2)
        
        # Future predictions
        try:
            future_df = self.predict_future(symbol, days=60)
            plt.plot(future_df['Date'], future_df['Predicted_Price'], 
                    label='Future (60 days)', linestyle=':', linewidth=3, color='red')
        except Exception as e:
            logger.warning(f"Could not plot future predictions: {e}")
        
        plt.title(f'{symbol} - Comprehensive Price Prediction', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Model Comparison - Test Set Only
        plt.subplot(4, 3, 2)
        model_colors = ['blue', 'green', 'orange', 'purple', 'brown', 'red']
        for i, (model_name, model_pred) in enumerate(predictions.items()):
            if model_name != 'Ensemble':
                test_pred = inverse_transform_target(model_pred['test'])
                plt.plot(test_dates, test_pred, 
                        label=model_name, alpha=0.7, 
                        color=model_colors[i % len(model_colors)])
        
        plt.plot(test_dates, y_test_actual, label='Actual', color='black', linewidth=2)
        plt.title('Model Comparison - Test Set', fontsize=12, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Prediction vs Actual Scatter (Test Set)
        plt.subplot(4, 3, 3)
        if 'Ensemble' in predictions:
            ensemble_test = inverse_transform_target(predictions['Ensemble']['test'])
            plt.scatter(y_test_actual, ensemble_test, alpha=0.6)
            
            # Perfect prediction line
            min_val = min(y_test_actual.min(), ensemble_test.min())
            max_val = max(y_test_actual.max(), ensemble_test.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            
            # R² score
            r2 = metrics.get('Ensemble', {}).get('R2', 0)
            plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.xlabel('Actual Price ($)')
        plt.ylabel('Predicted Price ($)')
        plt.title('Prediction Accuracy - Ensemble Model', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Model Performance Metrics
        plt.subplot(4, 3, 4)
        model_names = list(metrics.keys())
        rmse_values = [metrics[model]['RMSE'] for model in model_names]
        
        bars = plt.bar(model_names, rmse_values, alpha=0.7)
        plt.title('Model Performance - RMSE', fontsize=12, fontweight='bold')
        plt.ylabel('RMSE')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # Plot 5: Directional Accuracy
        plt.subplot(4, 3, 5)
        dir_accuracy = [metrics[model]['Directional_Accuracy'] for model in model_names]
        bars = plt.bar(model_names, dir_accuracy, alpha=0.7, color='green')
        plt.title('Directional Accuracy (%)', fontsize=12, fontweight='bold')
        plt.ylabel('Accuracy (%)')
        plt.xticks(rotation=45)
        plt.ylim(0, 100)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        # Plot 6: Residuals Analysis
        plt.subplot(4, 3, 6)
        if 'Ensemble' in predictions:
            ensemble_test = inverse_transform_target(predictions['Ensemble']['test'])
            residuals = y_test_actual - ensemble_test
            plt.plot(test_dates, residuals, alpha=0.7)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.title('Residuals - Ensemble Model', fontsize=12, fontweight='bold')
            plt.xlabel('Date')
            plt.ylabel('Residuals ($)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        
        # Plot 7: Error Distribution
        plt.subplot(4, 3, 7)
        if 'Ensemble' in predictions:
            ensemble_test = inverse_transform_target(predictions['Ensemble']['test'])
            percentage_errors = ((y_test_actual - ensemble_test) / y_test_actual) * 100
            plt.hist(percentage_errors, bins=30, alpha=0.7, edgecolor='black')
            plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
            plt.title('Error Distribution (%)', fontsize=12, fontweight='bold')
            plt.xlabel('Percentage Error (%)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
        
        # Plot 8: Feature Importance (if available)
        plt.subplot(4, 3, 8)
        try:
            if 'RandomForest' in self.models[symbol]:
                rf_model = self.models[symbol]['RandomForest']['model']
                feature_names = [f'Feature_{i}' for i in range(len(rf_model.feature_importances_))]
                importances = rf_model.feature_importances_
                
                # Get top 10 features
                top_indices = np.argsort(importances)[-10:]
                top_importances = importances[top_indices]
                top_names = [feature_names[i] for i in top_indices]
                
                plt.barh(range(len(top_importances)), top_importances, alpha=0.7)
                plt.yticks(range(len(top_importances)), top_names)
                plt.title('Top 10 Feature Importance - RF', fontsize=12, fontweight='bold')
                plt.xlabel('Importance')
        except:
            plt.text(0.5, 0.5, 'Feature Importance\nNot Available', 
                    ha='center', va='center', transform=plt.gca().transAxes,
                    fontsize=12)
            plt.title('Feature Importance', fontsize=12, fontweight='bold')
        
        # Plot 9: Training History (for deep learning models)
        plt.subplot(4, 3, 9)
        try:
            if 'LSTM' in self.models[symbol] and 'history' in self.models[symbol]['LSTM']:
                history = self.models[symbol]['LSTM']['history'].history
                epochs = range(1, len(history['loss']) + 1)
                
                plt.plot(epochs, history['loss'], label='Training Loss', alpha=0.7)
                plt.plot(epochs, history['val_loss'], label='Validation Loss', alpha=0.7)
                plt.title('LSTM Training History', fontsize=12, fontweight='bold')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)
        except:
            plt.text(0.5, 0.5, 'Training History\nNot Available', 
                    ha='center', va='center', transform=plt.gca().transAxes,
                    fontsize=12)
            plt.title('Training History', fontsize=12, fontweight='bold')
        
        # Plot 10: Volume Analysis
        plt.subplot(4, 3, 10)
        try:
            volume_data = self.data[symbol]['Volume'].iloc[-len(test_dates):]
            plt.bar(test_dates, volume_data, alpha=0.7, width=1)
            plt.title('Volume Analysis - Test Period', fontsize=12, fontweight='bold')
            plt.xlabel('Date')
            plt.ylabel('Volume')
            plt.xticks(rotation=45)
        except:
            plt.text(0.5, 0.5, 'Volume Data\nNot Available', 
                    ha='center', va='center', transform=plt.gca().transAxes,
                    fontsize=12)
            plt.title('Volume Analysis', fontsize=12, fontweight='bold')
        
        # Plot 11: Volatility Analysis
        plt.subplot(4, 3, 11)
        try:
            if 'Volatility_30' in self.data[symbol].columns:
                volatility_data = self.data[symbol]['Volatility_30'].iloc[-len(test_dates):]
                plt.plot(test_dates, volatility_data, alpha=0.7, color='orange')
                plt.title('30-Day Volatility', fontsize=12, fontweight='bold')
                plt.xlabel('Date')
                plt.ylabel('Volatility')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
        except:
            plt.text(0.5, 0.5, 'Volatility Data\nNot Available', 
                    ha='center', va='center', transform=plt.gca().transAxes,
                    fontsize=12)
            plt.title('Volatility Analysis', fontsize=12, fontweight='bold')
        
        # Plot 12: Performance Summary Table
        plt.subplot(4, 3, 12)
        plt.axis('off')
        
        # Create performance summary table
        summary_data = []
        for model_name in model_names:
            model_metrics = metrics[model_name]
            summary_data.append([
                model_name,
                f"{model_metrics.get('RMSE', 0):.2f}",
                f"{model_metrics.get('R2', 0):.3f}",
                f"{model_metrics.get('Directional_Accuracy', 0):.1f}%",
                f"{model_metrics.get('Accuracy_5pct', 0):.1f}%"
            ])
        
        table = plt.table(cellText=summary_data,
                         colLabels=['Model', 'RMSE', 'R²', 'Dir. Acc.', 'Acc. ±5%'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(model_names) + 1):
            for j in range(5):
                if i == 0:  # Header
                    table[(i, j)].set_facecolor('#4CAF50')
                    table[(i, j)].set_text_props(weight='bold', color='white')
                else:
                    if j % 2 == 0:
                        table[(i, j)].set_facecolor('#f0f0f0')
        
        plt.title('Performance Summary', fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_comprehensive_analysis_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved as {filename}")
        
        plt.show()
    
    def generate_trading_signals(self, symbol: str, confidence_threshold: float = 0.7) -> pd.DataFrame:
        """Generate trading signals based on predictions"""
        if symbol not in self.predictions:
            raise ValueError(f"No predictions found for {symbol}")
        
        # Get future predictions
        future_df = self.predict_future(symbol, days=30)
        current_price = self.data[symbol]['Close'].iloc[-1]
        
        signals = []
        
        for idx, row in future_df.iterrows():
            predicted_price = row['Predicted_Price']
            price_change = (predicted_price - current_price) / current_price * 100
            
            # Generate signal based on price change
            if price_change > 5:
                signal = 'STRONG_BUY'
                confidence = min(0.95, 0.6 + abs(price_change) / 100)
            elif price_change > 2:
                signal = 'BUY'
                confidence = min(0.85, 0.5 + abs(price_change) / 100)
            elif price_change < -5:
                signal = 'STRONG_SELL'
                confidence = min(0.95, 0.6 + abs(price_change) / 100)
            elif price_change < -2:
                signal = 'SELL'
                confidence = min(0.85, 0.5 + abs(price_change) / 100)
            else:
                signal = 'HOLD'
                confidence = 0.6
            
            # Only include signals above confidence threshold
            if confidence >= confidence_threshold:
                signals.append({
                    'Date': row['Date'],
                    'Current_Price': current_price,
                    'Predicted_Price': predicted_price,
                    'Expected_Return': price_change,
                    'Signal': signal,
                    'Confidence': confidence,
                    'Symbol': symbol
                })
        
        return pd.DataFrame(signals)
    
    def save_models(self, directory: str = "saved_models"):
        """Save all trained models"""
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        for symbol in self.models:
            symbol_dir = os.path.join(directory, symbol)
            if not os.path.exists(symbol_dir):
                os.makedirs(symbol_dir)
            
            # Save scalers
            joblib.dump(self.scalers[symbol], os.path.join(symbol_dir, 'scaler.pkl'))
            
            # Save each model
            for model_name, model_info in self.models[symbol].items():
                if model_name == 'data':
                    continue
                
                try:
                    if model_info['type'] == 'deep_learning':
                        model_info['model'].save(os.path.join(symbol_dir, f'{model_name}.h5'))
                    else:
                        joblib.dump(model_info['model'], os.path.join(symbol_dir, f'{model_name}.pkl'))
                    
                    logger.info(f"Saved {model_name} for {symbol}")
                except Exception as e:
                    logger.error(f"Error saving {model_name} for {symbol}: {e}")
            
            # Save metadata
            metadata = {
                'symbol': symbol,
                'config': self.config.__dict__,
                'feature_columns': list(self.processed_data[symbol].columns),
                'data_shape': self.processed_data[symbol].shape,
                'models': list(self.models[symbol].keys()),
                'evaluation_metrics': self.evaluation_metrics.get(symbol, {}),
                'timestamp': dt.datetime.now().isoformat()
            }
            
            with open(os.path.join(symbol_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"All models saved to {directory}")
    
    def run_complete_pipeline(self, symbols: Optional[List[str]] = None, 
                            period: str = '5y', save_models: bool = True, 
                            create_visualizations: bool = True) -> Dict[str, Any]:
        """Run the complete pipeline for all symbols"""
        if symbols is None:
            symbols_to_run = self.symbols
        else:
            symbols_to_run = symbols
        
        results = {}
        logger.info(" STARTING FULL-SCALE STOCK PREDICTION PIPELINE")
        logger.info("=" * 80)
        
        try:
            # Step 1: Load data for all symbols
            logger.info("Step 1: Loading data for all symbols...")
            self.load_data(period=period)
            
            # Step 2: Prepare features
            logger.info("Step 2: Preparing features...")
            self.prepare_features()
            
            # Step 3-6: Train models for each symbol
            for symbol in symbols_to_run :
                if symbol not in self.data:
                    logger.warning(f"Skipping {symbol} - no data available")
                    continue
                
                logger.info(f"\n📈 Processing {symbol}...")
                logger.info("-" * 50)
                
                try:
                    # Train ensemble models
                    logger.info(f"Step 3: Training ensemble models for {symbol}...")
                    models = self.train_ensemble_models(symbol)
                    
                    # Make predictions
                    logger.info(f"Step 4: Making predictions for {symbol}...")
                    predictions = self.make_ensemble_predictions(symbol)
                    
                    # Evaluate models
                    logger.info(f"Step 5: Evaluating models for {symbol}...")
                    metrics = self.evaluate_models(symbol)
                    
                    # Generate trading signals
                    logger.info(f"Step 6: Generating trading signals for {symbol}...")
                    signals = self.generate_trading_signals(symbol)
                    
                    # Create visualizations
                    if create_visualizations:
                        logger.info(f"Step 7: Creating visualizations for {symbol}...")
                        self.create_comprehensive_visualization(symbol)
                    
                    # Store results
                    results[symbol] = {
                        'models_trained': len(models) - 1,  # -1 for 'data'
                        'best_model': max(metrics.keys(), key=lambda x: metrics[x]['R2']),
                        'best_r2': max([m['R2'] for m in metrics.values()]),
                        'best_rmse': min([m['RMSE'] for m in metrics.values()]),
                        'ensemble_accuracy': metrics.get('Ensemble', {}).get('Accuracy_5pct', 0),
                        'trading_signals': len(signals),
                        'metrics': metrics
                    }
                    
                    logger.info(f"✅ {symbol} completed successfully!")
                    logger.info(f"   Best Model: {results[symbol]['best_model']}")
                    logger.info(f"   Best R²: {results[symbol]['best_r2']:.4f}")
                    logger.info(f"   Best RMSE: {results[symbol]['best_rmse']:.2f}")
                    logger.info(f"   Ensemble Accuracy (±5%): {results[symbol]['ensemble_accuracy']:.1f}%")
                    logger.info(f"   Trading Signals Generated: {results[symbol]['trading_signals']}")
                    
                except Exception as e:
                    logger.error(f" Error processing {symbol}: {e}")
                    results[symbol] = {'error': str(e)}
                    continue
            
            # Save models if requested
            if save_models:
                logger.info("\nStep 8: Saving models...")
                self.save_models()
            
            # Print final summary
            logger.info("\n" + "=" * 80)
            logger.info(" PIPELINE COMPLETED!")
            logger.info("=" * 80)
            
            successful_symbols = [s for s in results if 'error' not in results[s]]
            failed_symbols = [s for s in results if 'error' in results[s]]
            
            logger.info(f" Successfully processed: {len(successful_symbols)} symbols")
            logger.info(f" Failed to process: {len(failed_symbols)} symbols")
            
            if successful_symbols:
                logger.info("\n PERFORMANCE SUMMARY:")
                logger.info("-" * 40)
                for symbol in successful_symbols:
                    result = results[symbol]
                    logger.info(f"{symbol}:")
                    logger.info(f"  └─ Best R²: {result['best_r2']:.4f}")
                    logger.info(f"  └─ Best RMSE: ${result['best_rmse']:.2f}")
                    logger.info(f"  └─ Accuracy: {result['ensemble_accuracy']:.1f}%")
            
            return results
            
        except Exception as e:
            logger.error(f" Pipeline failed: {e}")
            return {'error': str(e)}

# Example usage and configuration
if __name__ == "__main__":
    
    print(" FULL-SCALE STOCK PREDICTOR")
    print("=" * 50)
    
    # Configuration
    config = ModelConfig(
        sequence_length=60,
        test_size=0.2,
        validation_size=0.1,
        batch_size=32,
        epochs=50,  # Reduced for demo
        learning_rate=0.001,
        dropout_rate=0.2,
        l2_reg=0.01
    )
    
    # Define symbols to analyze
    symbols = ['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'NVDA']  # Top tech stocks
    # symbols = ['AAPL']  # Start with one for testing
    
    # Initialize predictor
    predictor = FullScaleStockPredictor(symbols, config)
    
    # Run complete pipeline
    results = predictor.run_complete_pipeline(
        period='2y',  # 2 years of data
        save_models=True,
        create_visualizations=True
    )
    
    # Print results summary
    print("\n" + "=" * 50)
    print("FINAL RESULTS SUMMARY")
    print("=" * 50)
    
    for symbol, result in results.items():
        if 'error' not in result:
            print(f"\n{symbol}:")
            print(f"   Models Trained: {result['models_trained']}")
            print(f"   Best Model: {result['best_model']}")
            print(f"   Best R²: {result['best_r2']:.4f}")
            print(f"   Best RMSE: ${result['best_rmse']:.2f}")
            print(f"   Ensemble Accuracy: {result['ensemble_accuracy']:.1f}%")
            print(f"   Trading Signals: {result['trading_signals']}")
        else:
            print(f"\n{symbol}:  {result['error']}")
    
    print("\n Full-Scale Pipeline Completed!")
    
    # Optional: Generate detailed report for best performing stock
    try:
        best_symbol = max([s for s in results if 'error' not in results[s]], 
                         key=lambda x: results[x]['best_r2'])
        print(f"\n Best performing stock: {best_symbol}")
        
        # Generate trading signals for best stock
        signals = predictor.generate_trading_signals(best_symbol)
        if not signals.empty:
            print(f"\n Next 5 Trading Signals for {best_symbol}:")
            print(signals.head().to_string(index=False))
        
        # Future predictions
        future_df = predictor.predict_future(best_symbol, days=30)
        print(f"\n 30-Day Price Prediction for {best_symbol}:")
        print(f"Current Price: ${predictor.data[best_symbol]['Close'].iloc[-1]:.2f}")
        print(f"Predicted Price (30 days): ${future_df['Predicted_Price'].iloc[-1]:.2f}")
        expected_return = ((future_df['Predicted_Price'].iloc[-1] - predictor.data[best_symbol]['Close'].iloc[-1]) 
                          / predictor.data[best_symbol]['Close'].iloc[-1] * 100)
        print(f"Expected Return: {expected_return:.2f}%")
        
    except Exception as e:
        print(f"Error generating detailed report: {e}")