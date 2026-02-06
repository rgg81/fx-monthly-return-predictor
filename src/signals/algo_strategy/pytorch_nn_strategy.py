import pandas as pd
import numpy as np
import random
import optuna
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from strategy import Strategy
from functools import partial


class MLPNet(nn.Module):
    """Multi-layer perceptron network optimized for small datasets"""
    
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout_rate=0.5, activation='relu'):
        super(MLPNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self._get_activation(activation))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer (binary classification)
        layers.append(nn.Linear(prev_dim, 2))
        
        self.network = nn.Sequential(*layers)
    
    def _get_activation(self, activation):
        """Get activation function"""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'elu': nn.ELU(),
            'leaky_relu': nn.LeakyReLU(),
            'selu': nn.SELU()
        }
        return activations.get(activation, nn.ReLU())
    
    def forward(self, x):
        return self.network(x)


class PyTorchNeuralNetOptunaStrategy(Strategy):
    """PyTorch Neural Network strategy with Optuna optimization"""
    
    def __init__(self, n_trials=50, n_splits=6, feature_set="macro_", 
                 feature_frequency="_18", symbol="EURUSD"):
        """
        Initialize PyTorch NN strategy following LGBM pattern
        
        Args:
            n_trials: Number of Optuna optimization trials
            n_splits: Number of time series cross-validation splits
            feature_set: Feature prefix filter (e.g., "macro_", "tech_", "regime_")
            feature_frequency: Feature frequency suffix (e.g., "_18", "_12", "_6")
            symbol: Trading pair symbol (e.g., "EURUSD", "XAUUSD")
        """
        # Initialize base class with step_size=6 (matching LGBM)
        super().__init__(
            symbol=symbol,
            step_size=6,
            feature_set=feature_set,
            feature_frequency=feature_frequency
        )
        
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.models = []  # Multiple models for ensemble
        self.scalers = []  # Scaler per model
        self.features_per_model = []  # Track features used by each model
        self.best_params = None

    def _clean_data(self, df):
        """
        Clean data by removing rows with NaN or infinity values
        PyTorch-specific: Neural networks cannot handle NaN/inf like tree models
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        # Remove rows with infinity values
        df_clean = df_clean[~df_clean.isin([np.inf, -np.inf]).any(axis=1)]
        # Remove rows with NaN values
        df_clean = df_clean.dropna()
        
        if len(df_clean) < len(df):
            print(f"Data cleaning: Removed {len(df) - len(df_clean)} rows with NaN/inf values")
        
        return df_clean

    def fit(self, X, y):
        """
        Train PyTorch Neural Network with Optuna hyperparameter optimization
        
        Args:
            X: Feature DataFrame
            y: Target labels
        """
        
        def objective(trial, InputFeatures, y_label, seed_random):
            """Optuna objective function matching LGBM pattern"""
            
            # Network architecture - simpler for small datasets
            n_layers = trial.suggest_int('n_layers', 2, 4)
            hidden_dims = []
            for i in range(n_layers):
                dim = trial.suggest_int(f'hidden_dim_{i}', 32, 128)
                hidden_dims.append(dim)
            
            # Training hyperparameters
            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
            dropout_rate = trial.suggest_float('dropout_rate', 0.3, 0.7)
            activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'elu', 'leaky_relu'])
            weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
            max_epochs = trial.suggest_int('max_epochs', 30, 100)
            
            # TimeSeriesSplit parameters (matching LGBM)
            max_train_size = trial.suggest_int('max_train_size', 30, 120)
            gap = trial.suggest_int('gap', 0, 10)
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(
                n_splits=self.n_splits,
                max_train_size=max_train_size,
                test_size=self.step_size,
                gap=gap
            )
            
            scores = []
            splits = list(tscv.split(InputFeatures))
            
            # Train on all splits except the last one (validation only)
            for train_idx, val_idx in splits[:-1]:
                X_train, X_val = InputFeatures.iloc[train_idx], InputFeatures.iloc[val_idx]
                y_train, y_val = y_label.iloc[train_idx], y_label.iloc[val_idx]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                # Build and train model
                input_dim = X_train_scaled.shape[1]
                effective_batch_size = min(batch_size, len(X_train) // 3, 32)
                if effective_batch_size < 4:
                    effective_batch_size = min(16, len(X_train) // 2)
                
                net = NeuralNetClassifier(
                    MLPNet,
                    module__input_dim=input_dim,
                    module__hidden_dims=hidden_dims,
                    module__dropout_rate=dropout_rate,
                    module__activation=activation,
                    max_epochs=max_epochs,
                    lr=lr,
                    batch_size=effective_batch_size,
                    optimizer=torch.optim.Adam,
                    optimizer__weight_decay=weight_decay,
                    criterion=nn.CrossEntropyLoss,
                    train_split=None,
                    verbose=0,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                
                net.fit(X_train_scaled.astype(np.float32), y_train.values.astype(np.int64))
                preds = net.predict(X_val_scaled.astype(np.float32))
                scores.append(accuracy_score(y_val, preds))
            
            # Final validation on last split (matching LGBM pattern)
            for train_idx, val_idx in splits[-1:]:
                X_train, X_val = InputFeatures.iloc[train_idx], InputFeatures.iloc[val_idx]
                y_train, y_val = y_label.iloc[train_idx], y_label.iloc[val_idx]
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                input_dim = X_train_scaled.shape[1]
                effective_batch_size = min(batch_size, len(X_train) // 3, 32)
                
                net = NeuralNetClassifier(
                    MLPNet,
                    module__input_dim=input_dim,
                    module__hidden_dims=hidden_dims,
                    module__dropout_rate=dropout_rate,
                    module__activation=activation,
                    max_epochs=max_epochs,
                    lr=lr,
                    batch_size=effective_batch_size,
                    optimizer=torch.optim.Adam,
                    optimizer__weight_decay=weight_decay,
                    criterion=nn.CrossEntropyLoss,
                    train_split=None,
                    verbose=0,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                
                net.fit(X_train_scaled.astype(np.float32), y_train.values.astype(np.int64))
                preds = net.predict(X_val_scaled.astype(np.float32))
                print(f"\n Trial number: {trial.number} Validation classification report:\n", 
                      classification_report(y_val, preds), flush=True)
            
            print(f"\n Trial number: {trial.number} Scores: {scores}\n", flush=True)
            mean_score = np.mean(scores)
            
            return 1 - mean_score  # Minimize error
        
        # Set random seeds
        seed_random = random.randint(1, 100000)
        random.seed(seed_random)
        np.random.seed(seed_random)
        torch.manual_seed(seed_random)
        
        # Prepare features
        X_train = X
        y_train = y
        columns_to_drop = ['Label', 'Date', f'{self.symbol}_Close']
        X_selected = X_train.drop(columns=columns_to_drop)
        
        # Optuna optimization
        objective_func = partial(objective, InputFeatures=X_selected, y_label=y_train, seed_random=seed_random)
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=seed_random)
        )
        study.optimize(objective_func, n_trials=self.n_trials)
        
        best_trial = study.best_trial
        best_params = best_trial.params
        
        # Update best_params
        self.best_params = best_params
        print(f"\nBest PyTorch Neural Network parameters: {self.best_params}")
        
        # Train final model with best parameters
        self._train_final_model(X_selected, y_train, seed_random, best_trial)
        self.fitted = True
    
    def _train_final_model(self, X_selected, y, seed_random, best_trial):
        """Train final model using last split of TimeSeriesSplit (matching LGBM)"""
        
        # Extract hyperparameters
        max_train_size = self.best_params.pop('max_train_size')
        gap = self.best_params.pop('gap')
        
        # Build hidden dimensions
        n_layers = self.best_params['n_layers']
        hidden_dims = [self.best_params[f'hidden_dim_{i}'] for i in range(n_layers)]
        
        # Time series split - use last split only
        tscv = TimeSeriesSplit(
            n_splits=self.n_splits,
            max_train_size=max_train_size,
            test_size=self.step_size,
            gap=gap
        )
        splits = list(tscv.split(X_selected))
        
        for train_idx, val_idx in splits[-1:]:
            X_train, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Build final model
            input_dim = X_train_scaled.shape[1]
            effective_batch_size = min(self.best_params['batch_size'], len(X_train) // 3, 32)
            
            model = NeuralNetClassifier(
                MLPNet,
                module__input_dim=input_dim,
                module__hidden_dims=hidden_dims,
                module__dropout_rate=self.best_params['dropout_rate'],
                module__activation=self.best_params['activation'],
                max_epochs=self.best_params['max_epochs'],
                lr=self.best_params['lr'],
                batch_size=effective_batch_size,
                optimizer=torch.optim.Adam,
                optimizer__weight_decay=self.best_params['weight_decay'],
                criterion=nn.CrossEntropyLoss,
                train_split=None,
                verbose=0,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            # Train model
            model.fit(X_train_scaled.astype(np.float32), y_train.values.astype(np.int64))
            
            # Store model and components
            self.models.append(model)
            self.scalers.append(scaler)
            self.features_per_model.append(X_train.columns.tolist())
            
            # Validation report
            preds = model.predict(X_val_scaled.astype(np.float32))
            print(f"\n Feature Best Selection Trial number: {best_trial.number} Classification report predictions:\n",
                  classification_report(y_val, preds), flush=True)
        
        print(f"\nNeural Network Architecture:")
        print(f"  Input features: {input_dim}")
        print(f"  Hidden layers: {hidden_dims}")
        print(f"  Activation: {self.best_params['activation']}")
        print(f"  Dropout rate: {self.best_params['dropout_rate']:.3f}")
        print(f"  Learning rate: {self.best_params['lr']:.6f}")
        print(f"  Weight decay: {self.best_params['weight_decay']:.6f}")
        print(f"  Batch size: {effective_batch_size}")
        print(f"  Max epochs: {self.best_params['max_epochs']}")
        print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    def generate_signal(self, past_data, current_data):
        """
        Generate trading signals using ensemble of trained models (matching LGBM pattern)
        
        Args:
            past_data: Historical data for training
            current_data: Current data for prediction
            
        Returns:
            Tuple of (signals, amounts) - lists of predictions and trade sizes
        """
        # Filter features using base class method
        past_data_filtered = self._filter_features(past_data)
        current_data_filtered = self._filter_features(current_data)
        
        # Clean data (PyTorch-specific - neural networks can't handle NaN/inf)
        past_data_clean = self._clean_data(past_data_filtered)
        current_data_clean = self._clean_data(current_data_filtered)
        
        if current_data_clean.empty:
            return [0], [10]
        
        # Prepare training data
        columns_to_drop = ['Label', 'Date', f'{self.symbol}_Close']
        X = past_data_clean
        y = past_data_clean['Label']
        
        # Reset and retrain model (matching LGBM)
        self.models = []
        self.scalers = []
        self.features_per_model = []
        self.fit(X, y)
        
        # Prepare prediction data
        X_preds = current_data_clean.drop(columns=columns_to_drop)
        
        preds = []
        amounts = []
        
        # Generate predictions for each row (matching LGBM pattern)
        for _, X_pred in X_preds.iterrows():
            votes = []
            
            # Get prediction from each model in ensemble
            for i, model in enumerate(self.models):
                # Get features for this specific model
                model_features = self.features_per_model[i]
                X_pred_filtered = X_pred[model_features]
                
                # Scale using model's scaler
                X_pred_scaled = self.scalers[i].transform([X_pred_filtered])
                
                # Make prediction
                vote = model.predict(X_pred_scaled.astype(np.float32))[0]
                votes.append(vote)
            
            print(f"Individual model votes: {votes}", flush=True)
            
            # Weighted voting (identical to LGBM logic)
            total_weight = 0
            weighted_signal_sum = 0
            for vote in votes:
                if vote == 1:
                    total_weight += 10
                    weighted_signal_sum += 10
                else:
                    total_weight += 10
                    weighted_signal_sum -= 10
            
            normalized_signal = weighted_signal_sum / total_weight
            
            # Calculate final prediction and amount
            if normalized_signal > 0:
                pred = 1
            else:
                pred = 0
            
            preds.append(pred)
            amounts.append(int(round(min(abs(normalized_signal) * 10, 10))))
        
        return preds, amounts
