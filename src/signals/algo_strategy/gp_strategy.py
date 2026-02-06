import pandas as pd
import numpy as np
import random
import optuna
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from strategy import Strategy
from functools import partial

class GaussianProcessOptunaStrategy(Strategy):
    """Gaussian Process strategy with Optuna hyperparameter optimization"""
    
    def __init__(self, n_trials=50, n_splits=6, feature_set="macro_", feature_frequency="_18", symbol="EURUSD"):
        # Initialize base class with step_size=6
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
        self.is_fallback = False

    def _clean_data(self, df):
        """Clean data by removing rows with NaN or infinity values"""
        df_clean = df.copy()
        # Remove rows with infinity values
        df_clean = df_clean[~df_clean.isin([np.inf, -np.inf]).any(axis=1)]
        # Remove rows with NaN values
        df_clean = df_clean.dropna()
        
        if len(df_clean) < len(df):
            print(f"Data cleaning: Removed {len(df) - len(df_clean)} rows with NaN/inf values")
        
        return df_clean

    def fit(self, X, y):

        def objective(trial, InputFeatures, y_label, seed_random):
            # Simplified hyperparameter space for speed
            kernel_type = trial.suggest_categorical('kernel_type', ['rbf', 'matern'])
            length_scale = trial.suggest_float('length_scale', 0.5, 5.0)
            
            if kernel_type == 'rbf':
                kernel = RBF(length_scale=length_scale)
            else:  # matern
                nu = trial.suggest_categorical('nu', [1.5, 2.5])
                kernel = Matern(length_scale=length_scale, nu=nu)
            
            max_iter_predict = trial.suggest_int('max_iter_predict', 10, 20)
            
            # TimeSeriesSplit parameters (matching LGBM)
            max_train_size = trial.suggest_int('max_train_size', 30, 60)
            gap = trial.suggest_int('gap', 0, 10)
            
            # Limit samples for GP efficiency
           
            InputFeatures_sampled = InputFeatures
            y_label_sampled = y_label
            
            tscv = TimeSeriesSplit(
                n_splits=self.n_splits,
                max_train_size=max_train_size,
                test_size=self.step_size,
                gap=gap
            )
            scores = []
            splits = list(tscv.split(InputFeatures_sampled))
            
            # Train on all splits except the last one
            for train_idx, val_idx in splits[:-1]:
                try:
                    X_train, X_val = InputFeatures_sampled.iloc[train_idx], InputFeatures_sampled.iloc[val_idx]
                    y_train, y_val = y_label_sampled.iloc[train_idx], y_label_sampled.iloc[val_idx]
                    
                    if len(y_val) < 2:
                        continue
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    
                    # Train GP classifier
                    gp = GaussianProcessClassifier(
                        kernel=kernel,
                        max_iter_predict=max_iter_predict,
                        optimizer=None
                    )
                    gp.fit(X_train_scaled, y_train)
                    preds = gp.predict(X_val_scaled)
                    scores.append(accuracy_score(y_val, preds))
                    
                except Exception as e:
                    print(f"Error in GP fold: {e}")
                    continue
            
            # Final validation on last split
            for train_idx, val_idx in splits[-1:]:
                try:
                    X_train, X_val = InputFeatures_sampled.iloc[train_idx], InputFeatures_sampled.iloc[val_idx]
                    y_train, y_val = y_label_sampled.iloc[train_idx], y_label_sampled.iloc[val_idx]

                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    
                    gp = GaussianProcessClassifier(
                        kernel=kernel,
                        max_iter_predict=max_iter_predict,
                        optimizer=None
                    )
                    gp.fit(X_train_scaled, y_train)
                    preds = gp.predict(X_val_scaled)
                    print(f"\n Trial number: {trial.number} Validation classification report:\n",
                          classification_report(y_val, preds), flush=True)
                except Exception as e:
                    print(f"Error in GP validation: {e}")
            
            print(f"\n Trial number: {trial.number} Scores: {scores}\n", flush=True)
            
            if len(scores) == 0:
                return 1.0
            
            mean_score = np.mean(scores)
            return 1 - mean_score
        
        # Set random seeds
        seed_random = random.randint(1, 100000)
        random.seed(seed_random)
        np.random.seed(seed_random)
        
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
        
        self.best_params = best_params
        print(f"\nBest Gaussian Process parameters: {self.best_params}")
        
        # Train final model with best parameters
        self._train_final_model(X_selected, y_train, seed_random, best_trial)
        self.fitted = True
    
    def _train_final_model(self, X_selected, y, seed_random, best_trial):
        """Train final model using last split of TimeSeriesSplit"""
        
        # Extract hyperparameters
        max_train_size = self.best_params.pop('max_train_size')
        gap = self.best_params.pop('gap')
        
        # Build kernel
        kernel_type = self.best_params['kernel_type']
        length_scale = self.best_params['length_scale']
        
        if kernel_type == 'rbf':
            best_kernel = RBF(length_scale=length_scale)
        else:  # matern
            nu = self.best_params['nu']
            best_kernel = Matern(length_scale=length_scale, nu=nu)
        
        X_selected_sampled = X_selected
        y_sampled = y
        
        # Time series split - use last split only
        tscv = TimeSeriesSplit(
            n_splits=self.n_splits,
            max_train_size=max_train_size,
            test_size=self.step_size,
            gap=gap
        )
        splits = list(tscv.split(X_selected_sampled))
        
        for train_idx, val_idx in splits[-1:]:
            X_train, X_val = X_selected_sampled.iloc[train_idx], X_selected_sampled.iloc[val_idx]
            y_train, y_val = y_sampled.iloc[train_idx], y_sampled.iloc[val_idx]
                    
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Build final model
            model = GaussianProcessClassifier(
                kernel=best_kernel,
                max_iter_predict=self.best_params['max_iter_predict'],
                optimizer=None
            )
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Store model and components
            self.models.append(model)
            self.scalers.append(scaler)
            self.features_per_model.append(X_train.columns.tolist())
            
            # Validation report
            preds = model.predict(X_val_scaled)
            print(f"\n Feature Best Selection Trial number: {best_trial.number} Classification report predictions:\n",
                  classification_report(y_val, preds), flush=True)
        
        print(f"\nGaussian Process Configuration:")
        print(f"  Kernel: {kernel_type}")
        print(f"  Length scale: {length_scale:.3f}")
        if kernel_type == 'matern':
            print(f"  Nu: {self.best_params['nu']}")
        print(f"  Max iter predict: {self.best_params['max_iter_predict']}")
        print(f"  Training samples: {len(X_train)}")

    def generate_signal(self, past_data, current_data):
        """Generate trading signals using ensemble of trained models"""
        # Filter features using base class method
        past_data_filtered = self._filter_features(past_data)
        current_data_filtered = self._filter_features(current_data)
        
        # Clean data (GP requires clean data like PyTorch)
        past_data_clean = self._clean_data(past_data_filtered)
        current_data_clean = self._clean_data(current_data_filtered)
        
        if current_data_clean.empty:
            return [0], [10]
        
        # Prepare training data
        columns_to_drop = ['Label', 'Date', f'{self.symbol}_Close']
        X = past_data_clean
        y = past_data_clean['Label']
        
        # Reset and retrain model
        self.models = []
        self.scalers = []
        self.features_per_model = []
        self.fit(X, y)
        
        # Prepare prediction data
        X_preds = current_data_clean.drop(columns=columns_to_drop)
        
        preds = []
        amounts = []
        
        # Generate predictions for each row
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
                vote = model.predict(X_pred_scaled)[0]
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
