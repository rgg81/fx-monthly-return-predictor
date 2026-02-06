import pandas as pd
import numpy as np
import random
import optuna
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from strategy import Strategy
from functools import partial

class GaussianNBOptunaStrategy(Strategy):
    """Gaussian Naive Bayes strategy with Optuna hyperparameter optimization"""
    
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
        self.models = []
        self.scalers = []
        self.features_per_model = []
        self.best_params = None

    def _clean_data(self, df):
        """Remove NaN and infinity values - Naive Bayes requires clean data"""
        df_clean = df.copy()
        df_clean = df_clean[~df_clean.isin([np.inf, -np.inf]).any(axis=1)]
        df_clean = df_clean.dropna()
        return df_clean

    def fit(self, X, y):

        def objective(trial, InputFeatures, y_label, seed_random):
            params = {
                'var_smoothing': trial.suggest_float('var_smoothing', 1e-12, 1.0, log=True)
            }
            
            # TimeSeriesSplit parameters (matching PyTorch)
            max_train_size = trial.suggest_int('max_train_size', 30, 120)
            gap = trial.suggest_int('gap', 0, 10)
            
            tscv = TimeSeriesSplit(
                n_splits=self.n_splits,
                max_train_size=max_train_size,
                test_size=self.step_size,
                gap=gap
            )
            scores = []
            splits = list(tscv.split(InputFeatures))
            
            # Train on all splits except the last one
            for train_idx, val_idx in splits[:-1]:
                X_train, X_val = InputFeatures.iloc[train_idx], InputFeatures.iloc[val_idx]
                y_train, y_val = y_label.iloc[train_idx], y_label.iloc[val_idx]
                
                # Scale data for Naive Bayes
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                nb = GaussianNB(**params)
                nb.fit(X_train_scaled, y_train)
                preds = nb.predict(X_val_scaled)
                scores.append(accuracy_score(y_val, preds))
            
            # Final validation on last split
            for train_idx, val_idx in splits[-1:]:
                X_train, X_val = InputFeatures.iloc[train_idx], InputFeatures.iloc[val_idx]
                y_train, y_val = y_label.iloc[train_idx], y_label.iloc[val_idx]
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                nb = GaussianNB(**params)
                nb.fit(X_train_scaled, y_train)
                preds = nb.predict(X_val_scaled)
                print(f"\n Trial number: {trial.number} Validation classification report:\n", 
                      classification_report(y_val, preds), flush=True)
            
            print(f"\n Trial number: {trial.number} Scores: {scores}\n", flush=True)
            mean_score = np.mean(scores)
            
            return 1 - mean_score
        
        seed_random = random.randint(1, 100000)
        random.seed(seed_random)
        np.random.seed(seed_random)
        X_train = X
        y_train = y
        X_selected = X_train
        columns_to_drop = ['Label', 'Date', f'{self.symbol}_Close']
        X_selected = X_selected.drop(columns=columns_to_drop)
        
        # Clean data before optimization
        X_selected = self._clean_data(X_selected)
        y_train = y_train.loc[X_selected.index]

        objective_func = partial(objective, InputFeatures=X_selected, y_label=y_train, seed_random=seed_random)
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed_random))
        study.optimize(objective_func, n_trials=self.n_trials)
        best_trial = study.best_trial
        best_params = best_trial.params
        max_train_size = best_params.pop('max_train_size')
        gap = best_params.pop('gap')

        tscv = TimeSeriesSplit(
            n_splits=self.n_splits,
            max_train_size=max_train_size,
            test_size=self.step_size,
            gap=gap
        )
        splits = list(tscv.split(X_selected))

        for train_idx, val_idx in splits[-1:]:
            X_train, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
            y_train, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Create and fit scaler for this model
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            nb = GaussianNB(**best_params)
            nb.fit(X_train_scaled, y_train)
            preds = nb.predict(X_val_scaled)
            
            self.models.append(nb)
            self.scalers.append(scaler)
            self.features_per_model.append(X_train.columns.tolist())
            print(f"\n Feature Best Selection Trial number: {best_trial.number} Classification report predictions:\n", 
                  classification_report(y_val, preds), flush=True)
        self.fitted = True

    def generate_signal(self, past_data, current_data):
        # Use base class feature filtering
        past_data_filtered = self._filter_features(past_data)
        current_data_filtered = self._filter_features(current_data)
        
        # Clean data - Naive Bayes requires clean data
        past_data_clean = self._clean_data(past_data_filtered)
        current_data_clean = self._clean_data(current_data_filtered)
        
        columns_to_drop = ['Label', 'Date', f'{self.symbol}_Close']
        X = past_data_clean
        y = past_data_clean['Label']
        
        # Reset and retrain
        self.models = []
        self.scalers = []
        self.features_per_model = []
        self.fit(X, y)
        
        # X_pred should be the same features as X
        X_preds = current_data_clean.drop(columns=columns_to_drop)
        preds = []
        amounts = []
        
        # Generate predictions for each row
        for _, X_pred in X_preds.iterrows():
            votes = []
            
            # Each model gets only its relevant features
            for i, (model, scaler) in enumerate(zip(self.models, self.scalers)):
                # Get the features this specific model was trained on
                model_features = self.features_per_model[i]
                
                # Select only those features from X_pred
                X_pred_filtered = X_pred[model_features]
                
                # Scale the features
                X_pred_scaled = scaler.transform([X_pred_filtered])
                
                # Make prediction with the scaled features
                vote = model.predict(X_pred_scaled)[0]
                votes.append(vote)
            
            print(f"Individual model votes: {votes}", flush=True)
            
            # Weighted voting (identical to PyTorch logic)
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