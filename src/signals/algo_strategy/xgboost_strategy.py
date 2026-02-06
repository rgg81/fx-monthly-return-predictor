import pandas as pd
import numpy as np
import random
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from strategy import Strategy
from functools import partial

class XGBoostOptunaStrategy(Strategy):
    """XGBoost strategy with Optuna hyperparameter optimization"""
    
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
        self.features_per_model = []
        self.best_params = None

    def fit(self, X, y):

        def objective(trial, InputFeatures, y_label, seed_random):
            # Calculate class balance ratio
            negative_count = (y_label == 0).sum()
            positive_count = (y_label == 1).sum()
            scale_pos_weight = negative_count / positive_count if positive_count > 0 else 1.0
            
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.1, 0.3),
                'max_depth': trial.suggest_int('max_depth', 4, 5),
                'booster': trial.suggest_categorical('booster', ['gbtree']),
                'scale_pos_weight': scale_pos_weight,
                'random_state': seed_random,
                'use_label_encoder': False,
                'eval_metric': 'logloss'
            }
            
            # TimeSeriesSplit parameters (matching LGBM)
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
                xgb = XGBClassifier(**params)
                xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                
                preds = xgb.predict(X_val)
                scores.append(accuracy_score(y_val, preds))
            
            # Final validation on last split
            for train_idx, val_idx in splits[-1:]:
                X_train, X_val = InputFeatures.iloc[train_idx], InputFeatures.iloc[val_idx]
                y_train, y_val = y_label.iloc[train_idx], y_label.iloc[val_idx]
                xgb = XGBClassifier(**params)
                xgb.fit(X_train, y_train, verbose=False)
                preds = xgb.predict(X_val)
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

        objective_func = partial(objective, InputFeatures=X_selected, y_label=y_train, seed_random=seed_random)
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed_random))
        study.optimize(objective_func, n_trials=self.n_trials)
        best_trial = study.best_trial
        best_params = best_trial.params
        
        # Calculate class balance for final model
        negative_count = (y_train == 0).sum()
        positive_count = (y_train == 1).sum()
        scale_pos_weight = negative_count / positive_count if positive_count > 0 else 1.0
        
        best_params.update({
            'scale_pos_weight': scale_pos_weight,
            'random_state': seed_random,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        })
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
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            xgb = XGBClassifier(**best_params)
            xgb.fit(X_train, y_train, verbose=False)
            preds = xgb.predict(X_val)
            self.models.append(xgb)
            self.features_per_model.append(X_train.columns.tolist())
            print(f"\n Feature Best Selection Trial number: {best_trial.number} Classification report predictions:\n", 
                  classification_report(y_val, preds), flush=True)
        self.fitted = True
        

    def generate_signal(self, past_data, current_data):
        # Use base class feature filtering
        past_data_filtered = self._filter_features(past_data)
        current_data_filtered = self._filter_features(current_data)
        
        # No data cleaning needed - XGBoost handles NaN/inf natively
        columns_to_drop = ['Label', 'Date', f'{self.symbol}_Close']
        X = past_data_filtered
        y = past_data_filtered['Label']
        
        # Reset and retrain
        self.models = []
        self.features_per_model = []
        self.fit(X, y)
        
        # X_pred should be the same features as X
        X_preds = current_data_filtered.drop(columns=columns_to_drop)
        preds = []
        amounts = []
        
        # Generate predictions for each row
        for _, X_pred in X_preds.iterrows():
            votes = []
            
            # Each model gets only its relevant features
            for i, model in enumerate(self.models):
                # Get the features this specific model was trained on
                model_features = self.features_per_model[i]
                
                # Select only those features from X_pred
                X_pred_filtered = X_pred[model_features]
                
                # Make prediction with the filtered features
                vote = model.predict([X_pred_filtered])[0]
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
