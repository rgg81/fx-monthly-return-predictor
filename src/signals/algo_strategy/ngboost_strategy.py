import random
import pandas as pd
import numpy as np
import optuna
from ngboost import NGBoost
from ngboost.learners import default_tree_learner
from ngboost.distns import Bernoulli
from ngboost.scores import LogScore
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from strategy import Strategy
from functools import partial

class NGBoostOptunaStrategy(Strategy):
    """NGBoost strategy with Optuna hyperparameter optimization"""
    
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
            # Simplified hyperparameter search space for NGBoost
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'verbose': False,
                'random_state': seed_random
            }
            
            # Simple tree-specific parameters - more conservative to avoid singular matrix
            tree_params = {
                'max_depth': trial.suggest_int('max_depth', 2, 3),
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'random_state': seed_random
            }
            
            # Optimize max_train_size and gap
            max_train_size = trial.suggest_int('max_train_size', 30, 120)
            gap = trial.suggest_int('gap', 0, 10)
            
            tscv = TimeSeriesSplit(n_splits=self.n_splits, max_train_size=max_train_size, test_size=self.step_size, gap=gap)
            scores = []
            splits = list(tscv.split(InputFeatures))
            
            for train_idx, val_idx in splits[:-1]:
                X_train, X_val = InputFeatures.iloc[train_idx], InputFeatures.iloc[val_idx]
                y_train, y_val = y_label.iloc[train_idx], y_label.iloc[val_idx]
                
                try:
                    # Create base learner with tree parameters
                    base_learner = DecisionTreeRegressor(**tree_params)
                    
                    # Create NGBoost classifier
                    ngb = NGBoost(
                        Base=base_learner,
                        Dist=Bernoulli,
                        Score=LogScore,
                        **params
                    )
                    
                    # Fit the model without validation to avoid singular matrix issues
                    ngb.fit(X_train.values, y_train.values)
                    
                    # Make predictions
                    preds = ngb.predict(X_val.values)
                    scores.append(accuracy_score(y_val, preds))
                    
                except Exception as e:
                    print(f"Error in NGBoost training fold: {e}", flush=True)
                    continue
                    
            # Evaluate on last split
            for train_idx, val_idx in splits[-1:]:
                X_train, X_val = InputFeatures.iloc[train_idx], InputFeatures.iloc[val_idx]
                y_train, y_val = y_label.iloc[train_idx], y_label.iloc[val_idx]
                
                try:
                    base_learner = DecisionTreeRegressor(**tree_params)
                    ngb = NGBoost(
                        Base=base_learner,
                        Dist=Bernoulli,
                        Score=LogScore,
                        **params
                    )
                    ngb.fit(X_train.values, y_train.values)
                    preds = ngb.predict(X_val.values)
                    print(f"\n Trial number: {trial.number} Validation classification report:\n", classification_report(y_val, preds), flush=True)
                except Exception as e:
                    print(f"Error in final fold: {e}", flush=True)
                    
            if not scores:
                return 1.0
                
            print(f"\n Trial number: {trial.number} Scores:{scores}\n", flush=True)
            mean_score = np.mean(scores)
            return 1.0 - mean_score
        
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
        
        # Extract parameters
        tree_params = {
            'max_depth': best_params.pop('max_depth'),
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': seed_random
        }
        
        ngb_params = {
            'n_estimators': best_params.pop('n_estimators'),
            'learning_rate': best_params.pop('learning_rate'),
            'verbose': False,
            'random_state': seed_random
        }
        
        max_train_size = best_params.pop('max_train_size')
        gap = best_params.pop('gap')

        tscv = TimeSeriesSplit(n_splits=self.n_splits, max_train_size=max_train_size, test_size=self.step_size, gap=gap)
        splits = list(tscv.split(X_selected))

        for train_idx, val_idx in splits[-1:]:
            X_train, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
            y_train_split, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            try:
                base_learner = DecisionTreeRegressor(**tree_params)
                ngb = NGBoost(
                    Base=base_learner,
                    Dist=Bernoulli,
                    Score=LogScore,
                    **ngb_params
                )
                ngb.fit(X_train.values, y_train_split.values)
                preds = ngb.predict(X_val.values)
                self.models.append(ngb)
                self.features_per_model.append(X_train.columns.tolist())
                print(f"\n Feature Best Selection Trial number: {best_trial.number} Classification report predictions:\n", classification_report(y_val, preds), flush=True)
            except Exception as e:
                print(f"Error training final model: {e}", flush=True)
                
        self.fitted = True
    def generate_signal(self, past_data, current_data):
        # Use base class feature filtering
        past_data_filtered = self._filter_features(past_data)
        current_data_filtered = self._filter_features(current_data)
        
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
        
        # Iterate over all predictions
        for _, X_pred in X_preds.iterrows():
            votes = []
        
            # Each model gets only its relevant features
            for i, model in enumerate(self.models):
                # Get the features this specific model was trained on
                model_features = self.features_per_model[i]
                
                # Select only those features from X_pred
                X_pred_filtered = X_pred[model_features]
                
                # Make prediction with the filtered features
                try:
                    vote = model.predict(X_pred_filtered.values.reshape(1, -1))[0]
                    votes.append(vote)
                except Exception as e:
                    print(f"Error in model prediction: {e}", flush=True)
                    continue
                    
            print(f"Individual model votes: {votes}", flush=True)
            
            # Calculate weighted signal
            total_weight = 0
            weighted_signal_sum = 0
            for vote in votes:
                if vote == 1:
                    total_weight += 10
                    weighted_signal_sum += 10
                else:
                    total_weight += 10
                    weighted_signal_sum -= 10

            normalized_signal = weighted_signal_sum / total_weight if total_weight > 0 else 0
            
            # Calculate the final prediction and amount
            if normalized_signal > 0:
                pred = 1
            else:
                pred = 0
            preds.append(pred)
            amounts.append(int(round(min(abs(normalized_signal) * 10, 10))))
            
        return preds, amounts
