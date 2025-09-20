import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
from datetime import datetime, timedelta
import logging
from typing import Dict, Tuple, Any, Optional


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AirQualityPredictor:
    """Main ML Model for Air Quality Prediction"""
    def __init__(self, model_dir: str = 'models') -> None:
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.model_metadata = {}
        self.ensure_model_dir()

    def ensure_model_dir(self):
        os.makedirs(self.model_dir, exist_ok=True)

    def prepare_features(self, df: pd.DataFrame):
        feature_columns = ['temperature',
                           'pressure',
                           'humidity',
                           'wind_speed','day_of_week',
                           'month', 'latitude', 'longitude']

        df_processed = df.copy()
        df_processed = df_processed.sort_values(['city','date'])
        df_processed['pm2_5_lag1'] = df_processed.groupby('city')['pm2_5'].shift(1)

        df_processed['pm2_5_ma3'] = df_processed.groupby('city')['pm2_5'].rolling(window=3).mean().reset_index(0,
                                                                                                               drop=True)
        df_processed['pm2_5_ma7'] = df_processed.groupby('city')['pm2_5'].rolling(window=7).mean().reset_index(0,
                                                                                                               drop=True)

        # Add city encoding (one-hot)
        df_processed = pd.get_dummies(df_processed, columns=['city'], prefix='city')

        # Update feature columns
        city_cols = [col for col in df_processed.columns if col.startswith('city_')]
        feature_columns.extend(city_cols)
        feature_columns.extend(['pm2_5_lag1', 'pm2_5_ma3', 'pm2_5_ma7'])

        # Remove rows with NaN values (due to lag features)
        df_processed = df_processed.dropna()

        self.feature_columns = feature_columns
        logger.info(f"Prepared {len(feature_columns)} features: {feature_columns}")

        return df_processed


    def train_model(self, df: pd.DataFrame, model_type: str = 'random_forest',
                    test_size: float = 0.2) -> Dict[str, float]:
        df_processed = self.prepare_features(df)

        X = df_processed[self.feature_columns]
        Y = df_processed['pm2_5']

        logger.info(f"Training {model_type} model")
        logger.info(f'Training Data Shape: {X.shape}, {Y.shape}')

        #Train-Test split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= test_size, random_state=42, stratify=df_processed['city'].str.split('_').str[1] if 'city_' in X.columns else None)

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        if model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100,
                                               max_depth=15,
                                               min_samples_split=5,
                                               random_state=42,
                                               n_jobs=2
                                               )
        elif model_type == 'linear_regression':
            self.model = LinearRegression()
        else:
            raise ValueError("Invalid model type")

        self.model.fit(X_train_scaled, Y_train)

        #Predictions
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)

        #metrics
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(Y_train, y_pred_train)),
            'train_mae': mean_absolute_error(Y_train, y_pred_train),
            'test_rmse': np.sqrt(mean_squared_error(Y_test, y_pred_test)),
            'test_mae': mean_absolute_error(Y_test, y_pred_test),
            'train_r2': r2_score(Y_train, y_pred_train),
            'test_r2': r2_score(Y_test, y_pred_test)
        }

        #cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, Y_train, cv=5, scoring='neg_mean_squared_error')
        metrics['cv_rmse'] = np.sqrt(-cv_scores.mean())
        metrics['cv_rmse_std'] = np.sqrt(cv_scores.std())

        #store metadata
        self.model_metadata = {
            'model_type': model_type,
            'training_date': datetime.now().isoformat(),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features': self.feature_columns,
            'metrics': metrics
        }

        logger.info("Training completed!")
        logger.info(f"Test RMSE: {metrics['test_rmse']:.2f}")
        logger.info(f"Test R²: {metrics['test_r2']:.3f}")

        return metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.scaler is None:
            raise ValueError("Model must be trained before predicting")

        df_processed = self.prepare_features(df)
        X = df_processed[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return predictions

    def save_model(self, model_name: str = 'air_quality_prediction') -> str :
        if self.model is None:
            raise ValueError("Model must be trained before saving")

        timestamp = datetime.now().isoformat()
        model_filename = f"{model_name}_{timestamp}.pkl"
        metadata_filename = f"{model_name}_{timestamp}_metadata.json"

        model_path = os.path.join(self.model_dir, model_filename)
        metadata_path = os.path.join(self.model_dir, metadata_filename)

        # Save model and scaler
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }, model_path)

        #Save Metadata
        import json
        with open(metadata_path, 'w') as f:
            json.dump(self.model_metadata, f, indent=2)

        logger.info(f"Model saved to {model_path}")
        logger.info(f"Metadata saved to {metadata_path}")

        return model_path

    def load_model(self, model_path: str):
        """Load a saved model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']

        logger.info(f"Model loaded from: {model_path}")

def main():
    try:
        from core.data_manager import DataManager
    except ImportError:
        # Fallback: add src to path
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from core.data_manager import DataManager

    # Generate test data
    dm = DataManager()
    df = dm.generate_synthetic_data(60)  # 60 days for better training
    print(f"✅ Generated data: {df.shape}")

    predictor = AirQualityPredictor()

    metrics_rf = predictor.train_model(df, model_type='random_forest')

    print('Random Forest Results:')
    print(f"   Test RMSE: {metrics_rf['test_rmse']:.2f} µg/m³")
    print(f"   Test R²: {metrics_rf['test_r2']:.3f}")
    print(f"   CV RMSE: {metrics_rf['cv_rmse']:.2f} ± {metrics_rf['cv_rmse_std']:.2f}")

    model_path = predictor.save_model('random_forest_v1')

    recent_data = df.tail(10)
    predictions = predictor.predict(recent_data)
    for i, (idx, row) in enumerate(recent_data.iterrows()):
        print(f"   {row['city']}: Actual={row['pm2_5']:.1f}, Predicted={predictions[i]:.1f} µg/m³")

    print("\n✅ ML MODEL TEST COMPLETE!")

    return predictor, metrics_rf

if __name__ == "__main__":
    predictor, metrics = main()






