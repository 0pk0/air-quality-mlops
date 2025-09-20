"""
Model Evaluation and Visualization
Comprehensive model performance analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate and visualize model performance"""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        self._ensure_output_dir()

        # Set style for plots
        plt.style.use('default')
        sns.set_palette("husl")

    def _ensure_output_dir(self):
        """Create output directory"""
        os.makedirs(self.output_dir, exist_ok=True)

    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray,
                             city_labels: List[str] = None) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""

        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'max_error': np.max(np.abs(y_true - y_pred))
        }

        # City-specific metrics if labels provided
        if city_labels is not None:
            city_metrics = {}
            unique_cities = list(set(city_labels))

            for city in unique_cities:
                city_mask = np.array(city_labels) == city
                y_true_city = y_true[city_mask]
                y_pred_city = y_pred[city_mask]

                city_metrics[city] = {
                    'rmse': np.sqrt(mean_squared_error(y_true_city, y_pred_city)),
                    'mae': mean_absolute_error(y_true_city, y_pred_city),
                    'r2': r2_score(y_true_city, y_pred_city),
                    'count': len(y_true_city)
                }

            metrics['city_metrics'] = city_metrics

        return metrics

    def plot_predictions_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   title: str = "Predictions vs Actual",
                                   save_name: str = "predictions_vs_actual.png"):
        """Create scatter plot of predictions vs actual values"""

        plt.figure(figsize=(10, 8))

        # Main scatter plot
        plt.scatter(y_true, y_pred, alpha=0.6, s=50)

        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

        # Calculate R²
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        plt.xlabel('Actual PM2.5 (µg/m³)', fontsize=12)
        plt.ylabel('Predicted PM2.5 (µg/m³)', fontsize=12)
        plt.title(f'{title}\nR² = {r2:.3f}, RMSE = {rmse:.2f} µg/m³', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save plot
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        logger.info(f"Plot saved: {save_path}")

        return save_path

    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                       save_name: str = "residuals_analysis.png"):
        """Create residuals analysis plots"""

        residuals = y_true - y_pred

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Histogram of residuals
        axes[0, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Residuals')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Q-Q plot (approximation)
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot of Residuals')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Absolute residuals vs predicted
        axes[1, 1].scatter(y_pred, np.abs(residuals), alpha=0.6)
        axes[1, 1].set_xlabel('Predicted Values')
        axes[1, 1].set_ylabel('|Residuals|')
        axes[1, 1].set_title('Absolute Residuals vs Predicted')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        logger.info(f"Residuals plot saved: {save_path}")

        return save_path

    def plot_city_performance(self, df: pd.DataFrame, y_true: np.ndarray,
                              y_pred: np.ndarray, save_name: str = "city_performance.png"):
        """Plot performance metrics by city"""

        # Prepare data
        results_df = df.copy()
        results_df['actual'] = y_true
        results_df['predicted'] = y_pred
        results_df['error'] = y_true - y_pred
        results_df['abs_error'] = np.abs(results_df['error'])

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Average error by city
        city_metrics = results_df.groupby('city').agg({
            'error': ['mean', 'std'],
            'abs_error': 'mean',
            'actual': 'mean',
            'predicted': 'mean'
        }).round(2)

        city_names = city_metrics.index
        mean_errors = city_metrics[('error', 'mean')]

        axes[0, 0].bar(city_names, mean_errors)
        axes[0, 0].set_xlabel('City')
        axes[0, 0].set_ylabel('Mean Error (µg/m³)')
        axes[0, 0].set_title('Mean Error by City')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Box plot of errors by city
        results_df.boxplot(column='error', by='city', ax=axes[0, 1])
        axes[0, 1].set_xlabel('City')
        axes[0, 1].set_ylabel('Error (µg/m³)')
        axes[0, 1].set_title('Error Distribution by City')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # 3. Actual vs Predicted by city
        for city in results_df['city'].unique():
            city_data = results_df[results_df['city'] == city]
            axes[1, 0].scatter(city_data['actual'], city_data['predicted'],
                               label=city, alpha=0.7)

        # Perfect prediction line
        min_val = results_df[['actual', 'predicted']].min().min()
        max_val = results_df[['actual', 'predicted']].max().max()
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[1, 0].set_xlabel('Actual PM2.5 (µg/m³)')
        axes[1, 0].set_ylabel('Predicted PM2.5 (µg/m³)')
        axes[1, 0].set_title('Actual vs Predicted by City')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. MAE by city
        mae_by_city = results_df.groupby('city')['abs_error'].mean()
        axes[1, 1].bar(mae_by_city.index, mae_by_city.values)
        axes[1, 1].set_xlabel('City')
        axes[1, 1].set_ylabel('Mean Absolute Error (µg/m³)')
        axes[1, 1].set_title('MAE by City')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        logger.info(f"City performance plot saved: {save_path}")

        return save_path

    def generate_report(self, metrics: Dict, model_name: str = "Air Quality Model") -> str:
        """Generate a comprehensive evaluation report"""

        report_lines = [
            f"# {model_name} - Evaluation Report",
            f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Overall Performance Metrics",
            f"- **RMSE**: {metrics['rmse']:.2f} µg/m³",
            f"- **MAE**: {metrics['mae']:.2f} µg/m³",
            f"- **R²**: {metrics['r2']:.3f}",
            f"- **MAPE**: {metrics['mape']:.1f}%",
            f"- **Max Error**: {metrics['max_error']:.2f} µg/m³",
            ""
        ]

        # Add city-specific metrics if available
        if 'city_metrics' in metrics:
            report_lines.extend([
                "## City-Specific Performance",
                ""
            ])

            for city, city_metrics in metrics['city_metrics'].items():
                report_lines.extend([
                    f"### {city}",
                    f"- RMSE: {city_metrics['rmse']:.2f} µg/m³",
                    f"- MAE: {city_metrics['mae']:.2f} µg/m³",
                    f"- R²: {city_metrics['r2']:.3f}",
                    f"- Samples: {city_metrics['count']}",
                    ""
                ])

        # Performance interpretation
        report_lines.extend([
            "## Performance Interpretation",
            ""
        ])

        if metrics['r2'] > 0.8:
            report_lines.append("✅ **Excellent** model performance (R² > 0.8)")
        elif metrics['r2'] > 0.6:
            report_lines.append("🟡 **Good** model performance (R² > 0.6)")
        else:
            report_lines.append("🔴 **Needs improvement** (R² < 0.6)")

        if metrics['rmse'] < 5:
            report_lines.append("✅ **Low** prediction error (RMSE < 5 µg/m³)")
        elif metrics['rmse'] < 10:
            report_lines.append("🟡 **Moderate** prediction error (RMSE < 10 µg/m³)")
        else:
            report_lines.append("🔴 **High** prediction error (RMSE > 10 µg/m³)")

        # Save report
        report_path = os.path.join(self.output_dir, "evaluation_report.md")
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))

        logger.info(f"Evaluation report saved: {report_path}")

        return report_path


def main():
    """Test the ModelEvaluator"""
    print("📊 TESTING MODEL EVALUATION FRAMEWORK")
    print("=" * 40)

    # Import required modules
    import sys
    sys.path.append('src/core')
    sys.path.append('src/models')

    from data_manager import DataManager
    from base_model import AirQualityPredictor

    # Generate and prepare data
    dm = DataManager()
    df = dm.generate_synthetic_data(60)

    # Train model
    predictor = AirQualityPredictor()
    metrics = predictor.train_model(df)

    # Prepare test data
    df_processed = predictor.prepare_features(df)
    test_data = df_processed.tail(20)  # Last 20 records for testing

    y_true = test_data['pm2_5'].values
    y_pred = predictor.predict(test_data)
    city_labels = test_data['city'].values if 'city' in test_data.columns else None

    # Initialize evaluator
    evaluator = ModelEvaluator()

    # Evaluate predictions
    eval_metrics = evaluator.evaluate_predictions(y_true, y_pred, city_labels)

    print(f"\n📊 Evaluation Results:")
    print(f"   RMSE: {eval_metrics['rmse']:.2f} µg/m³")
    print(f"   R²: {eval_metrics['r2']:.3f}")
    print(f"   MAPE: {eval_metrics['mape']:.1f}%")

    # Generate visualizations
    print(f"\n📈 Generating visualization plots...")
    evaluator.plot_predictions_vs_actual(y_true, y_pred)
    evaluator.plot_residuals(y_true, y_pred)

    # Generate report
    report_path = evaluator.generate_report(eval_metrics, "Random Forest Air Quality Model")
    print(f"✅ Report generated: {report_path}")

    print("\n✅ MODEL EVALUATION TEST COMPLETE!")

    return evaluator, eval_metrics


if __name__ == "__main__":
    evaluator, metrics = main()
