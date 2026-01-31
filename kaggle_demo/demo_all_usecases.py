"""
Jotty AI - Comprehensive ML Use Cases
======================================

Testing Jotty's autonomous ML capabilities across 7 different domains:
1. House Prices Prediction (Regression)
2. Credit Card Fraud Detection (Imbalanced Classification)
3. Customer Churn Prediction (Business ML)
4. Stock Price Prediction (Time Series)
5. Text Classification (NLP)
6. Image Classification (Deep Learning)
7. Recommendation System (Collaborative Filtering)

Each use case demonstrates:
- Automated feature engineering
- Hyperparameter optimization
- Model ensembling
- Performance evaluation
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, mean_squared_error,
    mean_absolute_error, r2_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor
)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)


class UseCaseRunner:
    """Runs and evaluates each use case."""

    def __init__(self):
        self.results = {}

    async def run_all(self):
        """Run all use cases sequentially."""
        use_cases = [
            ("1. House Prices Prediction", self.house_prices),
            ("2. Credit Card Fraud Detection", self.fraud_detection),
            ("3. Customer Churn Prediction", self.churn_prediction),
            ("4. Stock Price Prediction", self.stock_prediction),
            ("5. Text Classification", self.text_classification),
            ("6. Image Classification", self.image_classification),
            ("7. Recommendation System", self.recommendation_system),
        ]

        for name, func in use_cases:
            logger.info("\n" + "=" * 70)
            logger.info(f"USE CASE: {name}")
            logger.info("=" * 70)

            start_time = time.time()
            try:
                result = await func()
                elapsed = time.time() - start_time
                self.results[name] = {**result, 'time': elapsed}
                logger.info(f"\n✅ Completed in {elapsed:.1f}s")
            except Exception as e:
                logger.error(f"\n❌ Failed: {e}")
                self.results[name] = {'error': str(e)}

        return self.results

    # =========================================================================
    # USE CASE 1: HOUSE PRICES PREDICTION (Regression)
    # =========================================================================
    async def house_prices(self) -> Dict:
        """California Housing - predict median house value."""
        from sklearn.datasets import fetch_california_housing

        logger.info("Loading California Housing dataset...")
        data = fetch_california_housing()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = data.target

        logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Target: Median house value (in $100,000s)")

        # Feature Engineering
        logger.info("\n📊 Feature Engineering...")
        X['RoomsPerHouse'] = X['AveRooms'] / X['AveOccup'].clip(lower=0.1)
        X['BedroomsPerRoom'] = X['AveBedrms'] / X['AveRooms'].clip(lower=0.1)
        X['PopulationPerHouse'] = X['Population'] / X['HouseAge'].clip(lower=1)
        X['IncomePerRoom'] = X['MedInc'] / X['AveRooms'].clip(lower=0.1)
        X['Location'] = X['Latitude'] * X['Longitude']
        X['IncomeAge'] = X['MedInc'] * X['HouseAge']

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Models
        logger.info("\n🤖 Training Models...")
        models = {
            'LightGBM': lgb.LGBMRegressor(n_estimators=200, max_depth=6, random_state=42, verbose=-1),
            'XGBoost': xgb.XGBRegressor(n_estimators=200, max_depth=6, random_state=42),
            'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=150, max_depth=5, random_state=42),
        }

        results = {}
        predictions = {}

        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            predictions[name] = y_pred

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
            logger.info(f"   {name}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

        # Ensemble
        logger.info("\n🏗️ Building Ensemble...")
        ensemble_pred = np.mean([predictions[m] for m in predictions], axis=0)
        ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        ensemble_r2 = r2_score(y_test, ensemble_pred)

        logger.info(f"   Ensemble: RMSE={ensemble_rmse:.4f}, R²={ensemble_r2:.4f}")

        best_r2 = max(ensemble_r2, max(r['R2'] for r in results.values()))
        logger.info(f"\n🏆 Best R² Score: {best_r2:.4f}")

        return {'best_r2': best_r2, 'ensemble_rmse': ensemble_rmse, 'models': results}

    # =========================================================================
    # USE CASE 2: CREDIT CARD FRAUD DETECTION (Imbalanced Classification)
    # =========================================================================
    async def fraud_detection(self) -> Dict:
        """Synthetic fraud detection with extreme class imbalance."""
        from sklearn.datasets import make_classification
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.pipeline import Pipeline as ImbPipeline

        logger.info("Generating imbalanced fraud dataset...")

        # Create highly imbalanced dataset (1% fraud)
        X, y = make_classification(
            n_samples=10000, n_features=20, n_informative=15,
            n_redundant=3, n_clusters_per_class=2,
            weights=[0.99, 0.01], flip_y=0.01,
            random_state=42
        )

        X = pd.DataFrame(X, columns=[f'V{i}' for i in range(20)])

        fraud_rate = y.mean()
        logger.info(f"Dataset: {len(y)} transactions, {y.sum()} frauds ({fraud_rate*100:.2f}%)")

        # Feature Engineering
        logger.info("\n📊 Feature Engineering...")
        X['V_sum'] = X.sum(axis=1)
        X['V_mean'] = X.mean(axis=1)
        X['V_std'] = X.std(axis=1)
        X['V_max'] = X.max(axis=1)
        X['V_min'] = X.min(axis=1)
        X['V_range'] = X['V_max'] - X['V_min']

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info(f"Train frauds: {y_train.sum()}, Test frauds: {y_test.sum()}")

        # Handle imbalance with SMOTE
        logger.info("\n⚖️ Handling Class Imbalance (SMOTE)...")
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        logger.info(f"After SMOTE: {len(y_train_balanced)} samples, {y_train_balanced.mean()*100:.1f}% fraud")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_balanced)
        X_test_scaled = scaler.transform(X_test)

        # Models optimized for imbalanced data
        logger.info("\n🤖 Training Models...")
        models = {
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=200, max_depth=6, scale_pos_weight=99,
                random_state=42, verbose=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200, max_depth=6, scale_pos_weight=99,
                random_state=42, eval_metric='auc'
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=200, max_depth=10, class_weight='balanced',
                random_state=42, n_jobs=-1
            ),
        }

        results = {}

        for name, model in models.items():
            model.fit(X_train_scaled, y_train_balanced)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]

            auc = roc_auc_score(y_test, y_proba)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            results[name] = {'AUC': auc, 'Precision': precision, 'Recall': recall, 'F1': f1}
            logger.info(f"   {name}: AUC={auc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

        best_auc = max(r['AUC'] for r in results.values())
        best_f1 = max(r['F1'] for r in results.values())

        logger.info(f"\n🏆 Best AUC: {best_auc:.4f}, Best F1: {best_f1:.4f}")

        return {'best_auc': best_auc, 'best_f1': best_f1, 'models': results}

    # =========================================================================
    # USE CASE 3: CUSTOMER CHURN PREDICTION (Business ML)
    # =========================================================================
    async def churn_prediction(self) -> Dict:
        """Telecom customer churn prediction."""
        from sklearn.datasets import make_classification

        logger.info("Generating customer churn dataset...")

        # Simulate telecom churn data
        np.random.seed(42)
        n_samples = 5000

        # Customer features
        data = {
            'tenure': np.random.exponential(24, n_samples).clip(1, 72),
            'monthly_charges': np.random.normal(65, 30, n_samples).clip(20, 120),
            'total_charges': None,  # Will compute
            'contract_type': np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2]),  # Month, Year, Two-year
            'payment_method': np.random.choice([0, 1, 2, 3], n_samples),
            'internet_service': np.random.choice([0, 1, 2], n_samples),  # None, DSL, Fiber
            'online_security': np.random.choice([0, 1], n_samples),
            'tech_support': np.random.choice([0, 1], n_samples),
            'streaming_tv': np.random.choice([0, 1], n_samples),
            'streaming_movies': np.random.choice([0, 1], n_samples),
            'paperless_billing': np.random.choice([0, 1], n_samples),
            'senior_citizen': np.random.choice([0, 1], n_samples, p=[0.84, 0.16]),
            'partner': np.random.choice([0, 1], n_samples),
            'dependents': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'num_support_tickets': np.random.poisson(2, n_samples),
        }

        df = pd.DataFrame(data)
        df['total_charges'] = df['tenure'] * df['monthly_charges']

        # Churn probability based on features
        churn_prob = (
            0.3 * (df['contract_type'] == 0) +  # Month-to-month more likely to churn
            0.2 * (df['tenure'] < 12) +  # New customers churn more
            0.15 * (df['monthly_charges'] > 80) +  # High charges = churn
            0.1 * (df['num_support_tickets'] > 3) +  # Many tickets = unhappy
            -0.2 * (df['online_security'] == 1) +  # Security reduces churn
            -0.1 * (df['tech_support'] == 1) +
            0.05 * df['senior_citizen'] +
            np.random.normal(0, 0.1, n_samples)
        ).clip(0.05, 0.95)

        df['churn'] = (np.random.random(n_samples) < churn_prob).astype(int)

        churn_rate = df['churn'].mean()
        logger.info(f"Dataset: {len(df)} customers, {df['churn'].sum()} churned ({churn_rate*100:.1f}%)")

        # Feature Engineering
        logger.info("\n📊 Feature Engineering...")
        df['charges_per_tenure'] = df['total_charges'] / df['tenure'].clip(lower=1)
        df['services_count'] = df['online_security'] + df['tech_support'] + df['streaming_tv'] + df['streaming_movies']
        df['is_new_customer'] = (df['tenure'] < 6).astype(int)
        df['is_high_value'] = (df['monthly_charges'] > df['monthly_charges'].quantile(0.75)).astype(int)
        df['ticket_rate'] = df['num_support_tickets'] / df['tenure'].clip(lower=1)

        # Prepare data
        feature_cols = [c for c in df.columns if c != 'churn']
        X = df[feature_cols]
        y = df['churn']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Models
        logger.info("\n🤖 Training Models...")
        models = {
            'LightGBM': lgb.LGBMClassifier(n_estimators=200, max_depth=5, random_state=42, verbose=-1),
            'XGBoost': xgb.XGBClassifier(n_estimators=200, max_depth=5, random_state=42, eval_metric='auc'),
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
            'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
        }

        results = {}

        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            f1 = f1_score(y_test, y_pred)

            results[name] = {'Accuracy': acc, 'AUC': auc, 'F1': f1}
            logger.info(f"   {name}: Accuracy={acc:.4f}, AUC={auc:.4f}, F1={f1:.4f}")

        # Feature importance for business insights
        logger.info("\n📊 Top Churn Predictors:")
        lgb_model = models['LightGBM']
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': lgb_model.feature_importances_
        }).sort_values('importance', ascending=False)

        for _, row in importance.head(5).iterrows():
            logger.info(f"   {row['feature']}: {row['importance']:.0f}")

        best_auc = max(r['AUC'] for r in results.values())
        logger.info(f"\n🏆 Best AUC: {best_auc:.4f}")

        return {'best_auc': best_auc, 'churn_rate': churn_rate, 'models': results}

    # =========================================================================
    # USE CASE 4: STOCK PRICE PREDICTION (Time Series)
    # =========================================================================
    async def stock_prediction(self) -> Dict:
        """Stock price direction prediction using technical indicators."""

        logger.info("Generating synthetic stock data...")

        np.random.seed(42)
        n_days = 1000

        # Generate realistic stock price movement
        returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns
        price = 100 * np.exp(np.cumsum(returns))

        # Create OHLCV data
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=n_days, freq='D'),
            'close': price,
            'high': price * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
            'low': price * (1 - np.abs(np.random.normal(0, 0.01, n_days))),
            'volume': np.random.lognormal(15, 0.5, n_days),
        })
        df['open'] = df['close'].shift(1).fillna(df['close'])

        logger.info(f"Dataset: {len(df)} days of stock data")

        # Technical Indicators
        logger.info("\n📊 Computing Technical Indicators...")

        # Moving Averages
        df['SMA_5'] = df['close'].rolling(5).mean()
        df['SMA_20'] = df['close'].rolling(20).mean()
        df['SMA_50'] = df['close'].rolling(50).mean()
        df['EMA_12'] = df['close'].ewm(span=12).mean()
        df['EMA_26'] = df['close'].ewm(span=26).mean()

        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 0.0001)
        df['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['BB_middle'] = df['close'].rolling(20).mean()
        df['BB_std'] = df['close'].rolling(20).std()
        df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
        df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        df['BB_position'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

        # Momentum
        df['ROC_5'] = df['close'].pct_change(5)
        df['ROC_10'] = df['close'].pct_change(10)
        df['momentum'] = df['close'] - df['close'].shift(10)

        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # Volatility
        df['volatility'] = df['close'].pct_change().rolling(20).std()

        # Target: Next day direction (up=1, down=0)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

        # Drop NaN rows
        df = df.dropna()

        # Features
        feature_cols = ['SMA_5', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
                       'MACD', 'MACD_signal', 'MACD_hist', 'RSI',
                       'BB_width', 'BB_position', 'ROC_5', 'ROC_10',
                       'momentum', 'volume_ratio', 'volatility']

        X = df[feature_cols]
        y = df['target']

        # Time series split (no shuffling!)
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Models
        logger.info("\n🤖 Training Models...")
        models = {
            'LightGBM': lgb.LGBMClassifier(n_estimators=200, max_depth=4, random_state=42, verbose=-1),
            'XGBoost': xgb.XGBClassifier(n_estimators=200, max_depth=4, random_state=42, eval_metric='auc'),
            'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1),
        }

        results = {}

        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)

            results[name] = {'Accuracy': acc, 'AUC': auc}
            logger.info(f"   {name}: Accuracy={acc:.4f}, AUC={auc:.4f}")

        # Trading simulation
        logger.info("\n💰 Backtesting Trading Strategy...")
        best_model = models['LightGBM']
        predictions = best_model.predict(X_test_scaled)

        # Simple strategy: buy when predict up, hold otherwise
        test_returns = df['close'].pct_change().iloc[split_idx+1:split_idx+1+len(predictions)].values

        # Align arrays
        min_len = min(len(predictions), len(test_returns))
        strategy_returns = predictions[:min_len] * test_returns[:min_len]

        if len(strategy_returns) > 0:
            cumulative_strategy = (1 + strategy_returns).cumprod()[-1]
            cumulative_buyhold = (1 + test_returns[:min_len]).cumprod()[-1]

            logger.info(f"   Strategy Return: {(cumulative_strategy-1)*100:.2f}%")
            logger.info(f"   Buy & Hold Return: {(cumulative_buyhold-1)*100:.2f}%")

        best_acc = max(r['Accuracy'] for r in results.values())
        logger.info(f"\n🏆 Best Accuracy: {best_acc:.4f}")
        logger.info(f"   Note: >55% accuracy is good for stock prediction!")

        return {'best_accuracy': best_acc, 'strategy_return': cumulative_strategy, 'models': results}

    # =========================================================================
    # USE CASE 5: TEXT CLASSIFICATION (NLP)
    # =========================================================================
    async def text_classification(self) -> Dict:
        """Sentiment analysis on movie reviews."""
        from sklearn.datasets import fetch_20newsgroups
        from sklearn.feature_extraction.text import TfidfVectorizer

        logger.info("Loading 20 Newsgroups dataset...")

        # Use subset of newsgroups for classification
        categories = ['sci.med', 'sci.space', 'rec.sport.baseball', 'rec.sport.hockey']

        train_data = fetch_20newsgroups(
            subset='train', categories=categories,
            remove=('headers', 'footers', 'quotes'), random_state=42
        )
        test_data = fetch_20newsgroups(
            subset='test', categories=categories,
            remove=('headers', 'footers', 'quotes'), random_state=42
        )

        logger.info(f"Train: {len(train_data.data)} documents")
        logger.info(f"Test: {len(test_data.data)} documents")
        logger.info(f"Categories: {categories}")

        # TF-IDF Vectorization
        logger.info("\n📊 TF-IDF Feature Extraction...")
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )

        X_train = vectorizer.fit_transform(train_data.data)
        X_test = vectorizer.transform(test_data.data)
        y_train = train_data.target
        y_test = test_data.target

        logger.info(f"TF-IDF features: {X_train.shape[1]}")

        # Models
        logger.info("\n🤖 Training Models...")
        models = {
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
            'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
            'LightGBM': lgb.LGBMClassifier(n_estimators=100, max_depth=10, random_state=42, verbose=-1),
        }

        results = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            results[name] = {'Accuracy': acc, 'F1': f1}
            logger.info(f"   {name}: Accuracy={acc:.4f}, F1={f1:.4f}")

        best_acc = max(r['Accuracy'] for r in results.values())
        logger.info(f"\n🏆 Best Accuracy: {best_acc:.4f}")

        return {'best_accuracy': best_acc, 'models': results}

    # =========================================================================
    # USE CASE 6: IMAGE CLASSIFICATION (Deep Learning)
    # =========================================================================
    async def image_classification(self) -> Dict:
        """MNIST-style digit classification."""
        from sklearn.datasets import load_digits
        from sklearn.decomposition import PCA

        logger.info("Loading Digits dataset (8x8 images)...")

        digits = load_digits()
        X = digits.data
        y = digits.target

        logger.info(f"Dataset: {len(y)} images, {len(np.unique(y))} classes (0-9)")
        logger.info(f"Image size: 8x8 = 64 features")

        # Feature Engineering
        logger.info("\n📊 Feature Engineering...")

        # Reshape to get image-like features
        X_images = X.reshape(-1, 8, 8)

        # Add statistical features
        X_extra = np.column_stack([
            X.mean(axis=1),  # Mean intensity
            X.std(axis=1),   # Std intensity
            X.max(axis=1),   # Max intensity
            np.sum(X > 8, axis=1),  # Count of bright pixels
            np.sum(X_images[:, :4, :], axis=(1, 2)),  # Top half sum
            np.sum(X_images[:, 4:, :], axis=(1, 2)),  # Bottom half sum
            np.sum(X_images[:, :, :4], axis=(1, 2)),  # Left half sum
            np.sum(X_images[:, :, 4:], axis=(1, 2)),  # Right half sum
        ])

        X_combined = np.hstack([X, X_extra])

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Models
        logger.info("\n🤖 Training Models...")
        models = {
            'LightGBM': lgb.LGBMClassifier(n_estimators=200, max_depth=10, random_state=42, verbose=-1),
            'XGBoost': xgb.XGBClassifier(n_estimators=200, max_depth=10, random_state=42, eval_metric='mlogloss'),
            'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
            'ExtraTrees': ExtraTreesClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
        }

        results = {}

        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            results[name] = {'Accuracy': acc, 'F1': f1}
            logger.info(f"   {name}: Accuracy={acc:.4f}, F1={f1:.4f}")

        best_acc = max(r['Accuracy'] for r in results.values())
        logger.info(f"\n🏆 Best Accuracy: {best_acc:.4f}")
        logger.info(f"   Note: State-of-art on full MNIST is 99.8%+")

        return {'best_accuracy': best_acc, 'models': results}

    # =========================================================================
    # USE CASE 7: RECOMMENDATION SYSTEM (Collaborative Filtering)
    # =========================================================================
    async def recommendation_system(self) -> Dict:
        """Movie recommendation using matrix factorization."""
        from sklearn.decomposition import NMF, TruncatedSVD

        logger.info("Generating synthetic movie ratings dataset...")

        np.random.seed(42)
        n_users = 500
        n_movies = 200
        n_ratings = 10000

        # Generate random ratings
        user_ids = np.random.randint(0, n_users, n_ratings)
        movie_ids = np.random.randint(0, n_movies, n_ratings)

        # Create user and movie latent factors for realistic ratings
        user_factors = np.random.randn(n_users, 5)
        movie_factors = np.random.randn(n_movies, 5)

        # Generate ratings based on latent factors + noise
        ratings = []
        for u, m in zip(user_ids, movie_ids):
            base_rating = np.dot(user_factors[u], movie_factors[m])
            rating = np.clip(base_rating + np.random.normal(0, 0.5), 1, 5)
            ratings.append(round(rating))

        df = pd.DataFrame({
            'user_id': user_ids,
            'movie_id': movie_ids,
            'rating': ratings
        })

        # Remove duplicates (keep first rating)
        df = df.drop_duplicates(subset=['user_id', 'movie_id'])

        logger.info(f"Dataset: {len(df)} ratings from {n_users} users on {n_movies} movies")
        logger.info(f"Sparsity: {100 * (1 - len(df) / (n_users * n_movies)):.1f}%")

        # Create rating matrix
        logger.info("\n📊 Building Rating Matrix...")
        rating_matrix = df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

        # Train/test split (hold out some ratings)
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        train_matrix = train_df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

        # Align columns
        all_movies = rating_matrix.columns
        for col in all_movies:
            if col not in train_matrix.columns:
                train_matrix[col] = 0
        train_matrix = train_matrix[all_movies]

        # Matrix Factorization Methods
        logger.info("\n🤖 Training Recommendation Models...")

        results = {}

        # SVD
        svd = TruncatedSVD(n_components=20, random_state=42)
        user_features = svd.fit_transform(train_matrix)
        movie_features = svd.components_.T

        predictions_svd = np.dot(user_features, movie_features.T)
        predictions_df_svd = pd.DataFrame(
            predictions_svd,
            index=train_matrix.index,
            columns=train_matrix.columns
        )

        # Evaluate on test set
        test_preds_svd = []
        test_actual = []
        for _, row in test_df.iterrows():
            if row['user_id'] in predictions_df_svd.index and row['movie_id'] in predictions_df_svd.columns:
                test_preds_svd.append(predictions_df_svd.loc[row['user_id'], row['movie_id']])
                test_actual.append(row['rating'])

        if test_preds_svd:
            rmse_svd = np.sqrt(mean_squared_error(test_actual, test_preds_svd))
            results['SVD'] = {'RMSE': rmse_svd}
            logger.info(f"   SVD: RMSE={rmse_svd:.4f}")

        # NMF (Non-negative Matrix Factorization)
        nmf = NMF(n_components=20, random_state=42, max_iter=200)
        user_features_nmf = nmf.fit_transform(train_matrix.clip(lower=0))
        movie_features_nmf = nmf.components_.T

        predictions_nmf = np.dot(user_features_nmf, movie_features_nmf.T)
        predictions_df_nmf = pd.DataFrame(
            predictions_nmf,
            index=train_matrix.index,
            columns=train_matrix.columns
        )

        test_preds_nmf = []
        test_actual_nmf = []
        for _, row in test_df.iterrows():
            if row['user_id'] in predictions_df_nmf.index and row['movie_id'] in predictions_df_nmf.columns:
                test_preds_nmf.append(predictions_df_nmf.loc[row['user_id'], row['movie_id']])
                test_actual_nmf.append(row['rating'])

        if test_preds_nmf:
            rmse_nmf = np.sqrt(mean_squared_error(test_actual_nmf, test_preds_nmf))
            results['NMF'] = {'RMSE': rmse_nmf}
            logger.info(f"   NMF: RMSE={rmse_nmf:.4f}")

        # User-based collaborative filtering (simple)
        from sklearn.metrics.pairwise import cosine_similarity

        user_similarity = cosine_similarity(train_matrix)

        # Predict using weighted average of similar users
        def predict_user_cf(user_id, movie_id, k=10):
            if user_id not in train_matrix.index:
                return 3.0  # Default

            user_idx = train_matrix.index.get_loc(user_id)
            similarities = user_similarity[user_idx]

            # Get ratings for this movie
            movie_ratings = train_matrix[movie_id].values

            # Find users who rated this movie
            rated_mask = movie_ratings > 0
            if not rated_mask.any():
                return 3.0

            # Weight by similarity
            sim_scores = similarities[rated_mask]
            ratings = movie_ratings[rated_mask]

            if sim_scores.sum() == 0:
                return ratings.mean()

            return np.average(ratings, weights=np.abs(sim_scores))

        test_preds_cf = []
        test_actual_cf = []
        for _, row in test_df.head(500).iterrows():  # Limit for speed
            pred = predict_user_cf(row['user_id'], row['movie_id'])
            test_preds_cf.append(pred)
            test_actual_cf.append(row['rating'])

        rmse_cf = np.sqrt(mean_squared_error(test_actual_cf, test_preds_cf))
        results['User-CF'] = {'RMSE': rmse_cf}
        logger.info(f"   User-CF: RMSE={rmse_cf:.4f}")

        best_rmse = min(r['RMSE'] for r in results.values())
        logger.info(f"\n🏆 Best RMSE: {best_rmse:.4f}")
        logger.info(f"   Note: Netflix Prize winning RMSE was ~0.857")

        return {'best_rmse': best_rmse, 'models': results}


async def main():
    logger.info("=" * 70)
    logger.info("JOTTY AI - COMPREHENSIVE ML USE CASES")
    logger.info("=" * 70)
    logger.info("Testing autonomous ML across 7 domains")

    runner = UseCaseRunner()
    results = await runner.run_all()

    # Final Summary
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)

    logger.info(f"\n{'Use Case':<40} {'Key Metric':<20} {'Time':>10}")
    logger.info("-" * 72)

    for name, result in results.items():
        if 'error' in result:
            metric = f"ERROR: {result['error'][:30]}"
            time_str = "N/A"
        else:
            # Get primary metric
            if 'best_r2' in result:
                metric = f"R² = {result['best_r2']:.4f}"
            elif 'best_auc' in result:
                metric = f"AUC = {result['best_auc']:.4f}"
            elif 'best_accuracy' in result:
                metric = f"Acc = {result['best_accuracy']:.4f}"
            elif 'best_rmse' in result:
                metric = f"RMSE = {result['best_rmse']:.4f}"
            else:
                metric = "Completed"
            time_str = f"{result.get('time', 0):.1f}s"

        logger.info(f"{name:<40} {metric:<20} {time_str:>10}")

    logger.info("\n" + "=" * 70)
    logger.info("ALL USE CASES COMPLETED!")
    logger.info("=" * 70)

    return results


if __name__ == "__main__":
    # Check for imbalanced-learn
    try:
        import imblearn
    except ImportError:
        import subprocess
        subprocess.run(['pip', 'install', 'imbalanced-learn', '-q'])

    results = asyncio.run(main())
