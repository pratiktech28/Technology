# utils/preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    """Advanced data preprocessing utilities"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        self.scaler = None
        self.encoders = {}
    
    def handle_missing_values(self, strategy='mean'):
        """
        Handle missing values in dataset
        strategy: 'mean', 'median', 'mode', 'drop'
        """
        if strategy == 'drop':
            self.df = self.df.dropna()
        else:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            
            # Numeric columns
            if len(numeric_cols) > 0:
                imputer = SimpleImputer(strategy=strategy)
                self.df[numeric_cols] = imputer.fit_transform(self.df[numeric_cols])
            
            # Categorical columns
            if len(categorical_cols) > 0:
                imputer = SimpleImputer(strategy='most_frequent')
                self.df[categorical_cols] = imputer.fit_transform(self.df[categorical_cols])
        
        return self
    
    def remove_duplicates(self):
        """Remove duplicate rows"""
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        removed = before - len(self.df)
        print(f"Removed {removed} duplicate rows")
        return self
    
    def normalize_features(self, method='standard'):
        """
        Normalize numerical features
        method: 'standard' or 'minmax'
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if method == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        
        self.df[numeric_cols] = self.scaler.fit_transform(self.df[numeric_cols])
        return self
    
    def encode_categorical(self):
        """Encode categorical variables"""
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.encoders[col] = le
        
        return self
    
    def remove_outliers(self, threshold=3):
        """Remove outliers using Z-score method"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
            self.df = self.df[z_scores < threshold]
        
        return self
    
    def feature_engineering(self):
        """Create new features"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 2:
            # Interaction features
            self.df['feature_sum'] = self.df[numeric_cols].sum(axis=1)
            self.df['feature_mean'] = self.df[numeric_cols].mean(axis=1)
            self.df['feature_std'] = self.df[numeric_cols].std(axis=1)
            
            # Polynomial features (limited to avoid explosion)
            for i, col1 in enumerate(numeric_cols[:3]):
                for col2 in numeric_cols[i+1:4]:
                    self.df[f'{col1}_x_{col2}'] = self.df[col1] * self.df[col2]
        
        return self
    
    def get_processed_data(self):
        """Return processed dataframe"""
        return self.df
    
    def get_report(self):
        """Generate preprocessing report"""
        return {
            'original_shape': self.original_shape,
            'final_shape': self.df.shape,
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'columns_added': self.df.shape[1] - self.original_shape[1],
            'missing_values': self.df.isnull().sum().sum(),
            'data_types': self.df.dtypes.value_counts().to_dict()
        }


# =============================================
# utils/model_training.py
# =============================================

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

class ModelTrainer:
    """Train and evaluate ML models"""
    
    def __init__(self, X, y, test_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        self.models = {}
        self.results = {}
    
    def train_random_forest(self, n_estimators=100):
        """Train Random Forest"""
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = model
        return self._evaluate(model, 'Random Forest')
    
    def train_gradient_boosting(self, n_estimators=100):
        """Train Gradient Boosting"""
        model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(self.X_train, self.y_train)
        self.models['Gradient Boosting'] = model
        return self._evaluate(model, 'Gradient Boosting')
    
    def train_logistic_regression(self):
        """Train Logistic Regression"""
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(self.X_train, self.y_train)
        self.models['Logistic Regression'] = model
        return self._evaluate(model, 'Logistic Regression')
    
    def train_svm(self):
        """Train Support Vector Machine"""
        model = SVC(kernel='rbf', random_state=42)
        model.fit(self.X_train, self.y_train)
        self.models['SVM'] = model
        return self._evaluate(model, 'SVM')
    
    def _evaluate(self, model, name):
        """Evaluate model performance"""
        y_pred = model.predict(self.X_test)
        
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, average='weighted'),
            'recall': recall_score(self.y_test, y_pred, average='weighted'),
            'f1_score': f1_score(self.y_test, y_pred, average='weighted')
        }
        
        # Cross-validation
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        self.results[name] = metrics
        return metrics
    
    def train_all_models(self):
        """Train all available models"""
        self.train_random_forest()
        self.train_gradient_boosting()
        self.train_logistic_regression()
        # self.train_svm()  # Commented out as it can be slow
        return self.results
    
    def get_best_model(self):
        """Get the best performing model"""
        best_model_name = max(self.results, key=lambda x: self.results[x]['accuracy'])
        return best_model_name, self.models[best_model_name], self.results[best_model_name]
    
    def save_model(self, model_name, filepath):
        """Save trained model"""
        if model_name in self.models:
            joblib.dump(self.models[model_name], filepath)
            print(f"Model saved to {filepath}")
        else:
            print(f"Model {model_name} not found")
    
    def get_feature_importance(self, model_name):
        """Get feature importance for tree-based models"""
        if model_name in self.models:
            model = self.models[model_name]
            if hasattr(model, 'feature_importances_'):
                return model.feature_importances_
        return None


# =============================================
# utils/visualization.py
# =============================================

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class DataVisualizer:
    """Create beautiful visualizations"""
    
    def __init__(self, df):
        self.df = df
    
    def plot_distribution(self, column):
        """Plot distribution of a column"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Histogram', 'Box Plot')
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=self.df[column], name='Distribution'),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(y=self.df[column], name='Box Plot'),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text=f'Distribution Analysis: {column}',
            showlegend=False
        )
        
        return fig
    
    def plot_correlation_heatmap(self):
        """Plot correlation heatmap"""
        numeric_df = self.df.select_dtypes(include=['number'])
        corr = numeric_df.corr()
        
        fig = px.imshow(
            corr,
            text_auto=True,
            aspect='auto',
            color_continuous_scale='RdBu_r',
            title='Feature Correlation Heatmap'
        )
        
        return fig
    
    def plot_feature_importance(self, features, importance):
        """Plot feature importance"""
        df_importance = pd.DataFrame({
            'Feature': features,
            'Importance': importance
        }).sort_values('Importance', ascending=False).head(10)
        
        fig = px.bar(
            df_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Top 10 Feature Importance',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        
        return fig
    
    def plot_model_comparison(self, results):
        """Compare multiple model performances"""
        models = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig = go.Figure()
        
        for metric in metrics:
            values = [results[model][metric] for model in models]
            fig.add_trace(go.Bar(
                name=metric.title(),
                x=models,
                y=values
            ))
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Models',
            yaxis_title='Score',
            barmode='group'
        )
        
        return fig
    
    def plot_confusion_matrix(self, cm, labels):
        """Plot confusion matrix"""
        fig = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x='Predicted', y='Actual'),
            x=labels,
            y=labels,
            color_continuous_scale='Blues',
            title='Confusion Matrix'
        )
        
        return fig
    
    def plot_data_overview(self):
        """Create comprehensive data overview"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Missing Values',
                'Data Types',
                'Numeric Features Distribution',
                'Feature Correlations'
            )
        )
        
        # Missing values
        missing = self.df.isnull().sum().sort_values(ascending=False).head(10)
        fig.add_trace(
            go.Bar(x=missing.index, y=missing.values, name='Missing'),
            row=1, col=1
        )
        
        # Data types
        dtypes = self.df.dtypes.value_counts()
        fig.add_trace(
            go.Pie(labels=dtypes.index.astype(str), values=dtypes.values, name='Types'),
            row=1, col=2
        )
        
        fig.update_layout(height=800, title_text='Data Overview Dashboard')
        
        return fig


# =============================================
# Example Usage
# =============================================

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('your_data.csv')
    
    # Preprocessing
    preprocessor = DataPreprocessor(df)
    processed_df = (preprocessor
                   .handle_missing_values()
                   .remove_duplicates()
                   .normalize_features()
                   .encode_categorical()
                   .feature_engineering()
                   .get_processed_data())
    
    print("Preprocessing Report:")
    print(preprocessor.get_report())
    
    # Model Training
    X = processed_df.drop('target', axis=1)
    y = processed_df['target']
    
    trainer = ModelTrainer(X, y)
    results = trainer.train_all_models()
    
    print("\nModel Results:")
    for model, metrics in results.items():
        print(f"{model}: {metrics['accuracy']:.4f}")
    
    # Get best model
    best_name, best_model, best_metrics = trainer.get_best_model()
    print(f"\nBest Model: {best_name}")
    print(f"Accuracy: {best_metrics['accuracy']:.4f}")
    
    # Save model
    trainer.save_model(best_name, 'models/best_model.pkl')
    
    # Visualization
    visualizer = DataVisualizer(processed_df)
    fig = visualizer.plot_correlation_heatmap()
    fig.show()
