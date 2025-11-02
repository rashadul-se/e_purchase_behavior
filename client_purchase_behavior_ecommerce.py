"""
Customer Behavioral Analysis & Prediction System
Uses SQLite, Apache Spark, and Streamlit for ML-based customer analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# PySpark setup (optional - falls back to pandas if not available)
try:
    from pyspark.sql import SparkSession
    from pyspark.ml.feature import VectorAssembler, StringIndexer
    from pyspark.ml.regression import RandomForestRegressor as SparkRFRegressor
    from pyspark.ml.classification import RandomForestClassifier as SparkRFClassifier
    from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False


class CustomerAnalyticsPipeline:
    """Complete ML pipeline for customer behavior prediction"""
    
    def __init__(self, db_path='customer_data.db', use_spark=False):
        self.db_path = db_path
        self.use_spark = use_spark and SPARK_AVAILABLE
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        self.spark = None
        
        if self.use_spark:
            self.spark = SparkSession.builder \
                .appName("CustomerAnalytics") \
                .config("spark.driver.memory", "2g") \
                .getOrCreate()
            self.spark.sparkContext.setLogLevel("ERROR")
        
    def setup_database(self):
        """Initialize SQLite database with schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS customers (
                customer_id TEXT PRIMARY KEY,
                segment TEXT,
                registration_date TEXT,
                days_since_registration INTEGER,
                age INTEGER,
                gender TEXT,
                city TEXT,
                purchase_frequency INTEGER,
                avg_purchase_value REAL,
                total_spent REAL,
                last_purchase_days INTEGER,
                favorite_category TEXT,
                preferred_channel TEXT,
                device_type TEXT,
                session_count INTEGER,
                avg_session_duration INTEGER,
                cart_abandonment_rate REAL,
                email_open_rate REAL,
                click_through_rate REAL,
                customer_lifetime_value REAL,
                churn_probability REAL,
                nps_score INTEGER,
                support_tickets INTEGER,
                returns_count INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def load_data_to_db(self, df):
        """Load DataFrame to SQLite database"""
        conn = sqlite3.connect(self.db_path)
        df.to_sql('customers', conn, if_exists='replace', index=False)
        conn.close()
        
    def load_data_from_db(self):
        """Load data from SQLite database"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM customers", conn)
        conn.close()
        return df
    
    def load_github_dataset(self, url):
        """Load dataset from GitHub URL"""
        try:
            df = pd.read_csv(url)
            return df
        except Exception as e:
            raise Exception(f"Failed to load dataset from GitHub: {str(e)}")
    
    def preprocess_data(self, df):
        """Prepare data for machine learning"""
        data = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['gender', 'city', 'favorite_category', 'preferred_channel', 'device_type']
        
        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                data[f'{col}_encoded'] = self.encoders[col].fit_transform(data[col].astype(str))
            else:
                # Handle unseen categories
                data[f'{col}_encoded'] = data[col].apply(
                    lambda x: self.encoders[col].transform([str(x)])[0] 
                    if str(x) in self.encoders[col].classes_ 
                    else -1
                )
        
        # Feature engineering
        data['recency_score'] = 1 / (data['last_purchase_days'] + 1)
        data['frequency_score'] = np.log1p(data['purchase_frequency'])
        data['monetary_score'] = np.log1p(data['total_spent'])
        data['engagement_score'] = (data['session_count'] * data['avg_session_duration']) / 1000
        data['email_engagement'] = data['email_open_rate'] * data['click_through_rate']
        
        return data
    
    def train_models_sklearn(self, df):
        """Train models using Scikit-learn"""
        st.info("🔄 Training ML models using Scikit-learn... This may take a moment.")
        
        data = self.preprocess_data(df)
        
        # Define feature columns
        self.feature_columns = [
            'days_since_registration', 'age', 'purchase_frequency', 
            'avg_purchase_value', 'total_spent', 'last_purchase_days',
            'session_count', 'avg_session_duration', 'cart_abandonment_rate',
            'email_open_rate', 'click_through_rate', 'nps_score',
            'support_tickets', 'returns_count', 'gender_encoded', 
            'city_encoded', 'favorite_category_encoded', 'preferred_channel_encoded',
            'device_type_encoded', 'recency_score', 'frequency_score',
            'monetary_score', 'engagement_score', 'email_engagement'
        ]
        
        X = data[self.feature_columns]
        
        # Scale features
        self.scalers['features'] = StandardScaler()
        X_scaled = self.scalers['features'].fit_transform(X)
        
        metrics = {}
        
        # Train Churn Prediction Model
        y_churn = (data['churn_probability'] > 0.5).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_churn, test_size=0.2, random_state=42)
        
        self.models['churn'] = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        self.models['churn'].fit(X_train, y_train)
        metrics['churn_accuracy'] = accuracy_score(y_test, self.models['churn'].predict(X_test))
        
        # Train CLV Prediction Model
        y_clv = data['customer_lifetime_value']
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_clv, test_size=0.2, random_state=42)
        
        self.models['clv'] = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        self.models['clv'].fit(X_train, y_train)
        metrics['clv_r2'] = r2_score(y_test, self.models['clv'].predict(X_test))
        metrics['clv_mae'] = mean_absolute_error(y_test, self.models['clv'].predict(X_test))
        
        # Train Purchase Frequency Model
        y_freq = data['purchase_frequency']
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_freq, test_size=0.2, random_state=42)
        
        self.models['frequency'] = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        self.models['frequency'].fit(X_train, y_train)
        metrics['frequency_mae'] = mean_absolute_error(y_test, self.models['frequency'].predict(X_test))
        metrics['frequency_r2'] = r2_score(y_test, self.models['frequency'].predict(X_test))
        
        # Train Segment Classification Model
        segment_encoder = LabelEncoder()
        y_segment = segment_encoder.fit_transform(data['segment'])
        self.encoders['segment'] = segment_encoder
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_segment, test_size=0.2, random_state=42)
        
        self.models['segment'] = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        self.models['segment'].fit(X_train, y_train)
        metrics['segment_accuracy'] = accuracy_score(y_test, self.models['segment'].predict(X_test))
        
        return metrics
    
    def train_models_spark(self, df):
        """Train models using Apache Spark MLlib"""
        st.info("🔄 Training ML models using Apache Spark... This may take a moment.")
        
        data = self.preprocess_data(df)
        
        # Convert to Spark DataFrame
        spark_df = self.spark.createDataFrame(data)
        
        # Define feature columns
        self.feature_columns = [
            'days_since_registration', 'age', 'purchase_frequency', 
            'avg_purchase_value', 'total_spent', 'last_purchase_days',
            'session_count', 'avg_session_duration', 'cart_abandonment_rate',
            'email_open_rate', 'click_through_rate', 'nps_score',
            'support_tickets', 'returns_count', 'gender_encoded', 
            'city_encoded', 'favorite_category_encoded', 'preferred_channel_encoded',
            'device_type_encoded', 'recency_score', 'frequency_score',
            'monetary_score', 'engagement_score', 'email_engagement'
        ]
        
        # Prepare features
        assembler = VectorAssembler(inputCols=self.feature_columns, outputCol="features")
        spark_df = assembler.transform(spark_df)
        
        # Add churn label
        spark_df = spark_df.withColumn("churn_label", 
                                       (spark_df.churn_probability > 0.5).cast("integer"))
        
        # Split data
        train_df, test_df = spark_df.randomSplit([0.8, 0.2], seed=42)
        
        metrics = {}
        
        # Train CLV Model (Spark)
        rf_clv = SparkRFRegressor(featuresCol="features", labelCol="customer_lifetime_value", 
                                   numTrees=100, maxDepth=10, seed=42)
        self.models['clv_spark'] = rf_clv.fit(train_df)
        
        predictions_clv = self.models['clv_spark'].transform(test_df)
        evaluator_r2 = RegressionEvaluator(labelCol="customer_lifetime_value", 
                                           predictionCol="prediction", metricName="r2")
        evaluator_mae = RegressionEvaluator(labelCol="customer_lifetime_value", 
                                            predictionCol="prediction", metricName="mae")
        metrics['clv_r2'] = evaluator_r2.evaluate(predictions_clv)
        metrics['clv_mae'] = evaluator_mae.evaluate(predictions_clv)
        
        # Train Churn Model (Spark)
        rf_churn = SparkRFClassifier(featuresCol="features", labelCol="churn_label", 
                                      numTrees=100, maxDepth=10, seed=42)
        self.models['churn_spark'] = rf_churn.fit(train_df)
        
        predictions_churn = self.models['churn_spark'].transform(test_df)
        evaluator_acc = MulticlassClassificationEvaluator(labelCol="churn_label", 
                                                          predictionCol="prediction", 
                                                          metricName="accuracy")
        metrics['churn_accuracy'] = evaluator_acc.evaluate(predictions_churn)
        
        # Train Frequency Model (Spark)
        rf_freq = SparkRFRegressor(featuresCol="features", labelCol="purchase_frequency", 
                                    numTrees=100, maxDepth=10, seed=42)
        self.models['frequency_spark'] = rf_freq.fit(train_df)
        
        predictions_freq = self.models['frequency_spark'].transform(test_df)
        evaluator_r2 = RegressionEvaluator(labelCol="purchase_frequency", 
                                           predictionCol="prediction", metricName="r2")
        evaluator_mae = RegressionEvaluator(labelCol="purchase_frequency", 
                                            predictionCol="prediction", metricName="mae")
        metrics['frequency_r2'] = evaluator_r2.evaluate(predictions_freq)
        metrics['frequency_mae'] = evaluator_mae.evaluate(predictions_freq)
        
        # For segment, we'll use sklearn as Spark needs specific setup
        data_pd = data.copy()
        X = data_pd[self.feature_columns]
        self.scalers['features'] = StandardScaler()
        X_scaled = self.scalers['features'].fit_transform(X)
        
        segment_encoder = LabelEncoder()
        y_segment = segment_encoder.fit_transform(data_pd['segment'])
        self.encoders['segment'] = segment_encoder
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_segment, 
                                                              test_size=0.2, random_state=42)
        
        self.models['segment'] = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        self.models['segment'].fit(X_train, y_train)
        metrics['segment_accuracy'] = accuracy_score(y_test, self.models['segment'].predict(X_test))
        
        return metrics
    
    def train_models(self, df):
        """Train models using selected framework"""
        if self.use_spark:
            return self.train_models_spark(df)
        else:
            return self.train_models_sklearn(df)
    
    def predict_metrics(self, input_data):
        """Predict all metrics for given customer data"""
        data = self.preprocess_data(input_data)
        X = data[self.feature_columns]
        
        predictions = {}
        
        if self.use_spark and hasattr(self.models.get('clv_spark'), 'transform'):
            # Spark predictions
            spark_input = self.spark.createDataFrame(data)
            assembler = VectorAssembler(inputCols=self.feature_columns, outputCol="features")
            spark_input = assembler.transform(spark_input)
            
            # CLV
            clv_pred = self.models['clv_spark'].transform(spark_input)
            predictions['predicted_clv'] = clv_pred.select("prediction").toPandas()['prediction'].values
            
            # Churn
            churn_pred = self.models['churn_spark'].transform(spark_input)
            predictions['churn_probability'] = churn_pred.select("probability").toPandas()['probability'].apply(lambda x: x[1]).values
            predictions['will_churn'] = churn_pred.select("prediction").toPandas()['prediction'].values.astype(int)
            
            # Frequency
            freq_pred = self.models['frequency_spark'].transform(spark_input)
            predictions['predicted_frequency'] = freq_pred.select("prediction").toPandas()['prediction'].values
            
            # Segment (sklearn)
            X_scaled = self.scalers['features'].transform(X)
            segment_pred = self.models['segment'].predict(X_scaled)
            predictions['predicted_segment'] = self.encoders['segment'].inverse_transform(segment_pred)
        else:
            # Sklearn predictions
            X_scaled = self.scalers['features'].transform(X)
            
            predictions['churn_probability'] = self.models['churn'].predict_proba(X_scaled)[:, 1]
            predictions['will_churn'] = self.models['churn'].predict(X_scaled)
            predictions['predicted_clv'] = self.models['clv'].predict(X_scaled)
            predictions['predicted_frequency'] = self.models['frequency'].predict(X_scaled)
            
            segment_pred = self.models['segment'].predict(X_scaled)
            predictions['predicted_segment'] = self.encoders['segment'].inverse_transform(segment_pred)
        
        # Calculate additional KPIs
        predictions['customer_health_score'] = (
            (1 - predictions['churn_probability']) * 0.4 +
            (data['email_engagement'].values / (data['email_engagement'].max() + 0.01)) * 0.3 +
            (data['nps_score'].values + 100) / 200 * 0.3
        ) * 100
        
        predictions['retention_priority'] = np.where(
            predictions['churn_probability'] > 0.5,
            'HIGH',
            np.where(predictions['churn_probability'] > 0.3, 'MEDIUM', 'LOW')
        )
        
        return predictions


def main():
    st.set_page_config(page_title="Customer Analytics Platform", layout="wide", page_icon="📊")
    
    st.title("🎯 Customer Behavioral Analytics & Prediction System")
    st.markdown("### Advanced ML-Powered Customer Intelligence Platform")
    
    # Initialize session state
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
        st.session_state.models_trained = False
        st.session_state.data_loaded = False
        st.session_state.df = None
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # ML Framework Selection
        st.subheader("🔧 ML Framework")
        ml_framework = st.radio(
            "Select ML Framework:",
            ["Scikit-learn", "Apache Spark"],
            help="Choose the machine learning framework for model training"
        )
        
        use_spark = (ml_framework == "Apache Spark")
        
        if use_spark and not SPARK_AVAILABLE:
            st.error("⚠️ Apache Spark not available. Please install PySpark.")
            use_spark = False
        
        st.markdown("---")
        
        # Data Source Selection
        st.subheader("📁 Data Source")
        data_source = st.radio(
            "Select Data Source:",
            ["GitHub Dataset", "Upload CSV File"]
        )
        
        st.markdown("---")
        st.info(f"""
        **Current Config:**
        - Framework: {ml_framework}
        - Data Source: {data_source}
        """)
        
        if st.button("🔄 Reset Application"):
            st.session_state.clear()
            st.rerun()
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📥 Load & Train", "📊 Explore Data", "🔮 Predict Metrics"])
    
    with tab1:
        st.header("📥 Load Dataset & Train Models")
        
        df = None
        
        if data_source == "GitHub Dataset":
            st.markdown("""
            ### 📂 GitHub Dataset
            Loading dataset from: 
            `https://raw.githubusercontent.com/rashadul-se/e_purchase_behavior/refs/heads/main/customer_behavioral_data_10000_records.csv`
            """)
            
            if st.button("🚀 Load GitHub Dataset", type="primary"):
                with st.spinner("Loading dataset from GitHub..."):
                    try:
                        github_url = "https://raw.githubusercontent.com/rashadul-se/e_purchase_behavior/refs/heads/main/customer_behavioral_data_10000_records.csv"
                        
                        # Initialize pipeline with selected framework
                        st.session_state.pipeline = CustomerAnalyticsPipeline(use_spark=use_spark)
                        st.session_state.pipeline.setup_database()
                        
                        df = st.session_state.pipeline.load_github_dataset(github_url)
                        st.session_state.df = df
                        st.session_state.data_loaded = True
                        
                        st.success(f"✅ Successfully loaded {len(df):,} records from GitHub!")
                        
                    except Exception as e:
                        st.error(f"❌ Error loading dataset: {str(e)}")
                        st.stop()
        
        else:  # Upload CSV File
            st.markdown("### 📤 Upload Your Dataset")
            uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
            
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.success(f"✅ Loaded {len(df):,} records from uploaded file")
        
        # Display dataset info if loaded
    
