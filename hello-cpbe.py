"""
Customer Behavioral Analysis & Prediction System
Uses SQLite and Streamlit for ML-based customer analytics
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
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class CustomerAnalyticsPipeline:
    """Complete ML pipeline for customer behavior prediction"""
    
    def __init__(self, db_path='customer_data.db'):
        self.db_path = db_path
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        
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
    
    def train_models(self, df):
        """Train models using Scikit-learn"""
        st.info("Training ML models... This may take a moment.")
        
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
        
        self.models['churn'] = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=8, n_jobs=-1)
        self.models['churn'].fit(X_train, y_train)
        metrics['churn_accuracy'] = accuracy_score(y_test, self.models['churn'].predict(X_test))
        
        # Train CLV Prediction Model
        y_clv = data['customer_lifetime_value']
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_clv, test_size=0.2, random_state=42)
        
        self.models['clv'] = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=8, n_jobs=-1)
        self.models['clv'].fit(X_train, y_train)
        metrics['clv_r2'] = r2_score(y_test, self.models['clv'].predict(X_test))
        metrics['clv_mae'] = mean_absolute_error(y_test, self.models['clv'].predict(X_test))
        
        # Train Purchase Frequency Model
        y_freq = data['purchase_frequency']
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_freq, test_size=0.2, random_state=42)
        
        self.models['frequency'] = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=8, n_jobs=-1)
        self.models['frequency'].fit(X_train, y_train)
        metrics['frequency_mae'] = mean_absolute_error(y_test, self.models['frequency'].predict(X_test))
        metrics['frequency_r2'] = r2_score(y_test, self.models['frequency'].predict(X_test))
        
        # Train Segment Classification Model
        segment_encoder = LabelEncoder()
        y_segment = segment_encoder.fit_transform(data['segment'])
        self.encoders['segment'] = segment_encoder
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_segment, test_size=0.2, random_state=42)
        
        self.models['segment'] = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=8, n_jobs=-1)
        self.models['segment'].fit(X_train, y_train)
        metrics['segment_accuracy'] = accuracy_score(y_test, self.models['segment'].predict(X_test))
        
        return metrics
    
    def predict_metrics(self, input_data):
        """Predict all metrics for given customer data"""
        data = self.preprocess_data(input_data)
        X = data[self.feature_columns]
        
        # Sklearn predictions
        X_scaled = self.scalers['features'].transform(X)
        
        predictions = {}
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
        
        st.subheader("📁 Data Source")
        data_source = st.radio(
            "Select Data Source:",
            ["GitHub Dataset", "Upload CSV File"]
        )
        
        st.markdown("---")
        
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
            Loading dataset from repository
            """)
            
            if st.button("🚀 Load GitHub Dataset", type="primary"):
                with st.spinner("Loading dataset from GitHub..."):
                    try:
                        github_url = "https://raw.githubusercontent.com/rashadul-se/e_purchase_behavior/refs/heads/main/customer_behavioral_data_10000_records.csv"
                        
                        # Initialize pipeline
                        st.session_state.pipeline = CustomerAnalyticsPipeline()
                        st.session_state.pipeline.setup_database()
                        
                        df = st.session_state.pipeline.load_github_dataset(github_url)
                        st.session_state.df = df
                        st.session_state.data_loaded = True
                        
                        st.success(f"✅ Successfully loaded {len(df):,} records from GitHub!")
                        
                    except Exception as e:
                        st.error(f"❌ Error loading dataset: {str(e)}")
                        st.stop()
        
        else:
            st.markdown("### 📤 Upload Your Dataset")
            uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
            
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.success(f"✅ Loaded {len(df):,} records from uploaded file")
        
        # Display dataset info if loaded
        if st.session_state.data_loaded and st.session_state.df is not None:
            df = st.session_state.df
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Customers", f"{len(df):,}")
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                st.metric("Avg CLV", f"৳{df['customer_lifetime_value'].mean():,.0f}")
            with col4:
                st.metric("Avg Churn Risk", f"{df['churn_probability'].mean()*100:.1f}%")
            
            st.subheader("📊 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.markdown("---")
            
            # Train Models Section
            st.subheader("🎓 Train ML Models")
            
            if st.button("🚀 Train Models", type="primary", use_container_width=True):
                with st.spinner("Training models..."):
                    try:
                        # Initialize or update pipeline
                        if st.session_state.pipeline is None:
                            st.session_state.pipeline = CustomerAnalyticsPipeline()
                            st.session_state.pipeline.setup_database()
                        
                        # Save to database
                        st.session_state.pipeline.load_data_to_db(df)
                        
                        # Train models
                        metrics = st.session_state.pipeline.train_models(df)
                        st.session_state.models_trained = True
                        
                        st.success("✅ Models trained successfully!")
                        
                        # Display metrics
                        st.markdown("### 📈 Model Performance Metrics")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Churn Accuracy", f"{metrics['churn_accuracy']*100:.2f}%")
                        with col2:
                            st.metric("CLV R² Score", f"{metrics['clv_r2']:.4f}")
                        with col3:
                            st.metric("CLV MAE", f"৳{metrics['clv_mae']:,.0f}")
                        with col4:
                            st.metric("Segment Accuracy", f"{metrics['segment_accuracy']*100:.2f}%")
                        
                    except Exception as e:
                        st.error(f"❌ Error training models: {str(e)}")
    
    with tab2:
        st.header("📊 Explore Customer Dataset")
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load a dataset first in the 'Load & Train' tab.")
        else:
            df = st.session_state.df
            
            # KPI Cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Customers", f"{len(df):,}")
            with col2:
                high_value = len(df[df['segment'] == 'High-Value Loyal'])
                st.metric("High-Value Customers", f"{high_value:,}")
            with col3:
                at_risk = len(df[df['churn_probability'] > 0.5])
                st.metric("At-Risk Customers", f"{at_risk:,}")
            with col4:
                st.metric("Avg Purchase Freq", f"{df['purchase_frequency'].mean():.1f}")
            
            # Visualizations
            viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📈 Segments", "💰 Revenue", "⚠️ Churn"])
            
            with viz_tab1:
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.pie(df, names='segment', title='Customer Segment Distribution')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    segment_clv = df.groupby('segment')['customer_lifetime_value'].mean().reset_index()
                    fig = px.bar(segment_clv, x='segment', y='customer_lifetime_value', 
                                title='Average CLV by Segment')
                    st.plotly_chart(fig, use_container_width=True)
            
            with viz_tab2:
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.histogram(df, x='total_spent', nbins=50, 
                                      title='Total Spend Distribution')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.scatter(df, x='purchase_frequency', y='customer_lifetime_value',
                                    color='segment', title='Purchase Frequency vs CLV')
                    st.plotly_chart(fig, use_container_width=True)
            
            with viz_tab3:
                col1, col2 = st.columns(2)
                with col1:
                    churn_dist = df['churn_probability'].apply(
                        lambda x: 'High Risk' if x > 0.5 else 'Medium Risk' if x > 0.3 else 'Low Risk'
                    ).value_counts()
                    fig = px.bar(x=churn_dist.index, y=churn_dist.values,
                                title='Churn Risk Distribution')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    segment_churn = df.groupby('segment')['churn_probability'].mean().reset_index()
                    fig = px.bar(segment_churn, x='segment', y='churn_probability',
                                title='Average Churn Probability by Segment')
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("🔮 Predict Customer Metrics")
        
        if not st.session_state.models_trained:
            st.warning("⚠️ Please train models first in the 'Load & Train' tab.")
        else:
            st.markdown("### Enter Customer Information")
            
            with st.form("prediction_form"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Demographics**")
                    age = st.number_input("Age", min_value=18, max_value=100, value=30)
                    gender = st.selectbox("Gender", ['M', 'F', 'Other'])
                    city = st.selectbox("City", ['Dhaka', 'Chittagong', 'Sylhet'])
                    days_since_reg = st.number_input("Days Since Registration", min_value=0, value=365)
                
                with col2:
                    st.markdown("**Purchase Behavior**")
                    purchase_freq = st.number_input("Purchase Frequency", min_value=0, value=5)
                    avg_purchase_value = st.number_input("Avg Purchase Value", min_value=0, value=2000)
                    last_purchase_days = st.number_input("Days Since Last Purchase", min_value=0, value=30)
                    total_spent = st.number_input("Total Spent", min_value=0, value=10000)
                    favorite_category = st.selectbox("Favorite Category", ['Electronics', 'Fashion'])
                
                with col3:
                    st.markdown("**Digital Engagement**")
                    session_count = st.number_input("Session Count", min_value=0, value=20)
                    avg_session_duration = st.number_input("Avg Session Duration (sec)", min_value=0, value=300)
                    cart_abandonment = st.slider("Cart Abandonment Rate", 0.0, 1.0, 0.3)
                    email_open_rate = st.slider("Email Open Rate", 0.0, 1.0, 0.4)
                    click_through_rate = st.slider("Click-Through Rate", 0.0, 1.0, 0.2)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    preferred_channel = st.selectbox("Preferred Channel", ['Website', 'Mobile App'])
                    device_type = st.selectbox("Device Type", ['Desktop', 'Mobile', 'Tablet'])
                
                with col2:
                    nps_score = st.slider("NPS Score", -100, 100, 50)
                    support_tickets = st.number_input("Support Tickets", min_value=0, value=2)
   
