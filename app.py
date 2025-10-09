import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import plotly.express as px


# --- Page Configuration ---
st.set_page_config(
    page_title="E-commerce Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)

# --- Caching Functions for Performance ---
@st.cache_data
def load_and_process_data(file_path):
    """Loads, cleans, and processes the data to generate RFM metrics."""
    # Load data
    # In your load_and_process_data function
    df = pd.read_csv('data.zip', encoding='latin1', compression='zip')
    
    # Data Cleaning from the notebook
    df = df.dropna(subset=['CustomerID'])
    df.drop_duplicates(inplace=True)
    df = df[df['Quantity'] > 0]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='mixed')
    df['CustomerID'] = df['CustomerID'].astype(int)
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    
    # RFM Calculation
    snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda date: (snapshot_date - date.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    })
    rfm.rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'MonetaryValue'}, inplace=True)
    
    return rfm, df

@st.cache_resource
def train_models(rfm_data):
    """Performs clustering and trains the predictive model."""
    # Log Transform and Scale RFM data for clustering
    rfm_log = np.log1p(rfm_data)
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    
    # K-Means Clustering
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(rfm_scaled)
    rfm_data['Cluster'] = clusters
    
    # Define segment names based on cluster analysis
    segment_map = {
        0: 'Potential Loyalists',
        1: 'Champions',
        2: 'At-Risk Customers',
        3: 'Hibernating / Lost'
    }
    rfm_data['Segment'] = rfm_data['Cluster'].map(segment_map)
    
    # Train Predictive Model (Random Forest)
    X = rfm_data[['Recency', 'Frequency', 'MonetaryValue']]
    y = rfm_data['Cluster']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    
    return rf_model, scaler, rfm_data

# --- Load and Train ---
try:
    rfm, original_df = load_and_process_data('data.csv')
    rf_model, scaler, rfm_with_clusters = train_models(rfm.copy())
except FileNotFoundError:
    st.error("Error: `data.csv` not found. Please make sure the data file is in the same directory as `app.py`.")
    st.stop()

# --- UI: Sidebar ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Project Overview", "Exploratory Data Analysis", "Customer Segments", "Predict New Customer"])

# --- UI: Main Content ---

if page == "Project Overview":
    st.title("🛍️ E-commerce Customer Segmentation")
    st.markdown("""
    This application analyzes customer transaction data to segment customers into distinct groups based on their purchasing behavior. 
    - **Recency (R):** How recently a customer has made a purchase.
    - **Frequency (F):** How often a customer makes a purchase.
    - **Monetary (M):** How much money a customer spends.
    
    We use **K-Means Clustering** to identify segments and a **Random Forest Classifier** to predict the segment for new customers.
    """)
    
    st.header("Raw Data Snippet")
    st.dataframe(original_df.head())
    
    st.header("Calculated RFM Data")
    st.dataframe(rfm.head())

elif page == "Exploratory Data Analysis":
    st.title("🔍 Exploratory Data Analysis (EDA)")
    
    st.header("RFM Variable Distributions (Before Scaling)")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(rfm['Recency'], bins=30, kde=True, ax=axes[0]).set_title('Recency Distribution')
    sns.histplot(rfm['Frequency'], bins=30, kde=True, ax=axes[1]).set_title('Frequency Distribution')
    sns.histplot(rfm['MonetaryValue'], bins=30, kde=True, ax=axes[2]).set_title('Monetary Value Distribution')
    st.pyplot(fig)

    st.header("Correlation Between RFM Variables")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(rfm.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

elif page == "Customer Segments":
    st.title("📊 Customer Segments")

    st.header("3D Scatter Plot of Customer Segments")
    fig_3d = px.scatter_3d(rfm_with_clusters, x='Recency', y='Frequency', z='MonetaryValue',
                           color='Segment', symbol='Segment',
                           title='Customer Segments in 3D RFM Space')
    fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=40))
    st.plotly_chart(fig_3d, use_container_width=True)
    
    st.header("Segment Profiles")
    segment_profiles = rfm_with_clusters.groupby('Segment').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'MonetaryValue': 'mean',
        'Cluster': 'count'
    }).rename(columns={'Cluster': 'Count'}).sort_values(by='MonetaryValue', ascending=False)
    st.dataframe(segment_profiles)

    st.header("Data with Assigned Segments")
    st.dataframe(rfm_with_clusters.head())

elif page == "Predict New Customer":
    st.title("🔮 Predict New Customer Segment")
    st.markdown("Enter a customer's RFM values to predict which segment they belong to.")
    
    st.sidebar.header("Customer Input")
    recency = st.sidebar.number_input("Recency (days since last purchase)", min_value=1, max_value=500, value=30)
    frequency = st.sidebar.number_input("Frequency (total number of purchases)", min_value=1, max_value=100, value=5)
    monetary = st.sidebar.number_input("Monetary Value (total spent)", min_value=1.0, max_value=10000.0, value=500.0, format="%.2f")
    
    if st.button("Predict Segment"):
        # Create a DataFrame for the new customer
        new_customer_data = pd.DataFrame({
            'Recency': [recency],
            'Frequency': [frequency],
            'MonetaryValue': [monetary]
        })
        
        # Log transform and scale the data
        new_customer_log = np.log1p(new_customer_data)
        new_customer_scaled = scaler.transform(new_customer_log)
        
        # Predict the cluster
        prediction = rf_model.predict(new_customer_scaled)
        
        # Map prediction to segment name
        segment_map = {
            0: 'Potential Loyalists',
            1: 'Champions',
            2: 'At-Risk Customers',
            3: 'Hibernating / Lost'
        }
        predicted_segment = segment_map[prediction[0]]
        
        st.success(f"### This customer is a **{predicted_segment}**! 🎉")
        
        st.subheader("Actionable Recommendation:")
        if predicted_segment == 'Champions':
            st.info("Reward them! Offer VIP perks and early access to new products.")
        elif predicted_segment == 'Potential Loyalists':
            st.info("Nurture them. Offer a discount on their next purchase to build loyalty.")
        elif predicted_segment == 'At-Risk Customers':
            st.warning("Re-engage them! Send a personalized 'We miss you' email with a valuable offer.")
        else: # Hibernating / Lost

            st.error("Try to win them back with a high-value offer. If no response, reduce marketing.")
