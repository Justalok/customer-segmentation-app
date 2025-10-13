import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
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
    df = pd.read_csv(file_path, encoding='latin1', compression='zip')
    
    # Data Cleaning
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
    
    return rfm

@st.cache_resource
def train_models(rfm_data):
    """Performs clustering and trains the predictive model."""
    # Log Transform and Scale RFM data for clustering
    rfm_log = np.log1p(rfm_data)
    scaler = StandardScaler()
    scaler.fit(rfm_log) # Fit the scaler here
    
    # K-Means Clustering
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    # Important: Use the scaled data for clustering
    rfm_scaled = scaler.transform(rfm_log)
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

# --- Load Data and Train Models ---
try:
    rfm = load_and_process_data('data.zip')
    rf_model, scaler, rfm_with_clusters = train_models(rfm.copy())
except FileNotFoundError:
    st.error("Error: `data.zip` not found. Please ensure the data file is in the same directory.")
    st.stop()

# --- UI: Sidebar ---
st.sidebar.title("🛍️ Customer Segmentation App")
st.sidebar.markdown("""
This app analyzes customer transaction data to create **RFM** (Recency, Frequency, Monetary) segments. It also predicts which segment a new customer will belong to.
""")
st.sidebar.divider()

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'segmentation'

# Sidebar buttons to switch pages
if st.sidebar.button("Segmentation Overview", use_container_width=True):
    st.session_state.page = 'segmentation'
if st.sidebar.button("Predict New Customer", use_container_width=True):
    st.session_state.page = 'prediction'


# --- UI: Main Content ---

# Display Segmentation Overview Page
if st.session_state.page == 'segmentation':
    st.title("E-Commerce Customer Segmentation Report")
    st.markdown("A complete overview of customer segments based on their purchase behavior.")

    # --- Business Overview Metrics ---
    st.header("Business Overview")
    total_customers = len(rfm_with_clusters)
    total_revenue = rfm_with_clusters['MonetaryValue'].sum()
    avg_recency = rfm_with_clusters['Recency'].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Total Revenue", f"${total_revenue:,.0f}")
    col3.metric("Average Recency", f"{avg_recency:.1f} days")
    st.divider()

    # --- Customer Segment Details ---
    st.header("Our Customer Segments")
    st.markdown("Based on their purchasing behavior (Recency, Frequency, Monetary), customers have been grouped into four distinct segments using K-Means clustering.")
    
    segment_profiles = rfm_with_clusters.groupby('Segment').agg(
        Customer_Count=('Segment', 'count'),
        Avg_Recency=('Recency', 'mean'),
        Avg_Frequency=('Frequency', 'mean'),
        Avg_Monetary=('MonetaryValue', 'mean')
    ).reset_index()

    segment_order = ["Champions", "Potential Loyalists", "At-Risk Customers", "Hibernating / Lost"]
    
    # Create 4 columns for the segments
    cols = st.columns(4)
    
    for i, segment_name in enumerate(segment_order):
        # Filter the DataFrame for the current segment
        segment_data = segment_profiles[segment_profiles['Segment'] == segment_name].iloc[0]
        
        with cols[i]:
            st.subheader(f"👥 {segment_name}")
            st.metric("Customer Count", f"{int(segment_data['Customer_Count']):,}")
            st.metric("Avg. Recency (days)", f"{segment_data['Avg_Recency']:.1f}")
            st.metric("Avg. Frequency", f"{segment_data['Avg_Frequency']:.1f}")
            st.metric("Avg. Monetary Value ($)", f"${segment_data['Avg_Monetary']:,.0f}")
            
    st.success("💡 **Key Takeaway**: 'Champions' are your most valuable customers. 'Hibernating' and 'At-Risk' customers represent key opportunities for re-engagement campaigns.")
    st.divider()

    # --- Visualizing the Segments ---
    st.header("Visualizing the Segments")
    fig_scatter = px.scatter(
        rfm_with_clusters,
        x='Recency',
        y='Frequency',
        size='MonetaryValue',
        color='Segment',
        hover_name=rfm_with_clusters.index,
        title="RFM Segments (Size by Monetary Value)",
        labels={'Recency': 'Recency (Days)', 'Frequency': 'Frequency (Purchases)'},
        color_discrete_map={
            'Champions': 'green',
            'Potential Loyalists': 'blue',
            'At-Risk Customers': 'orange',
            'Hibernating / Lost': 'red'
        }
    )
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.info("ℹ️ **Insight**: This plot visually confirms our segments. Notice the cluster of highly frequent, high-spending customers ('Champions') with low recency (bottom-right).")

# Display Prediction Page
elif st.session_state.page == 'prediction':
    st.title("🔮 Predict New Customer Segment")
    st.markdown("Enter a customer's RFM values below to predict their segment and get recommendations.")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            recency = st.number_input("Recency (days since last purchase)", min_value=1, max_value=500, value=30)
        with col2:
            frequency = st.number_input("Frequency (total purchases)", min_value=1, max_value=100, value=5)
        with col3:
            monetary = st.number_input("Monetary Value (total spent)", min_value=1.0, value=500.0, format="%.2f")
        
        submitted = st.form_submit_button("Predict Segment", type="primary")

    if submitted:
        # Create a DataFrame for the new customer
        new_customer_data = pd.DataFrame({
            'Recency': [recency],
            'Frequency': [frequency],
            'MonetaryValue': [monetary]
        })
        
        # Predict the cluster using the trained RandomForest model
        prediction_raw = rf_model.predict(new_customer_data)
        
        # Map prediction to segment name
        segment_map = {
            0: 'Potential Loyalists',
            1: 'Champions',
            2: 'At-Risk Customers',
            3: 'Hibernating / Lost'
        }
        predicted_segment = segment_map[prediction_raw[0]]
        
        st.success(f"### This customer is a **{predicted_segment}**! 🎉")
        
        st.subheader("Actionable Recommendation:")
        if predicted_segment == 'Champions':
            st.info("💡 **Action:** Reward them! Offer VIP perks, loyalty points, and early access to new products to maintain their loyalty.")
        elif predicted_segment == 'Potential Loyalists':
            st.info("💡 **Action:** Nurture them. Offer a discount on their next purchase or a subscription service to increase their frequency and build loyalty.")
        elif predicted_segment == 'At-Risk Customers':
            st.warning("💡 **Action:** Re-engage them immediately! Send a personalized 'We miss you' email with a valuable, time-sensitive offer to win them back.")
        else: # Hibernating / Lost
            st.error("💡 **Action:** Try a high-value, one-time offer to win them back. If there's no response, reduce marketing spend on them to focus on other segments.")
