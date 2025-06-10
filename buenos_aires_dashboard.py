# buenos_aires_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------------------------------
# Page Configuration
# -------------------------------------------
st.set_page_config(
    page_title="Buenos Aires Real Estate Dashboard",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------
# Custom CSS
# -------------------------------------------
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .sidebar .sidebar-content {background-color: white;}
    h1 {color: #2c3e50;}
    h2 {color: #3498db;}
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------
# Data Loading and Preprocessing
# -------------------------------------------
@st.cache_data
def load_data(filepath):
    try:
        df = pd.read_csv(filepath, encoding='latin1')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='cp1252')

    mask_ba = df["place_with_parent_names"].str.contains("Capital Federal")
    mask_apt = df["property_type"] == "apartment"
    mask_price = df["price_aprox_usd"] < 400_000
    df = df[mask_ba & mask_apt & mask_price]

    low, high = df["surface_covered_in_m2"].quantile([0.1, 0.9])
    df = df[df["surface_covered_in_m2"].between(low, high)]

    return df

# -------------------------------------------
# Model Training
# -------------------------------------------
@st.cache_data
def train_model(data):
    data = data.dropna(subset=['surface_covered_in_m2', 'price_aprox_usd'])

    X = data[['surface_covered_in_m2']]
    y = data['price_aprox_usd']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test
    }

    return model, metrics

# -------------------------------------------
# Load Data and Train Model
# -------------------------------------------
df = load_data('C:/Users/STEVE/OneDrive/Desktop/PYTHON/data/02-housing-in-buenos-aires/buenos-aires-real-estate-1.csv')
model, metrics = train_model(df)

# -------------------------------------------
# Sidebar Filters
# -------------------------------------------
st.sidebar.title("Filters & Controls")

neighborhoods = df['place_with_parent_names'].str.split('|').str[-2].str.strip().unique()
selected_neighborhoods = st.sidebar.multiselect(
    "Select Neighborhoods",
    options=sorted(neighborhoods),
    default=sorted(neighborhoods)[:3]
)

price_min, price_max = int(df['price_aprox_usd'].min()), int(df['price_aprox_usd'].max())
price_range = st.sidebar.slider(
    "Price Range (USD)", min_value=price_min, max_value=price_max, value=(price_min, price_max)
)

area_min, area_max = int(df['surface_covered_in_m2'].min()), int(df['surface_covered_in_m2'].max())
area_range = st.sidebar.slider(
    "Covered Area Range (m²)", min_value=area_min, max_value=area_max, value=(area_min, area_max)
)

filtered_df = df[
    (df['place_with_parent_names'].str.split('|').str[-2].str.strip().isin(selected_neighborhoods)) &
    (df['price_aprox_usd'].between(price_range[0], price_range[1])) &
    (df['surface_covered_in_m2'].between(area_range[0], area_range[1]))
]

# -------------------------------------------
# Dashboard Title
# -------------------------------------------
st.title("🏙️ Buenos Aires Real Estate Dashboard")

st.markdown("""
This dashboard provides insights into apartment prices in Buenos Aires' Capital Federal district, 
including a price prediction model based on covered area.
""")

# -------------------------------------------
# Tabs
# -------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🏠 Price Analysis", "📈 Prediction Model", "🔍 Data Explorer"])

# Overview Tab
with tab1:
    st.header("Market Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Listings", len(filtered_df))
    col2.metric("Average Price (USD)", f"${filtered_df['price_aprox_usd'].mean():,.0f}")
    col3.metric("Avg Price per m²", f"${filtered_df['price_usd_per_m2'].mean():,.0f}")

    st.subheader("Geographical Distribution")
    if 'lat-lon' in filtered_df.columns and not filtered_df['lat-lon'].isna().all():
        filtered_df[['lat', 'lon']] = filtered_df['lat-lon'].str.split(',', expand=True).astype(float)
        fig = px.scatter_mapbox(
            filtered_df, lat='lat', lon='lon',
            color='price_aprox_usd', size='surface_covered_in_m2',
            hover_name='place_with_parent_names',
            hover_data=['price_aprox_usd', 'surface_covered_in_m2', 'rooms'],
            color_continuous_scale=px.colors.sequential.Viridis, zoom=12, height=500
        )
        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Location data not available for visualization")

    st.subheader("Price Distribution")
    fig = px.histogram(filtered_df, x='price_aprox_usd', nbins=30, color_discrete_sequence=['#3498db'])
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig, use_container_width=True)

# Price Analysis Tab
with tab2:
    st.header("Price Analysis")

    st.subheader("Price vs Covered Area")
    fig = px.scatter(
        filtered_df, x='surface_covered_in_m2', y='price_aprox_usd',
        color='place_with_parent_names', trendline="lowess",
        hover_data=['rooms', 'floor']
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Price per m² by Neighborhood")
    neighborhood_stats = filtered_df.groupby(
        filtered_df['place_with_parent_names'].str.split('|').str[-2].str.strip()
    ).agg({'price_usd_per_m2': 'mean'}).sort_values('price_usd_per_m2', ascending=False)

    fig = px.bar(
        neighborhood_stats, x=neighborhood_stats.index, y='price_usd_per_m2',
        color='price_usd_per_m2', color_continuous_scale=px.colors.sequential.Viridis
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Price Distribution by Neighborhood")
    fig = px.box(
        filtered_df, x=filtered_df['place_with_parent_names'].str.split('|').str[-2].str.strip(),
        y='price_aprox_usd'
    )
    st.plotly_chart(fig, use_container_width=True)

# Prediction Model Tab
with tab3:
    st.header("Price Prediction Model")
    st.markdown("""
    Simple linear regression model trained to predict apartment prices based on covered area.
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Training R² Score", f"{metrics['train_r2']:.2f}")
    col2.metric("Test R² Score", f"{metrics['test_r2']:.2f}")
    col3.metric("Training MAE (USD)", f"${metrics['train_mae']:,.0f}")
    col4.metric("Test MAE (USD)", f"${metrics['test_mae']:,.0f}")

    st.subheader("Model Visualization")
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=metrics['X_train']['surface_covered_in_m2'], y=metrics['y_train'],
        mode='markers', name='Training Data', marker=dict(color='blue', opacity=0.5)
    ))

    fig.add_trace(go.Scatter(
        x=metrics['X_test']['surface_covered_in_m2'], y=metrics['y_test'],
        mode='markers', name='Test Data', marker=dict(color='green', opacity=0.5)
    ))

    x_range = np.linspace(df['surface_covered_in_m2'].min(), df['surface_covered_in_m2'].max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))
    fig.add_trace(go.Scatter(x=x_range, y=y_range, name='Regression Line', line=dict(color='red', width=3)))

    fig.update_layout(xaxis_title='Covered Area (m²)', yaxis_title='Price (USD)')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Price Estimator")
    area_input = st.number_input("Enter covered area (m²)", min_value=10, max_value=500, value=70)
    predicted_price = model.predict([[area_input]])[0]
    st.metric("Estimated Price", f"${predicted_price:,.0f}")

# Data Explorer Tab
with tab4:
    st.header("Data Explorer")

    st.subheader("Filtered Data")
    st.dataframe(filtered_df, use_container_width=True)

    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download filtered data as CSV",
        data=csv,
        file_name="buenos_aires_real_estate_filtered.csv",
        mime="text/csv"
    )

    st.subheader("Data Statistics")
    st.dataframe(filtered_df.describe(), use_container_width=True)

    st.subheader("Correlation Matrix")
    numeric_cols = filtered_df.select_dtypes(include=['float64', 'int64']).columns
    corr_matrix = filtered_df[numeric_cols].corr()
    fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu', zmin=-1, zmax=1)
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
**About this dashboard:**
- Data source: Properati Buenos Aires real estate listings
- Model: Simple linear regression (price ~ area)
- Last updated: July 2023
""")
