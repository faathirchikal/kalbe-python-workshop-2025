import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import sample_colorscale


from lifelines import KaplanMeierFitter
from sklearn.linear_model import ElasticNetCV
import statsmodels.api as sm

# Streamlit Page Configuration
st.set_page_config(
    page_title="Sales Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    """Load and cache the data"""
    try:
        df = pd.read_parquet("data/preprocessed/cleaned_data.parquet")
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'data/preprocessed/cleaned_data.parquet' exists.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

@st.cache_data
def load_elasticity_summary():
    return pd.read_parquet('data/preprocessed/price_elasticity.parquet')

@st.cache_data
def load_market_basket():
    return pd.read_parquet('data/preprocessed/market_basket.parquet')
    
def create_filters(df: pd.DataFrame):
    """Create sidebar filters with enriched labels"""
    st.sidebar.header("Filters")
    
    # Total records for percentage calculations
    total_records = len(df)
    
    # Country filter with percentages
    country_counts = df['country'].value_counts()
    country_labels = ['All'] + [
        f"{country} ({country_counts[country] / total_records:.1%} of Data)"
        for country in country_counts.index
    ]
    selected_country_label = st.sidebar.selectbox("Select Country", country_labels)
    
    # Extract actual country value
    if selected_country_label == 'All':
        selected_country = 'All'
    else:
        selected_country = selected_country_label.split(' (')[0]
    
    # Date range filter
    min_date = df['order_date'].min().date()
    max_date = df['order_date'].max().date()
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        help = "Select start and end date"
    )
    
    # SKU filter sorted by highest adjusted GMV
    sku_gmv = df.groupby('sku_name')['adjusted_gmv'].sum()
    sorted_skus = sku_gmv.sort_values(ascending=False).index.tolist()
    sku_labels = ['All'] + sorted_skus
    selected_sku = st.sidebar.selectbox(
        "Select SKU",
        sku_labels
    )
    
    return selected_country, date_range, selected_sku


def filter_data(df, country, date_range, sku):
    """Apply filters to the dataframe"""
    filtered_df = df.copy()
    
    # Country filter
    if country != 'All':
        filtered_df = filtered_df[filtered_df['country'] == country]
    
    # Date filter
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['order_date'].dt.date >= start_date) & 
            (filtered_df['order_date'].dt.date <= end_date)
        ]
    
    # SKU filter
    if sku != 'All':
        filtered_df = filtered_df[filtered_df['sku_name'] == sku]
    
    return filtered_df

def create_scorecard(df):
    """Create scorecard metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_quantity = df['quantity'].sum()
        st.metric(
            label="Quantity Sold",
            value=f"{total_quantity:,}",
            help="Total number of items sold"
        )
    
    with col2:
        total_sales = df['adjusted_gmv'].sum()
        st.metric(
            label="GMV",
            value=f"£{total_sales:,.0f}",
            help="Total Gross Merchandise Value"
        )
    
    with col3:
        total_orders = df[~df['order_id_cancelled']]['order_id'].nunique()
        st.metric(
            label="Total Orders",
            value=f"{total_orders:,}",
            help="Total number of unique orders"
        )
    
    with col4:
        total_customers = df[~df['order_id_cancelled']]['customer_id'].nunique()
        st.metric(
            label="Total Customers",
            value=f"{total_customers:,}",
            help="Total number of unique customers"
        )

def plot_monthly_sales(df):
    """Create monthly sales vs. unique customers with dual y‑axes."""
    # filter out December 2011, aggregate GMV & unique customers
    df = df[df['order_date_month'] < '2011-12-01'].copy()
    monthly = (
        df
        .assign(order_date_month=pd.to_datetime(df['order_date_month']))
        .groupby('order_date_month')
        .agg(
            adjusted_gmv=('adjusted_gmv', 'sum'),
            unique_customers=('customer_id', 'nunique')  # adjust column if needed
        )
        .round(0)  # round both metrics
        .reset_index()
    )

    gmv_max = monthly['adjusted_gmv'].max() * 1.05  # 5% headroom
    cust_max = monthly['unique_customers'].max() * 1.05
    # create dual‑axis figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # GMV trace (left axis) with a warm color
    fig.add_trace(
        go.Scatter(
            x=monthly['order_date_month'],
            y=monthly['adjusted_gmv'],
            name="GMV",
            mode="lines+markers",
            line=dict(color='darkblue', width=3),
            marker=dict(size=8),
            hovertemplate="£%{y:,.0f}"
        ),
        secondary_y=False
    )

    # Unique customers trace (right axis) with a cool color
    fig.add_trace(
        go.Scatter(
            x=monthly['order_date_month'],
            y=monthly['unique_customers'],
            name="Unique Customers",
            mode="lines+markers",
            line=dict(color='darkgreen', width=3),
            marker=dict(size=8),
            hovertemplate="%{y:,}"
        ),
        secondary_y=True
    )

    # force both y‑axes to start at zero
    fig.update_yaxes(
        title_text="GMV (£)",
        tickformat="£,.0f",
        range=[0, gmv_max],
        secondary_y=False
    )
    fig.update_yaxes(
        title_text="Unique Customers",
        range=[0, cust_max],
        secondary_y=True
    )

    # x‑axis formatting
    fig.update_xaxes(
        title_text="Month",
        tickformat="%b %Y"
    )

    # move legend to top
    fig.update_layout(
        title="Monthly GMV & Unique Customers",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )

    return fig


def plot_top_products(df):
    """Create top products by GMV chart"""
    # aggregate GMV and compute percentages
    df = df[~df['order_id_cancelled']]
    per_product_sales = (
        df.groupby('sku_name')
          .agg(gmv=('adjusted_gmv', 'sum'))
          .reset_index()
    )
    total_gmv = per_product_sales['gmv'].sum()
    per_product_sales['pct'] = per_product_sales['gmv'] / total_gmv * 100

    # pick top 10 and sort for horizontal bar
    top_products = (
        per_product_sales
        .sort_values('gmv', ascending=False)
        .head(10)
        .reset_index(drop=True)
        .sort_values('gmv')
    )

    # build bar chart
    fig = px.bar(
        top_products,
        x='gmv',
        y='sku_name',
        orientation='h',
        title='Top 10 Products by GMV',
        labels={'gmv': 'GMV (£)', 'sku_name': 'Product'},
        text=[f"{pct:.1f}%" for pct in top_products['pct']]
    )

    # apply dark blue color and simplify hover to just GMV
    fig.update_traces(
        marker_color='darkblue',
        textposition='inside',
        hovertemplate="£%{x:,.0f}"
    )

    fig.update_layout(
        xaxis_title="GMV (£)",
        xaxis_tickformat='£,.0f',
        yaxis_title="",
        height=500
    )

    return fig

def plot_top_countries(df):
    """Create top countries by GMV chart"""
    df = df[~df['order_id_cancelled']]
    per_country_sales = (
        df.groupby('country')
        .agg(gmv=('adjusted_gmv', 'sum'))
        .reset_index()
    )
    
    total_gmv = per_country_sales['gmv'].sum()
    per_country_sales['pct'] = per_country_sales['gmv'] / total_gmv * 100
    
    top_countries = (
        per_country_sales
        .sort_values('gmv', ascending=False)
        .head(10)
        .reset_index(drop=True)
        .sort_values('gmv')
    )
    
    fig = px.bar(
        top_countries,
        x='gmv',
        y='country',
        orientation='h',
        title='Top 10 Countries by GMV',
        labels={'gmv': 'GMV (£)', 'country': 'Country'},
        text=[f"{pct:.1f}%" for pct in top_countries['pct']]
    )
    
    fig.update_traces(
        marker_color='darkblue',
        textposition='outside',
        hovertemplate="£%{x:,.0f}"
    )
    
    fig.update_layout(
        xaxis_title="GMV (£)",
        xaxis_tickformat='£,.0f',
        yaxis_title="",
        height=500
    )
    
    return fig

def plot_rfm_treemap(df):
    """Plot an RFM treemap with segment-level ranges and customer share in dark green palette."""
    # Filter out cancelled orders
    df = df[~df['order_id_cancelled']]

    # Calculate RFM metrics per customer
    now = df['order_date'].max() + timedelta(days=1)
    rfm = (
        df.groupby('customer_id')
          .agg(
              recency=('order_date', lambda x: (now - x.max()).days),
              frequency=('order_id', 'nunique'),
              monetary=('adjusted_gmv', 'sum')
          )
          .reset_index()
    )

    # Score metrics into quartiles using rank in case of duplicates
    rfm['recency_rank'] = rfm['recency'].rank(method='first')
    rfm['frequency_rank'] = rfm['frequency'].rank(method='first')
    rfm['monetary_rank'] = rfm['monetary'].rank(method='first')
    try:
        rfm['R'] = pd.qcut(rfm['recency_rank'], 4, labels=[4,3,2,1]).astype(int)
    except ValueError:
        rfm['R'] = 1
    
    try:
        rfm['F'] = pd.qcut(rfm['frequency_rank'], 4, labels=[1,2,3,4]).astype(int)
    except ValueError:
        rfm['F'] = 1
    
    try:
        rfm['M'] = pd.qcut(rfm['monetary_rank'], 4, labels=[1,2,3,4]).astype(int)
    except ValueError:
        rfm['M'] = 1

    if rfm['recency'].nunique() <= 1 or rfm['frequency'].nunique() <= 1 or rfm['monetary'].nunique() <= 1:
        st.warning("Not enough variation in RFM data to compute segments.")
        return go.Figure()
    rfm['R'] = pd.qcut(rfm['recency_rank'], 4, labels=[4,3,2,1]).astype(int)
    rfm['F'] = pd.qcut(rfm['frequency_rank'], 4, labels=[1,2,3,4]).astype(int)
    rfm['M'] = pd.qcut(rfm['monetary_rank'], 4, labels=[1,2,3,4]).astype(int)
    rfm['RFM_Score'] = rfm[['R','F','M']].sum(axis=1)

    # Map to named segments
    def name_segment(row):
        if row['R']==4 and row['F']==4 and row['M']==4:
            return 'Champions'
        if row['R']>=3 and row['F']>=3 and row['M']>=3:
            return 'Loyal Customers'
        if row['R']==4:
            return 'Recent Customers'
        if row['F']>=3:
            return 'Frequent Customers'
        if row['M']>=3:
            return 'Big Spenders'
        if row['R']==1 and row['F']==1 and row['M']==1:
            return 'Lost Cheap'
        return 'Needs Attention'
    rfm['Segment'] = rfm.apply(name_segment, axis=1)

    # Compute segment-level stats
    segment_stats = (
        rfm.groupby('Segment')
           .agg(
               recency_min=('recency','min'), recency_max=('recency','max'),
               freq_min=('frequency','min'), freq_max=('frequency','max'),
               mon_min=('monetary','min'), mon_max=('monetary','max'),
               cust_count=('customer_id','count')
           )
    ).reset_index()
    total_customers = len(rfm)
    segment_stats['Pct_Customers'] = (segment_stats['cust_count'] / total_customers * 100).round(1).astype(str) + '%'

    # Merge back for hover data
    rfm = rfm.merge(segment_stats, on='Segment', how='left')

    # Prepare hover labels
    rfm['Recency_Range'] = rfm.apply(lambda x: f"{int(x.recency_min)}–{int(x.recency_max)} days", axis=1)
    rfm['Frequency_Range'] = rfm.apply(lambda x: f"{int(x.freq_min)}–{int(x.freq_max)} orders", axis=1)
    rfm['Monetary_Range'] = rfm.apply(lambda x: f"£{x.mon_min:,.0f}–£{x.mon_max:,.0f}", axis=1)
    rfm['Customer_Pct'] = rfm['Pct_Customers']

    # Build treemap with customer percentage
    fig = px.treemap(
        rfm,
        path=['Segment'],
        values='monetary',
        color='RFM_Score',
        title='Customer RFM Segments',
        color_continuous_scale=[[0, 'lightgreen'], [1, 'darkgreen']],
        custom_data=['Recency_Range','Frequency_Range','Monetary_Range','Customer_Pct']
    )
    # Custom hover template including customer share
    fig.update_traces(
        hovertemplate=(
            '<b>%{label}</b><br>' +
            '%{customdata[0]}<br>' +
            '%{customdata[1]}<br>' +
            '%{customdata[2]}<br>' +
            'Share of Customers: %{customdata[3]}<extra></extra>'
        )
    )
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25), height=500)
    return fig


def plot_monthly_customer_area(df):
    """Monthly area of new vs existing customers (non-cumulative)."""
    df = df[~df['order_id_cancelled']].copy()
    df['month'] = df['order_date'].dt.to_period('M').dt.to_timestamp()

    # Determine first purchase month for each customer
    first_orders = df.groupby('customer_id')['order_date'].min().dt.to_period('M').dt.to_timestamp()
    df['first_month'] = df['customer_id'].map(first_orders)

    # New customers: customers whose first order is in that month
    new_customers = df[df['month'] == df['first_month']]
    new_monthly = new_customers.groupby('month')['customer_id'].nunique().reset_index(name='new_customers')

    # Total customers per month
    total_monthly = df.groupby('month')['customer_id'].nunique().reset_index(name='total_customers')

    # Existing customers = total - new for that month
    merged = pd.merge(total_monthly, new_monthly, on='month', how='left').fillna(0)
    merged['existing_customers'] = merged['total_customers'] - merged['new_customers']

    # Prepare data for area plot
    area_data = merged[['month', 'new_customers', 'existing_customers']].melt(
        id_vars='month', var_name='type', value_name='customers')

    fig = px.area(
        area_data,
        x='month',
        y='customers',
        color='type',
        title='Monthly New vs Existing Customers',
        labels={'month': 'Month', 'customers': 'Customers', 'type': 'Customer Type'},
        height=500,
        color_discrete_map={
            'new_customers': 'lightgreen',
            'existing_customers': 'darkgreen'
        }
    )
    fig.update_traces(
        hovertemplate='<b>%{fullData.name}</b><br>%{x|%b %Y}<br>Customers: %{y:,}<extra></extra>'
    )
    fig.update_layout(legend_title_text='Customer Type', xaxis=dict(tickformat='%b %Y'))
    return fig


def plot_order_frequency(df):
    """Binned order frequency histogram."""
    freq = df.groupby('customer_id')['order_id'].nunique()
    bins = [1,2,4,7,11, np.inf]
    labels = ['1','2-3','4-6','7-10','11+']
    freq_binned = pd.cut(freq, bins=bins, labels=labels, right=False)
    dist = freq_binned.value_counts().reindex(labels).reset_index()
    dist.columns = ['Order Frequency', 'Count']

    fig = px.bar(
        dist,
        x='Order Frequency',
        y='Count',
        title='Order Frequency Distribution',
        text='Count',
        height=500
    )
    fig.update_traces(marker_color='darkgreen', textposition='outside')
    return fig


def plot_purchase_interval_cdf(df):
    """
    CDF-like plot of days between purchases with proper censoring.

    Parameters:
        df (DataFrame): Must contain ['customer_id', 'order_date_only']
    """
    df = df[['customer_id', 'order_date_only']].drop_duplicates()
    df = df.sort_values(['customer_id', 'order_date_only'])
    df['order_date_only'] = pd.to_datetime(df['order_date_only'])

    # Compute purchase intervals
    df['prev_date'] = df.groupby('customer_id')['order_date_only'].shift(1)
    df['interval'] = (df['order_date_only'] - df['prev_date']).dt.days

    # Censoring
    analysis_date = df['order_date_only'].max()
    last_orders = df.groupby('customer_id')['order_date_only'].max().reset_index()
    last_orders['prev_date'] = df.groupby('customer_id')['order_date_only'].nth(-2).reset_index(drop=True)
    last_orders['interval'] = (analysis_date - last_orders['order_date_only']).dt.days
    last_orders['event_observed'] = 0  # Censored

    observed = df.dropna(subset=['interval']).copy()
    observed['event_observed'] = 1

    survival_df = pd.concat([
        observed[['interval', 'event_observed']],
        last_orders[['interval', 'event_observed']]
    ])

    # Fit Kaplan-Meier
    kmf = KaplanMeierFitter()
    kmf.fit(durations=survival_df['interval'], event_observed=survival_df['event_observed'])

    # CDF = 1 - Survival
    surv_df = kmf.survival_function_.reset_index()
    surv_df.columns = ['days_between', 'survival_rate']
    surv_df['cdf'] = 1 - surv_df['survival_rate']

    # Filter for first 3 months
    surv_df = surv_df[surv_df['days_between'] <= 90]

    # Find median day (where survival_rate ~ 0.5 or cdf ~ 0.5)
    median_day = kmf.median_survival_time_

    # Plot
    fig = go.Figure()

    # CDF line
    fig.add_trace(go.Scatter(
        x=surv_df['days_between'],
        y=surv_df['cdf'],
        mode='lines',
        name='CDF',
        line=dict(color='darkgreen', width=3)
    ))

    # Horizontal 50% line
    fig.add_trace(go.Scatter(
        x=[0, surv_df['days_between'].max()],
        y=[0.5, 0.5],
        mode='lines',
        name='50% Mark',
        line=dict(color='gray', dash='dash'),
        showlegend=False
    ))

    # Vertical line at median
    if np.isfinite(median_day):
        # Vertical line at median
        fig.add_trace(go.Scatter(
            x=[median_day, median_day],
            y=[0, 0.5],
            mode='lines',
            line=dict(color='gray', dash='dash'),
            name=f'Median: {int(median_day)} days',
        ))
    else:
        # Add annotation instead if median is infinite
        fig.add_annotation(
            x=surv_df['days_between'].max(),
            y=0.5,
            text="Median not reached",
            showarrow=False,
            font=dict(color="gray", size=12),
            xanchor='right'
        )

    fig.update_layout(
        title='Cumulative Distribution of Days Between Purchases',
        xaxis_title='Days Between Purchases',
        yaxis_title='Proportion',
        height=500,
        hovermode='x unified'
    )
    fig.update_yaxes(tickformat='.0%', range=[0, 1])
    fig.update_xaxes(range=[0, 90])
    return fig
    
def plot_price_elasticity(filtered_df, selected_sku):
    """
    Plotly figure showing price elasticity curve for a selected SKU.
    Title shows the elasticity effect instead of formula.
    """
    if not selected_sku or selected_sku == 'All':
        return None

    # Get SKU row & ID
    sku_row = filtered_df[filtered_df['sku_name'] == selected_sku]
    if sku_row.empty:
        return None
    sku_id = sku_row['sku_id'].iloc[0]  # needed for filtering only
    sku_name = selected_sku

    # Aggregate & transform
    df_agg = (
        filtered_df
        .groupby(['sku_id', 'order_date_only'])
        .agg(
            avg_price=('adjusted_price', 'mean'),
            total_qty=('quantity', 'sum')
        )
        .reset_index()
    )

    df_agg = df_agg[
        (df_agg['avg_price'] > 0) & (df_agg['total_qty'] > 0)
    ].assign(
        log_price=lambda d: np.log(d['avg_price']),
        log_qty=lambda d: np.log(d['total_qty'])
    )

    grp = df_agg[df_agg['sku_id'] == sku_id]
    if grp.empty or grp['avg_price'].nunique() < 2:
        return None

    # Fit model
    X_log = sm.add_constant(grp['log_price'])
    y_log = grp['log_qty']
    model = sm.OLS(y_log, X_log).fit()

    beta_0 = model.params['const']
    elasticity = model.params['log_price']
    r2 = model.rsquared

    # Predicted curve
    x_vals = np.linspace(grp['avg_price'].min(), grp['avg_price'].max(), 100)
    y_vals = np.exp(beta_0) * (x_vals ** elasticity)

    qty_change_pct = elasticity * 1
    direction = "decrease" if elasticity < 0 else "increase"

    # Build Plotly figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=grp['avg_price'],
        y=grp['total_qty'],
        mode='markers',
        marker=dict(size=8, color='steelblue', line=dict(width=1, color='black')),
        name='Observed',
        hovertemplate="Price: %{x:.2f}<br>Qty: %{y:.0f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines',
        line=dict(color='firebrick', width=2),
        name='Elasticity Curve'
    ))

    # Layout with R² added to subtitle
    fig.update_layout(
        title=(
            f"{sku_name}<br>"
            f"<span style='font-size:14px'>Elasticity: {elasticity:.2f} "
            f"(1% ↑ price → {abs(qty_change_pct):.1f}% {direction} in qty) | R² = {r2:.2f}</span>"
        ),
        xaxis_title="Average Price",
        yaxis_title="Total Quantity",
        template="simple_white",
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="right", x=0.99,  # <-- top right
            bgcolor='rgba(255,255,255,0.8)',  # optional: add white background
            bordercolor='gray',
            borderwidth=1
        )
    )

    return fig

def plot_unique_customers_per_hour(df):
    """
    Box‑and‑whisker of daily unique customers by hour of day,
    colored from light‑green (low median) to dark‑green (high median).
    Only the box fill varies; outlines and outliers stay black.
    
    Expects df with at least ['order_date', 'customer_id'].
    """
    df = df.copy()
    df['order_date_only'] = pd.to_datetime(df['order_date']).dt.date
    df['order_hour'] = pd.to_datetime(df['order_date']).dt.hour
    df = df[df['order_hour'] > 6]  # filter early morning

    # Group by date and hour
    per_hour = (
        df.groupby(['order_date_only', 'order_hour'])
          .agg(unique_customers=('customer_id', 'nunique'))
          .reset_index()
    )

    if per_hour.empty:
        return go.Figure().update_layout(
            title="No data available for selected filters",
            xaxis_title="Hour of Day",
            yaxis_title="Unique Customers per Day",
            template="simple_white",
        )

    # compute normalized median per hour
    medians = per_hour.groupby('order_hour')['unique_customers'].median()
    norm = (medians - medians.min()) / (medians.max() - medians.min())

    fig = go.Figure()
    valid_hours = sorted(per_hour['order_hour'].unique())

    for hour in valid_hours:
        hour_data = per_hour.loc[per_hour['order_hour'] == hour, 'unique_customers']
        if hour_data.empty:
            continue

        # safely normalize color (fallback to light green)
        raw_val = norm.get(hour, 0.0)
        color_val = 0.0 if pd.isna(raw_val) else float(raw_val)
        color_str = sample_colorscale("Greens", color_val)[0]

        fig.add_trace(go.Box(
            y=hour_data,
            name=str(hour),
            boxpoints='outliers',
            fillcolor=color_str,
            line_color='black',
            marker_color='black'
        ))

    fig.update_layout(
        title="Customer Purchase Time Preference",
        xaxis_title="Hour of Day",
        yaxis_title="Unique Customers per Day",
        xaxis=dict(tickmode="array", tickvals=valid_hours),
        margin=dict(t=40, b=40, l=40, r=40),
        template="simple_white",
    )
    return fig
def plot_cancelled_products(df):
    """
    Horizontal bar chart of top cancelled SKUs using cancel rate (dark red palette).
    Static label only — no hover.
    """
    # 1. Aggregate cancel stats per SKU
    per_product = (
        df[~df['sku_id_no_digit']]
        .groupby('sku_name', as_index=False)
        .agg(
            cancel_percentage=('order_id_cancelled', 'mean'),
            total_order=('order_id', 'nunique'),
            cancelled_count=('order_id_cancelled', 'sum')
        )
    )

    # 2. Filter and take top 10
    top = (
        per_product[per_product['total_order'] >= 100]
        .sort_values(['cancel_percentage', 'total_order'], ascending=[False, False])
        .head(10).sort_values('cancel_percentage')
    ).reset_index(drop=True)

    if top.empty:
        return go.Figure().update_layout(
            title="No cancelled products available for selected filters",
            template="simple_white"
        )

    # 3. Create static text label
    top['label'] = top.apply(
        lambda r: f"{int(r.cancelled_count)} out of {int(r.total_order)} cancelled ({r.cancel_percentage:.1%})",
        axis=1
    )

    # 4. Plot with Plotly
    fig = px.bar(
        top,
        x='cancel_percentage',
        y='sku_name',
        orientation='h',
        text='label',
        color='cancel_percentage',
        color_continuous_scale='Reds',
    )

    # 5. Remove hover text and finalize layout
    fig.update_traces(
        textposition='outside',
        hoverinfo='skip',
        hovertemplate=None
    )

    fig.update_layout(
        xaxis_title="Cancel Rate",
        xaxis_tickformat=".0%",
        xaxis_range=[0, 1],
        yaxis_title="",
        coloraxis_showscale=False,
        title="SKU Cancel Rate",
        template="simple_white",
        margin=dict(l=140, r=20, t=50, b=20),
        uniformtext_minsize=9,
        uniformtext_mode='hide'
    )

    return fig
def main():
    """Main application"""
    st.title("Sales Analytics Dashboard")
    st.markdown("Welcome to your comprehensive sales analytics dashboard!")
    
    # Load data
    df = load_data()

    elas_reg_df = load_elasticity_summary()

    market_basket_df = load_market_basket()
    
    # Create filters
    selected_country, date_range, selected_sku = create_filters(df)
    
    # Apply filters
    filtered_df = filter_data(df, selected_country, date_range, selected_sku)
    
    # Show filtered data info
    st.sidebar.markdown(f"**Filtered Data:** {len(filtered_df):,} records")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["General Overview", "Customer Analysis", "Product Analysis", "Sales Forecast", "Lost Sales"])
    
    with tab1:
        st.header("General Overview")
    
        # Scorecard
        create_scorecard(filtered_df)
    
        st.markdown("---")
    
        # Monthly sales trend
        fig_monthly_sales = plot_monthly_sales(filtered_df)
        st.plotly_chart(fig_monthly_sales, use_container_width=True)
    
        st.markdown("##")  # small spacer
    
        # Top products & countries
        col_prod, col_country = st.columns([1, 1], gap="large")
        with col_prod:
            fig_top_products = plot_top_products(filtered_df)
            st.plotly_chart(fig_top_products, use_container_width=True)
    
        with col_country:
            fig_top_countries = plot_top_countries(filtered_df)
            st.plotly_chart(fig_top_countries, use_container_width=True)
            
    
    with tab2:
        st.header("Customer Analysis")
    
        # RFM treemap
        fig_rfm = plot_rfm_treemap(filtered_df)
        st.plotly_chart(fig_rfm, use_container_width=True)
    
        st.markdown("---")
    
        # Monthly customer area
        fig_customer_area = plot_monthly_customer_area(filtered_df)
        st.plotly_chart(fig_customer_area, use_container_width=True)
    
        st.markdown("---")
    
        col1, col2 = st.columns(2, gap="large")
        with col1:
            fig_order_freq = plot_order_frequency(filtered_df)
            st.plotly_chart(fig_order_freq, use_container_width=True)
    
        with col2:
            fig_purchase_cdf = plot_purchase_interval_cdf(filtered_df)
            st.plotly_chart(fig_purchase_cdf, use_container_width=True)

        st.markdown("---")
        
        fig_hourly = plot_unique_customers_per_hour(filtered_df)
        st.plotly_chart(fig_hourly, use_container_width=True)
    
    with tab3:
        st.header("Price Elasticity")
        
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            fig_elasticity = plot_price_elasticity(filtered_df, selected_sku)
            if fig_elasticity:
                st.plotly_chart(fig_elasticity)
            else:
                st.info("Please select a SKU to view its price elasticity.")
        
        with col2:
            st.subheader("Elasticity Summary")
            st.dataframe(
                elas_reg_df[['sku_name', 'elasticity', 'n_obs', 'n_price_pts', 'r2']].sort_values('r2', ascending=False).reset_index(drop=True),
                use_container_width=True
            )

        # Spacer
        st.markdown("###")
    
        # Optional divider
        st.markdown("---")
    
        # 📊 Market Basket Table
        st.subheader("Market Basket Association Rules")
        st.dataframe(
            market_basket_df[['antecedents', 'consequents', 'support', 'confidence']]
            .sort_values('confidence', ascending=False)
            .reset_index(drop=True),
            use_container_width=True
        )
    with tab4:
        st.header("Forecast Placeholder")

    with tab5:
        st.header("Lost Sales")
        
        # 1) show the metric
        lost = filtered_df.loc[filtered_df['order_id_cancelled'], 'adjusted_gmv'].sum()
        st.metric("Lost Sales", f"£{lost:,.0f}")
        
        st.markdown("---")
        
        # 2) show the cancelled‐products bar chart
        fig_cancel = plot_cancelled_products(filtered_df)
        st.plotly_chart(fig_cancel)
if __name__ == "__main__":
    main()