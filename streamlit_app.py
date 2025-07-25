# Libraries
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
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
    page_title="Sales Analytics Dashboard", # Nama Page
    layout="wide",
    initial_sidebar_state="expanded" # Default state sidebar, on
)

# Data Loading ============================================================================================================================
@st.cache_data # Caching data agar tidak perlu reload 
def load_data(): 
    """Fungsi untuk load (cleaned) data"""
    try:
        df = pd.read_parquet("data/preprocessed/cleaned_data.parquet")
        df = df[df['order_date_month']<'2011-12-01'] # Filter out desember karena bulan belum selesai
        return df
    except FileNotFoundError: # Throw error jika tidak ada
        st.error("Data file not found. Please ensure 'data/preprocessed/cleaned_data.parquet' exists.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

@st.cache_data
def load_elasticity_summary(): # Price Elasticity, pastikan 05_(extra)_product_analysis.ipynb sudah di run
    return pd.read_parquet('data/preprocessed/price_elasticity.parquet')

@st.cache_data
def load_market_basket(): # Market Basket, pastikan 05_(extra)_product_analysis.ipynb sudah di run
    return pd.read_parquet('data/preprocessed/market_basket.parquet')

@st.cache_data
def load_forecast(): # Data Forecast, pastikan 03_forecast_preprocessing.ipynb dan 04_forecasting.ipynb sudah di run
    return pd.read_parquet('data/preprocessed/forecast_result.parquet')

@st.cache_data 
def convert_df_to_excel(df): # Fungsi untuk convert to downloadable excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Forecast')
    return output.getvalue()

# Filtering ============================================================================================================================
def create_filters(df: pd.DataFrame): 
    """Configurasi untuk Filter di kiri"""
    st.sidebar.header("Filters") # Nama sidebar
    
    # Total records 
    total_records = len(df)
    
    # List dari country dengan record terbanyak
    country_counts = df['country'].value_counts()

    # Tambahkan opsi untuk select All country
    country_labels = ['All'] + [
        f"{country} ({country_counts[country] / total_records:.1%} of Data)" # Info text (berapa persen data dari total)
        for country in country_counts.index
    ]

    # Opsi untuk men-filter country
    selected_country_label = st.sidebar.selectbox("Select Country", country_labels)
    
    # Olah pilihan user
    if selected_country_label == 'All': # Jika pilih all
        selected_country = 'All' 
    else: # Jika tidak, ambil country nya (buang label ( x% of data). makanya di split ( ambil [0])
        selected_country = selected_country_label.split(' (')[0]
    
    # Date range filter
    min_date = df['order_date'].min().date()
    max_date = df['order_date'].max().date()

    # Opsi untuk men-filter tanggal
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        help = "Select start and end date" # help text
    )
    
    # Hitung GMV per SKU untuk filter nanti
    sku_gmv = df.groupby('sku_name')['adjusted_gmv'].sum()
    # Sort dari yang terbesar
    sorted_skus = sku_gmv.sort_values(ascending=False).index.tolist()
    # Opsi untuk men-filter SKU + opsi select ALL
    sku_labels = ['All'] + sorted_skus
    selected_sku = st.sidebar.selectbox(
        "Select SKU",
        sku_labels
    )
    # Return selected country, date_range, dan selected_sku (komponen apa saja yang di filter)
    return selected_country, date_range, selected_sku


def filter_data(df, country, date_range, sku):
    """Fungsi untuk memfilter data berdasarkan 3 filter dari input user"""
    filtered_df = df.copy() # Copy data
    
    # Country filter
    if country != 'All': # Kalau tidak select All, filter country sesuai input
        filtered_df = filtered_df[filtered_df['country'] == country]
    
    # Date filter
    if len(date_range) == 2: # Kalau user sudah pilih dua tanggal start dan end
        start_date, end_date = date_range # Ambil start dan end date
        filtered_df = filtered_df[ # Filter data berdasarkan tanggal tersebut
            (filtered_df['order_date'].dt.date >= start_date) & 
            (filtered_df['order_date'].dt.date <= end_date)
        ]
    
    # SKU filter
    if sku != 'All':  # Kalau tidak select All, filter sku sesuai input
        filtered_df = filtered_df[filtered_df['sku_name'] == sku]
    
    return filtered_df

# General Overview Page ============================================================================================================================
    
def create_scorecard(df):
    """Fungsi untuk section scorecard"""

    # Buat 4 kolom / space untuk masing masing scorecard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1: # Kotak pertama
        total_quantity = df['quantity'].sum() # Total Quantity
        st.metric( # Fungsi streamlit untuk display scorecard
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
        # Exclude order yang cancelled
        total_orders = df[~df['order_id_cancelled']]['order_id'].nunique()
        st.metric(
            label="Total Orders",
            value=f"{total_orders:,}",
            help="Total number of unique orders"
        )
    
    with col4:
        # Akui customer yang tidak cancel saja
        total_customers = df[~df['order_id_cancelled']]['customer_id'].nunique()
        st.metric(
            label="Total Customers",
            value=f"{total_customers:,}",
            help="Total number of unique customers"
        )

def plot_monthly_sales(df):
    """Plot Monthly sales (GMV) (kiri) dan Customer count (kanan) dengan interactive line chart"""
    # Hitung monthly salesnya
    monthly = (
        df.groupby('order_date_month') # Group by order_date_month
        .agg(
            adjusted_gmv=('adjusted_gmv', 'sum'), # Hitung GMV
            unique_customers=('customer_id', 'nunique')  # Unique Customer
        )
        .round(0)  # round both metrics
        .reset_index()
    )

    gmv_max = monthly['adjusted_gmv'].max() * 1.05  # Untuk keperluan ylim
    cust_max = monthly['unique_customers'].max() * 1.05 # Untuk keperluan ylim
    
    # Buat dual axis dengan plotly make_subplots
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # GMV trace (kiri)
    fig.add_trace(
        go.Scatter( # Gunakan scatter plot dengan x bulan, y value gmv
            x=monthly['order_date_month'],
            y=monthly['adjusted_gmv'],
            name="GMV",
            mode="lines+markers", # Tambahkan line untuk connect the dot
            line=dict(color='darkblue', width=3), # Darkblue untuk GMV
            marker=dict(size=8),
            hovertemplate="£%{y:,.0f}" # Hover text nya ambil value gmv nya
        ),
        secondary_y=False # Kiri, jadi bukan secondary_y
    )

    # Unique customer (kanan)
    fig.add_trace(
        go.Scatter( # Plot yang sama, cuma untuk unique customer dan di kanan
            x=monthly['order_date_month'],
            y=monthly['unique_customers'],
            name="Unique Customers",
            mode="lines+markers",
            line=dict(color='darkgreen', width=3),
            marker=dict(size=8),
            hovertemplate="%{y:,}"
        ),
        secondary_y=True # Kanan, jadi pakai secondary_y
    )

    # Formatting y-axis kiri
    fig.update_yaxes(
        title_text="GMV (£)", # Label
        tickformat="£,.0f", # Rounded, no decimal
        range=[0, gmv_max], # Paksa dari 0 sampai max tadi
        secondary_y=False # Axis kiri
    )

    # Formatting y-axis kanan
    fig.update_yaxes(
        title_text="Unique Customers",
        range=[0, cust_max],
        secondary_y=True
    )

    # formatting x-axis
    fig.update_xaxes(
        title_text="Month",
        tickformat="%b %Y"
    )

    # Update legend jadi tengah bawah
    fig.update_layout(
        title="Monthly GMV & Unique Customers",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom", # Tengah
            y=1.02,
            xanchor="center", # Bawah
            x=0.5
        )
    )
    # Return fig object untuk dipakai oleh plotly di streamlit
    return fig


def plot_top_products(df):
    """Plot top 10 product berdasarkan GMV dengan Horizontal Bar Chart"""
    # Exclude yang cancelled
    df = df[~df['order_id_cancelled']]

    # Hitung sales per product, gunakan sku_name aja biar easy to interpret
    per_product_sales = (
        df.groupby('sku_name')
          .agg(gmv=('adjusted_gmv', 'sum'))
          .reset_index()
    )

    # Total GMV untuk persentase nanti
    total_gmv = per_product_sales['gmv'].sum()
    # Hitung persentase
    per_product_sales['pct'] = per_product_sales['gmv'] / total_gmv * 100

    # pick top 10
    top_products = (
        per_product_sales
        .sort_values('gmv', ascending=False) # Sort by yang paling tinggi
        .head(10) # Ambil top 10
        .reset_index(drop=True)
        .sort_values('gmv') # Sort ulang dari yang paling rendah
        # Kenapa di sort ulang, karena entah kenapa plotly bakal reverse lagi order datanya kalau horizontal bar chart
        # Makanya kita sort dari kecil ke besar, agar reversenya plotly jadi besar ke kecil
    )

    # plotly bar
    fig = px.bar(
        top_products, # data
        x='gmv',
        y='sku_name',
        orientation='h', # Horizontal
        title='Top 10 Products by GMV',
        labels={'gmv': 'GMV (£)', 'sku_name': 'Product'}, 
        text=[f"{pct:.1f}%" for pct in top_products['pct']] # Hover text nya adalah persentase penjualan
    )

    # Color dan hover formatting
    fig.update_traces(
        marker_color='darkblue',
        textposition='inside',
        hovertemplate="£%{x:,.0f}"
    )

    # axis formatting
    fig.update_layout(
        xaxis_title="GMV (£)",
        xaxis_tickformat='£,.0f',
        yaxis_title="",
        height=500 # Explicitly set height, nanti di plot lain juga set ketinggian yang sama
    )

    return fig

def plot_top_countries(df):
    """Plot top 10 country berdasarkan GMV dengan Horizontal Bar Chart"""
    # Exclude yang cancelled
    df = df[~df['order_id_cancelled']]

    # Hitung sales per country
    per_country_sales = (
        df.groupby('country')
        .agg(gmv=('adjusted_gmv', 'sum'))
        .reset_index()
    )

    # Total GMV untuk persentase nanti
    total_gmv = per_country_sales['gmv'].sum()

    # Hitung persentase
    per_country_sales['pct'] = per_country_sales['gmv'] / total_gmv * 100

    # pick top 10
    top_countries = (
        per_country_sales
        .sort_values('gmv', ascending=False)
        .head(10)
        .reset_index(drop=True)
        .sort_values('gmv')
    )

    # plotly bar
    fig = px.bar(
        top_countries,
        x='gmv',
        y='country',
        orientation='h',
        title='Top 10 Countries by GMV',
        labels={'gmv': 'GMV (£)', 'country': 'Country'},
        text=[f"{pct:.1f}%" for pct in top_countries['pct']]
    )

    # Color dan hover formatting
    fig.update_traces(
        marker_color='darkblue',
        textposition='outside',
        hovertemplate="£%{x:,.0f}"
    )

    # axis formatting
    fig.update_layout(
        xaxis_title="GMV (£)",
        xaxis_tickformat='£,.0f',
        yaxis_title="",
        height=500
    )
    
    return fig

# Customer Analysis Page ============================================================================================================================

def plot_rfm_treemap(df):
    """Plot treemap untuk RFM Analysis"""
    # Filter out cancelled orders
    df = df[~df['order_id_cancelled']]

    # Calculate RFM metrics per customer
    now = df['order_date'].max() + timedelta(days=1) # Latest date untuk recency
    rfm = (
        df.groupby('customer_id')
          .agg(
              recency=('order_date', lambda x: (now - x.max()).days), # Recency
              frequency=('order_id', 'nunique'), # Frequency
              monetary=('adjusted_gmv', 'sum') # Monetary
          )
          .reset_index()
    )

    # Rank metric tersebut
    rfm['recency_rank'] = rfm['recency'].rank(method='first')
    rfm['frequency_rank'] = rfm['frequency'].rank(method='first')
    rfm['monetary_rank'] = rfm['monetary'].rank(method='first')

    # Bagi jadi 4 quartil
    try: # In case data tidak cukup, wrap dengan try except
        # Recency makin kecil makin bagus (4)
        rfm['R'] = pd.qcut(rfm['recency_rank'], 4, labels=[4,3,2,1]).astype(int)
    except ValueError:
        rfm['R'] = 1
    
    try:
        # Frequency makin besar makin bagus (4)
        rfm['F'] = pd.qcut(rfm['frequency_rank'], 4, labels=[1,2,3,4]).astype(int)
    except ValueError:
        rfm['F'] = 1
    
    try:
        # Monetary makin besar makin bagus (4)
        rfm['M'] = pd.qcut(rfm['monetary_rank'], 4, labels=[1,2,3,4]).astype(int)
    except ValueError:
        rfm['M'] = 1

    # Kalau data terlalu sedikit, tidak bisa pakai RFM
    if rfm['recency'].nunique() <= 1 or rfm['frequency'].nunique() <= 1 or rfm['monetary'].nunique() <= 1:
        st.warning("Not enough variation in RFM data to compute segments.")
        return go.Figure()

    # RFM score nya dari sum score
    rfm['RFM_Score'] = rfm[['R','F','M']].sum(axis=1)

    # Penamaan manual untuk segment
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
            return 'Lost Low Value'
        return 'Needs Attention'
    rfm['Segment'] = rfm.apply(name_segment, axis=1)

    # Compute range dari masing masing segment
    segment_stats = (
        rfm.groupby('Segment')
           .agg(
               recency_min=('recency','min'), recency_max=('recency','max'),
               freq_min=('frequency','min'), freq_max=('frequency','max'),
               mon_min=('monetary','min'), mon_max=('monetary','max'),
               cust_count=('customer_id','count')
           )
    ).reset_index()

    # Total customer untuk persentase
    total_customers = len(rfm)
    # Persentase
    segment_stats['Pct_Customers'] = (segment_stats['cust_count'] / total_customers * 100).round(1).astype(str) + '%'

    # Merge back data segment ke rfm
    rfm = rfm.merge(segment_stats, on='Segment', how='left')

    # Buat kolom untuk label
    rfm['Recency_Range'] = rfm.apply(lambda x: f"{int(x.recency_min)}–{int(x.recency_max)} days", axis=1)
    rfm['Frequency_Range'] = rfm.apply(lambda x: f"{int(x.freq_min)}–{int(x.freq_max)} orders", axis=1)
    rfm['Monetary_Range'] = rfm.apply(lambda x: f"£{x.mon_min:,.0f}–£{x.mon_max:,.0f}", axis=1)
    rfm['Customer_Pct'] = rfm['Pct_Customers']

    # Untuk keperluan count customer
    rfm['count'] = 1
    
    # Plotly Treemap
    fig = px.treemap(
        rfm,
        path=['Segment'], # Group yang mau di gambarkan
        values='count', # Fokus ke count customer per segment
        color='RFM_Score',
        title='Customer RFM Segments',
        color_continuous_scale=[[0, 'lightgreen'], [1, 'darkgreen']],
        # Hover text
        custom_data=['Recency_Range','Frequency_Range','Monetary_Range','Customer_Pct']
    )
    # Custom hover template untuk persentase customer
    fig.update_traces(
        hovertemplate=(
            '<b>%{label}</b><br>' +
            '%{customdata[0]}<br>' +
            '%{customdata[1]}<br>' +
            '%{customdata[2]}<br>' +
            'Share of Customers: %{customdata[3]}<extra></extra>'
        )
    )

    # Layouting
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25), height=500)
    return fig


def plot_monthly_customer_area(df):
    """Area chart untuk perbandingan new customer dan total customer, monthly"""
    # Exclude order yang cancelled
    df = df[~df['order_id_cancelled']].copy()

    # Bulan pembelian
    df['month'] = df['order_date'].dt.to_period('M').dt.to_timestamp()

    # Cari first order month dari customer, untuk flagging 'new'
    first_orders = df.groupby('customer_id')['month'].min()
    df['first_month'] = df['customer_id'].map(first_orders)

    # Cocokan mana yang jadi pembelian pertama
    new_customers = df[df['month'] == df['first_month']]

    # Hitung monthly unique new customer
    new_monthly = new_customers.groupby('month')['customer_id'].nunique().reset_index(name='new_customers')

    # Hitung monthly unique customer
    total_monthly = df.groupby('month')['customer_id'].nunique().reset_index(name='total_customers')

    # Existing customers = total - new (agar tidak double perhitungan)
    # Gabungkan total bulanan unique customer dan new customer
    merged = pd.merge(total_monthly, new_monthly, on='month', how='left').fillna(0)
    # Selisih sebagai existing
    merged['existing_customers'] = merged['total_customers'] - merged['new_customers']

    # ubah jadi long form (month, tipe (new/existing), customers (count)) agar mudah di plot
    area_data = merged[['month', 'new_customers', 'existing_customers']].melt(
        id_vars='month', var_name='type', value_name='customers')

    # Plotly area chart
    fig = px.area(
        area_data,
        x='month',
        y='customers',
        color='type', # Grup berdasarkan tipe
        title='Monthly New vs Existing Customers',
        labels={'month': 'Month', 'customers': 'Customers', 'type': 'Customer Type'},
        height=500,
        color_discrete_map={ # Miripkan palette, hijau untuk customer
            'new_customers': 'lightgreen',
            'existing_customers': 'darkgreen'
        }
    )
    fig.update_traces( # Hover text
        hovertemplate='<b>%{fullData.name}</b><br>%{x|%b %Y}<br>Customers: %{y:,}<extra></extra>'
    )
    fig.update_layout(legend_title_text='Customer Type', xaxis=dict(tickformat='%b %Y'))
    return fig


def plot_order_frequency(df):
    """Plot barplot untuk order frequency per customer"""
    # Hitung ada berapa unique order per customer
    freq = df.groupby('customer_id')['order_id'].nunique()
    # Set custom range untuk bar nya
    bins = [1,2,4,7,11, np.inf]
    labels = ['1','2-3','4-6','7-10','11+']

    # Kelompokan data ke custom range tersebut
    freq_binned = pd.cut(freq, bins=bins, labels=labels, right=False)

    # Hitung count per label (ada berapa customer yang 1 kali pembelian, 2-3 kali pembelian, dst
    dist = freq_binned.value_counts().reindex(labels).reset_index()
    dist.columns = ['Order Frequency', 'Count']

    # Plot dengan barplot
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
    Plot untuk cumulative distribution dari days between purchase, untuk membantu program reminder
    """

    # Ambil unique customer id dan tanggal pembelian (untuk menghindari pembelian di hari yang sama)
    df = df[['customer_id', 'order_date_only']].drop_duplicates()
    # Sort per customer berdasarkan tanggal
    df = df.sort_values(['customer_id', 'order_date_only'])

    # Convert ke datetime
    df['order_date_only'] = pd.to_datetime(df['order_date_only'])

    # Hitung pembelian sebelumnya per customer
    df['prev_date'] = df.groupby('customer_id')['order_date_only'].shift(1)

    # Hitung selisih harinya
    df['interval'] = (df['order_date_only'] - df['prev_date']).dt.days

    # Gunakan konsep survival agar semua data dipakai dan tidak bias
    # Misal customer baru 1 kali pembelian, pembelian pertamanya 1 bulan lalu
    # Berarti days between purchase nya adalah 30 hari++ (>30 hari, tapi tidak tahu kapan / censored)

    # Get latest date
    analysis_date = df['order_date_only'].max()

    # Get pembelian terakhir (tidak punya interval)
    last_orders = df[df['interval'].isnull()]

    # Hitung intervalnya dengan tanggal sekarang
    last_orders['interval'] = (analysis_date - last_orders['order_date_only']).dt.days

    # Flag sebagai censored data
    last_orders['event_observed'] = 0  

    # Yang intervalnya tidak null, berarti observed data
    observed = df.dropna(subset=['interval']).copy()
    observed['event_observed'] = 1

    # Combine them
    survival_df = pd.concat([
        observed[['interval', 'event_observed']],
        last_orders[['interval', 'event_observed']]
    ])

    # Fit Kaplan-Meier untuk estimasi survival function
    kmf = KaplanMeierFitter()
    kmf.fit(durations=survival_df['interval'], event_observed=survival_df['event_observed'])

    # Hitung Cumulative Distribution Function (CDF)
    # CDF = 1 - Survival
    surv_df = kmf.survival_function_.reset_index()
    surv_df.columns = ['days_between', 'survival_rate']
    surv_df['cdf'] = 1 - surv_df['survival_rate']

    # Filter untuk 3 bulan pertama
    surv_df = surv_df[surv_df['days_between'] <= 90]

    # Cari median nya
    median_day = kmf.median_survival_time_

    # Plot
    fig = go.Figure()

    # Add line dengan plotly scatter
    fig.add_trace(go.Scatter(
        x=surv_df['days_between'],
        y=surv_df['cdf'],
        mode='lines',
        name='CDF',
        line=dict(color='darkgreen', width=3)
    ))

    # Horizontal 50% line (median)
    fig.add_trace(go.Scatter(
        x=[0, surv_df['days_between'].max()],
        y=[0.5, 0.5],
        mode='lines',
        name='50% Mark',
        line=dict(color='gray', dash='dash'),
        showlegend=False
    ))

    # Vertical line di median
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
        # Add annotation kalau data tidak cukup
        fig.add_annotation(
            x=surv_df['days_between'].max(),
            y=0.5,
            text="Median not reached",
            showarrow=False,
            font=dict(color="gray", size=12),
            xanchor='right'
        )

    # Layouting
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

# Product Analysis ============================================================================================================================

def plot_forecast(filtered_df, forecast_to_show):
    """ Line chart untuk hasil forecast quantity per product """
    
    # Buat kolom ds dari order_date_month, agar serupa dengan data forecast
    filtered_df['ds'] = filtered_df['order_date_month']

    # Data Actual, jumlahkan per bulan untuk quantity nya in case filter masih all product
    actuals = (
        filtered_df.groupby('ds')['quantity']
        .sum()
        .reset_index()
        .rename(columns={'quantity': 'actual_quantity'})
    )

    # Hal yang sama untuk forecast
    forecast = (
        forecast_to_show.groupby('ds')['forecast_quantity']
        .sum()
        .reset_index()
    )

    fig = go.Figure()

    # Actual line
    fig.add_trace(go.Scatter(
        x=actuals['ds'],
        y=actuals['actual_quantity'],
        mode='lines+markers',
        name='Actual Quantity'
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['forecast_quantity'],
        mode='lines+markers',
        name='Forecast Quantity',
        line=dict(dash='dash')
    ))

    fig.update_layout(
        title="Forecast 3 month ahead",
        xaxis_title="Date",
        yaxis_title="Quantity",
        template='plotly_white'
    )

    return fig
    
def plot_price_elasticity(filtered_df, selected_sku):
    """
    Plot Price Elasticity per SKU
    """
    # Kalau belum pilih SKU atau pilihnya All, tidak bisa dilanjut
    if not selected_sku or selected_sku == 'All':
        return None

    # Dapatkan SKU nya
    sku_row = filtered_df[filtered_df['sku_name'] == selected_sku]
    if sku_row.empty: # Kalau kosong, tidak bisa dilanjut
        return None
    sku_id = sku_row['sku_id'].iloc[0]  # Dapatkan SKU ID nya
    sku_name = selected_sku # Dapatkan SKU name nya
    
    # Aggregate & transform untuk sku_id per tanggal
    df_agg = (
        sku_row
        .groupby(['sku_id', 'order_date_only'])
        .agg(
            avg_price=('adjusted_price', 'mean'),
            total_qty=('quantity', 'sum')
        )
        .reset_index()
    )

    # Hanya menerima data dengan price dan quantity positif
    grp = df_agg[
        (df_agg['avg_price'] > 0) & (df_agg['total_qty'] > 0)
    ].assign(
        # Hitung log price dan log qty (agar pov nya dari persentase kenaikan)
        log_price=lambda d: np.log(d['avg_price']),
        log_qty=lambda d: np.log(d['total_qty'])
    )

    # Kalau tidak ada atau cuma ada 2 harga unik, tidak bisa buat model
    if grp.empty or grp['avg_price'].nunique() <= 2:
        return None

    # Fit model regresi
    X_log = sm.add_constant(grp['log_price'])
    y_log = grp['log_qty']
    model = sm.OLS(y_log, X_log).fit()

    # Dapatkan parameternya
    beta_0 = model.params['const']
    elasticity = model.params['log_price']
    r2 = model.rsquared

    # Buat kurvanya dari persamaan regresi
    x_vals = np.linspace(grp['avg_price'].min(), grp['avg_price'].max(), 100)
    y_vals = np.exp(beta_0) * (x_vals ** elasticity)

    # Cari tau arah elasticity
    qty_change_pct = elasticity 
    direction = "decrease" if elasticity < 0 else "increase"

    # Plot dengan scatter plot plotly
    fig = go.Figure()

    # Plot harga dan qty
    fig.add_trace(go.Scatter(
        x=grp['avg_price'],
        y=grp['total_qty'],
        mode='markers',
        marker=dict(size=8, color='steelblue', line=dict(width=1, color='black')),
        name='Observed',
        hovertemplate="Price: %{x:.2f}<br>Qty: %{y:.0f}<extra></extra>"
    ))

    # Plot regression line nya
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines',
        line=dict(color='firebrick', width=2),
        name='Elasticity Curve'
    ))

    # Layouting
    fig.update_layout(
        title=( # example: Nama SKU - elasticity : 0.13 - 1% ↑ price → x% decrease in qty | R² = 0.7
            f"{sku_name}<br>"
            f"<span style='font-size:14px'>Elasticity: {elasticity:.2f} "
            f"(1% ↑ price → {abs(qty_change_pct):.1f}% {direction} in qty) | R² = {r2:.2f}</span>"
        ),
        xaxis_title="Average Price",
        yaxis_title="Total Quantity",
        template="simple_white",
        legend=dict(
            yanchor="top", y=0.99, # top
            xanchor="right", x=0.99,  # right
            bgcolor='rgba(255,255,255,0.8)',  
            bordercolor='gray',
            borderwidth=1
        )
    )

    return fig

def plot_unique_customers_per_hour(df):
    """
    Boxplot per jam untuk unique customer
    """
    df = df.copy()
    # Buang data yang pagi (harusnya belum buka)
    df = df[df['order_hour'] > 6]  

    # Group by tanggal dan jam
    per_hour = (
        df.groupby(['order_date_only', 'order_hour'])
          .agg(unique_customers=('customer_id', 'nunique'))
          .reset_index()
    )

    # Kalau tidak ada data, infokan
    if per_hour.empty:
        return go.Figure().update_layout(
            title="No data available for selected filters",
            xaxis_title="Hour of Day",
            yaxis_title="Unique Customers per Day",
            template="simple_white",
        )

    # Hitung median unique customer per oder_hour
    medians = per_hour.groupby('order_hour')['unique_customers'].median()
    # Normalisasi untuk warna
    norm = (medians - medians.min()) / (medians.max() - medians.min())
    
    fig = go.Figure()
    # Ambil unique hour untuk per boxplot
    valid_hours = sorted(per_hour['order_hour'].unique())

    # looping per jam
    for hour in valid_hours:
        # Ambil data di jam tersebut
        hour_data = per_hour.loc[per_hour['order_hour'] == hour, 'unique_customers']
        # Kalau kosong, skip
        if hour_data.empty:
            continue

        # Get color value dari hasil normalisasi
        raw_val = norm.get(hour, 0.0) # Kalau kosong, kasih 0
        color_val = 0.0 if pd.isna(raw_val) else float(raw_val) 
        color_str = sample_colorscale("Greens", color_val)[0] # Makin tinggi median, makin hijau

        # Boxplot
        fig.add_trace(go.Box(
            y=hour_data,
            name=str(hour),
            boxpoints='outliers',
            fillcolor=color_str,
            line_color='black',
            marker_color='black'
        ))

    # Layouting
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
    Horizontal Bar chart untuk cancelled product percentage
    """
    # Hitung cancel percentage, total_order, cancelled_count per product
    per_product = (
        df[~df['sku_id_no_digit']] # Exclude SKU no digit
        .groupby('sku_name', as_index=False)
        .agg(
            cancel_percentage=('order_id_cancelled', 'mean'),
            total_order=('order_id', 'nunique'),
            cancelled_count=('order_id_cancelled', 'sum')
        )
    )

    # Ambil product yang punya banyak order, ambil top 10 yang punya cancel percentage terbesar
    top = (
        per_product[per_product['total_order'] >= 100]
        .sort_values(['cancel_percentage', 'total_order'], ascending=[False, False])
        .head(10).sort_values('cancel_percentage')
    ).reset_index(drop=True)

    # Kalau kosong, infokan
    if top.empty:
        return go.Figure().update_layout(
            title="No cancelled products available for selected filters",
            template="simple_white"
        )

    # Buat label "x of y cancelled (z%)"
    top['label'] = top.apply(
        lambda r: f"{int(r.cancelled_count)} out of {int(r.total_order)} cancelled ({r.cancel_percentage:.1%})",
        axis=1
    )

    # Plot horizontal bar chart
    fig = px.bar(
        top,
        x='cancel_percentage',
        y='sku_name',
        orientation='h',
        text='label',
        color='cancel_percentage',
        color_continuous_scale='Reds',
    )

    # Remove hover text
    fig.update_traces(
        textposition='outside',
        hoverinfo='skip',
        hovertemplate=None
    )

    # Layouting
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

# Streamlit Structure ============================================================================================================================

def main():
    # Title
    st.title("Sales Analytics Dashboard")

    # Text info
    st.markdown("Data mining on Online Retail II data")
    
    # Load data
    df = load_data()

    elas_reg_df = load_elasticity_summary()
    
    market_basket_df = load_market_basket()

    forecast_df = load_forecast()
    
    # Create filters
    selected_country, date_range, selected_sku = create_filters(df)
    
    # Apply filters
    filtered_df = filter_data(df, selected_country, date_range, selected_sku)
    
    # Show filtered data info
    st.sidebar.markdown(f"**Filtered Data:** {len(filtered_df):,} records")

    # Set the pages
    pages = ["General Overview", "Customer Analysis", "Product Analysis", "Lost Sales"]
    
    # initialize session state untuk penanda active page
    if "active_page" not in st.session_state:
        st.session_state.active_page = pages[0]

    # Buat 4 kolom sebagai pengganti fungsi tab
    cols = st.columns(len(pages))

    # Go through all the columns and pages
    for col, page in zip(cols, pages):
        # Cek mana yang active
        is_active = (st.session_state.active_page == page)
        # Jika aktif, bikin button nya jadi bold
        if col.button(f"{'**'+page+'**' if is_active else page}"):
            # Lalu tandai page yang aktif
            st.session_state.active_page = page

    # Markdown pembatas
    st.markdown("---")

    # Tab General Overview
    if st.session_state.active_page == "General Overview":
        st.header("Sales Summary")
    
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
            
    # Tab Customer Analysis
    elif st.session_state.active_page == "Customer Analysis":
        st.header("RFM Analysis")
    
        # RFM treemap
        fig_rfm = plot_rfm_treemap(filtered_df)
        st.plotly_chart(fig_rfm, use_container_width=True)
    
        st.markdown("---")
    
        # Monthly customer area
        fig_customer_area = plot_monthly_customer_area(filtered_df)
        st.plotly_chart(fig_customer_area, use_container_width=True)
    
        st.markdown("---")

        # Order Frequency and Purchase Interval
        col1, col2 = st.columns(2, gap="large")
        with col1:
            fig_order_freq = plot_order_frequency(filtered_df)
            st.plotly_chart(fig_order_freq, use_container_width=True)
    
        with col2:
            fig_purchase_cdf = plot_purchase_interval_cdf(filtered_df)
            st.plotly_chart(fig_purchase_cdf, use_container_width=True)

        st.markdown("---")

        # Hourly Customer Count
        fig_hourly = plot_unique_customers_per_hour(filtered_df)
        st.plotly_chart(fig_hourly, use_container_width=True)

    # Tab Product Analysis
    elif st.session_state.active_page == "Product Analysis":
        st.header("Quantity Forecast")
    
        # Intersect forecast_df with filtered_df (agar yang di plot sesuai apa yang di filter)
        forecast_to_show = forecast_df.merge(
            filtered_df[['sku_id', 'country']].drop_duplicates(),
            on=['sku_id', 'country']
        )
    
        st.subheader("Forecast Plot")
        fig_forecast = plot_forecast(filtered_df, forecast_to_show)
        st.plotly_chart(fig_forecast, use_container_width=True)

        # Section untuk mendownload forecast data
        st.subheader("Downloadable Data")
    
        forecast_table = forecast_to_show[['unique_id', 'country', 'sku_id', 'ds', 'forecast_quantity']] \
            .sort_values(['country', 'sku_id', 'ds']) \
            .reset_index(drop=True)
    
        st.dataframe(forecast_table, use_container_width=True)
    
        excel_data = convert_df_to_excel(forecast_table)
        
        st.download_button(
            label="Download Forecast as Excel",
            data=excel_data,
            file_name="forecast_output.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # Spacer
        st.markdown("###")

        # divider
        st.markdown("---")

        # Price Elasticity
        st.header("Price Elasticity")
        
        fig_elasticity = plot_price_elasticity(filtered_df, selected_sku)
        
        if fig_elasticity:
            st.plotly_chart(fig_elasticity)
        else:
            st.info("Please select a SKU to view its price elasticity.")
        
    
        st.subheader("Elasticity Summary")
        st.dataframe(
            elas_reg_df[['sku_name', 'elasticity', 'n_obs', 'n_price_pts', 'r2']].sort_values('r2', ascending=False).reset_index(drop=True),
            use_container_width=True
        )

        # Spacer
        st.markdown("###")
    
        # divider
        st.markdown("---")
    
        # Market Basket dataframe
        st.subheader("Market Basket Analysis")
        st.dataframe(
            market_basket_df[['antecedents', 'consequents', 'support', 'confidence']]
            .sort_values('confidence', ascending=False)
            .reset_index(drop=True),
            use_container_width=True
        )

    # Tab Lost Sales
    elif st.session_state.active_page == "Lost Sales":
        
        # 1) show the metric
        lost = filtered_df.loc[filtered_df['order_id_cancelled'], 'adjusted_gmv'].sum()
        st.metric("Lost Sales", f"£{lost:,.0f}")
        
        st.markdown("---")
        
        # 2) show the cancelled‐products bar chart
        fig_cancel = plot_cancelled_products(filtered_df)
        st.plotly_chart(fig_cancel)
        
if __name__ == "__main__":
    main()