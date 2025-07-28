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

st.set_page_config(
    page_title="Sales Analytics Dashboard", # Nama Page
    layout="wide",
    initial_sidebar_state="expanded" # Default state sidebar, on
)


def main():
    # Title
    st.title("Sales Analytics Dashboard")

    # Text info
    st.markdown("Data mining on Online Retail II data")
        
if __name__ == "__main__":
    main()

