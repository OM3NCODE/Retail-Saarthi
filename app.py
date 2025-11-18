import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
import os

# Import your pipeline class from the other file
try:
    from pipeline import KiranaPipeline
except ImportError:
    st.error("Error: `pipeline.py` not found. Please ensure it's in the same folder.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="Kirana Cash Forecast",
    page_icon="📊",
    layout="wide"
)

# --- Model Caching ---
# Cache the pipeline object so models are only loaded ONCE.
@st.cache_resource
def load_pipeline():
    try:
        pipeline = KiranaPipeline()
        return pipeline
    except FileNotFoundError as e:
        st.error(f"{e} Please run the export scripts from your notebooks.")
        return None

pipeline = load_pipeline()

# --- Helper Function to fetch data from DB ---
@st.cache_data(ttl=3600) # Cache DB result for 1 hour
def get_yesterday_cash():
    if not os.path.exists('kirana_demo.db'):
        return 0.0, "DB file not found. Run `setup_db.py`."
        
    try:
        conn = sqlite3.connect('kirana_demo.db')
        cursor = conn.cursor()
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        
        query = f"SELECT SUM(change_given) FROM transactions WHERE payment_method = 'cash' AND date(timestamp) = '{yesterday}'"
        
        cursor.execute(query)
        result = cursor.fetchone()[0]
        conn.close()
        
        return result if result else 0.0, None
    except Exception as e:
        return 0.0, f"DB Error: {e}"

# --- Sidebar (Inputs) ---
st.sidebar.title("Forecast Controls")

# 1. Date Input
forecast_date = st.sidebar.date_input(
    "Select Forecast Date",
    datetime.now() + timedelta(days=1) # Default to tomorrow
)

# 2. Lag Data Input (Yesterday's Cash)
# We pre-fill this by querying the database automatically.
db_cash, db_error = get_yesterday_cash()
if db_error:
    st.sidebar.error(db_error)

yesterday_cash = st.sidebar.number_input(
    "Yesterday's Total Change Given (₹)",
    min_value=0.0,
    value=round(db_cash, 2),
    step=50.0,
    help="This is required for the time-series model (lag1, roll7, etc.)"
)

# 3. Run Button
run_button = st.sidebar.button("Run Forecast", type="primary")


# --- Main Page (Outputs) ---
st.title("📊 Kirana Cash Flow Forecast")

if not pipeline:
    st.warning("Models could not be loaded. Please check model files and restart.")
elif run_button:
    with st.spinner("Running predictions..."):
        # Run the full pipeline
        res = pipeline.run_prediction(
            date_str=forecast_date.strftime("%Y-%m-%d"),
            yesterday_total_cash=yesterday_cash
        )

    st.header(f"Report for: {forecast_date.strftime('%A, %B %d, %Y')}")
    
    # --- Display Key Metrics ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Demand", f"₹{res['Total_Change']}")
    col2.metric("Recommended Float", f"₹{res['Inventory_Value']}", 
               delta=f"₹{res['Safety_Buffer']} buffer", delta_color="normal")
    col3.metric("Predicted Rush Hour", res['Spike_Hour'])
    
    st.markdown("---")
    
    # --- Display Inventory Checklist ---
    st.header("🛒 Recommended Inventory Checklist")
    
    notes = {k: v for k, v in res['Inventory'].items() if k > 10 and v > 0}
    coins = {k: v for k, v in res['Inventory'].items() if k <= 10 and v > 0}

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Notes (for Register)")
        if not notes:
            st.info("No large notes needed.")
        else:
            note_df = pd.DataFrame(notes.items(), columns=["Denomination (₹)", "Count"])
            st.dataframe(note_df.sort_values("Denomination (₹)", ascending=False), use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Coins / Small Notes (for Cash Box)")
        if not coins:
            st.info("No coins or small change needed.")
        else:
            coin_df = pd.DataFrame(coins.items(), columns=["Denomination (₹)", "Count"])
            st.dataframe(coin_df.sort_values("Denomination (₹)", ascending=False), use_container_width=True, hide_index=True)

else:
    st.info("Please select a date and click 'Run Forecast' to see the results.")