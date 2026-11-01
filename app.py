import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

# Optional forecasting dependency
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False

st.set_page_config(
    page_title="StockSense DSS",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------- Styling ---------------------------------
st.markdown("""
<style>
    .stApp { background: #0b1220; color: #e5e7eb; }
    [data-testid="stSidebar"] { background: #111827; }
    [data-testid="stMetric"] {
        background: #172235; border: 1px solid #26364d;
        padding: 14px; border-radius: 12px;
    }
    div[data-testid="stDataFrame"] { border: 1px solid #26364d; border-radius: 10px; }
    h1, h2, h3 { color: #f8fafc; }
    .small-note { color: #9ca3af; font-size: 0.85rem; }
    .stButton button { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ----------------------------- Helpers ---------------------------------
def find_column(df, candidates):
    """Return the first existing column matching a candidate, ignoring case/spaces."""
    lookup = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower().replace(" ", "_")
        if key in lookup:
            return lookup[key]
    return None

def standardise_inventory(df):
    df = df.copy()
    product_col = find_column(df, ["Product_No", "Product No", "ProductNo", "Product_ID"])
    if product_col is None:
        raise ValueError("Inventory file must contain Product_No (or Product No).")
    df = df.rename(columns={product_col: "Product No"})

    mappings = {
        "Product_Description": ["Product_Description", "Product Description", "Product", "Product_Name"],
        "Store": ["Store", "Store_Name"],
        "Qty_On_Hand": ["Qty_On_Hand", "Qty On Hand", "Quantity_On_Hand", "Stock"],
        "Warehouse_Qty": ["Warehouse_Qty", "Warehouse Qty", "Warehouse_Quantity"],
        "Stock_Unit_Cost_Price": ["Stock_Unit_Cost_Price", "Unit_Cost", "Unit Cost", "Cost_Price"],
        "Supplier": ["Supplier", "Supplier_Name"],
    }
    for target, candidates in mappings.items():
        source = find_column(df, candidates)
        if source is not None and source != target:
            df = df.rename(columns={source: target})

    defaults = {
        "Product_Description": "Product " + df["Product No"].astype(str),
        "Store": "Main Supermarket",
        "Qty_On_Hand": 0,
        "Warehouse_Qty": 0,
        "Stock_Unit_Cost_Price": 0,
        "Supplier": "Not specified",
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    for col in ["Qty_On_Hand", "Warehouse_Qty", "Stock_Unit_Cost_Price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)
    df["Product No"] = df["Product No"].astype(str).str.strip()
    return df

def standardise_sales(df):
    df = df.copy()
    mappings = {
        "Product No": ["Product_No", "Product No", "ProductNo", "Product_ID"],
        "Transaction_Date": ["Transaction_Date", "Transaction Date", "Date", "Sales_Date"],
        "Net_Units": ["Net_Units", "Net Units", "Units_Sold", "Gross_Units_Sold"],
        "Sales_Amount": ["Sales_Amount", "Sales Amount", "Revenue", "Sales_Value"],
    }
    for target, candidates in mappings.items():
        source = find_column(df, candidates)
        if source is not None and source != target:
            df = df.rename(columns={source: target})
    required = ["Product No", "Transaction_Date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Sales file is missing required columns: " + ", ".join(missing))
    if "Net_Units" not in df.columns:
        if "Gross_Units_Sold" in df.columns:
            df["Net_Units"] = df["Gross_Units_Sold"]
        else:
            raise ValueError("Sales file must contain Net_Units or Gross_Units_Sold.")
    df["Product No"] = df["Product No"].astype(str).str.strip()
    df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"], errors="coerce")
    df["Net_Units"] = pd.to_numeric(df["Net_Units"], errors="coerce").fillna(0).clip(lower=0)
    if "Sales_Amount" in df.columns:
        df["Sales_Amount"] = pd.to_numeric(df["Sales_Amount"], errors="coerce").fillna(0).clip(lower=0)
    else:
        df["Sales_Amount"] = 0.0
    df = df.dropna(subset=["Transaction_Date"])
    return df

@st.cache_data
def load_data():
    inventory = pd.read_csv("Clean_Inventory.csv")
    sales = pd.read_csv("Daily_Sales.csv.gz", compression="infer")
    return standardise_inventory(inventory), standardise_sales(sales)

def money(value):
    return f"₹{value:,.2f}"

def product_label(row):
    return f'{row["Product No"]} - {row["Product_Description"]}'

def calculate_abc(sales):
    # ABC classification by total sales value; use units as a fallback if value is unavailable/zero.
    abc = sales.groupby("Product No", as_index=False).agg(
        Sales_Value=("Sales_Amount", "sum"),
        Units_Sold=("Net_Units", "sum")
    )
    if abc["Sales_Value"].sum() <= 0:
        abc["Sales_Value"] = abc["Units_Sold"]
        value_label = "Units sold (sales value unavailable)"
    else:
        value_label = "Sales value"
    abc = abc.sort_values("Sales_Value", ascending=False).reset_index(drop=True)
    total = abc["Sales_Value"].sum()
    abc["Contribution_%"] = np.where(total > 0, abc["Sales_Value"] / total * 100, 0)
    abc["Cumulative_%"] = abc["Contribution_%"].cumsum()
    # Class is based on cumulative contribution.
    abc["ABC_Category"] = np.select(
        [abc["Cumulative_%"] <= 70, abc["Cumulative_%"] <= 90],
        ["A", "B"],
        default="C"
    )
    # Ensure the product crossing the 70%/90% boundary remains in the preceding class.
    for i in range(len(abc)):
        if i > 0 and abc.loc[i, "ABC_Category"] == "B" and abc.loc[i-1, "Cumulative_%"] < 70:
            abc.loc[i, "ABC_Category"] = "A"
        if i > 0 and abc.loc[i, "ABC_Category"] == "C" and abc.loc[i-1, "Cumulative_%"] < 90:
            abc.loc[i, "ABC_Category"] = "B"
    return abc, value_label

def daily_series(sales, product_no):
    selected = sales[sales["Product No"] == str(product_no)].copy()
    if selected.empty:
        return pd.Series(dtype=float)
    series = selected.groupby("Transaction_Date")["Net_Units"].sum().sort_index()
    full_index = pd.date_range(series.index.min(), series.index.max(), freq="D")
    return series.reindex(full_index, fill_value=0).astype(float)

def forecast_series(series, horizon, model_name):
    if len(series) < 20:
        raise ValueError("At least 20 days of sales history are recommended for forecasting.")
    if model_name == "Moving Average":
        window = min(30, len(series))
        value = float(series.tail(window).mean())
        return np.repeat(max(value, 0), horizon)
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels is not installed. Add statsmodels to requirements.txt.")
    if model_name == "ARIMA":
        model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0),
                        enforce_stationarity=False, enforce_invertibility=False)
    else:
        model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7),
                        enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False)
    pred = fitted.get_forecast(steps=horizon).predicted_mean
    return np.maximum(np.asarray(pred, dtype=float), 0)

# ----------------------------- Load data --------------------------------
st.sidebar.title("📦 StockSense DSS")
st.sidebar.caption("Inventory & demand decision support")
try:
    inventory, sales = load_data()
except Exception as exc:
    st.error("Unable to load the project datasets.")
    st.exception(exc)
    st.stop()

pages = ["Home", "Analysis", "ABC Analysis", "Forecast", "What-If Analysis"]
page = st.sidebar.radio("Navigation", pages)
st.sidebar.markdown("---")
st.sidebar.caption("Academic prototype • Synthetic supermarket dataset")
st.sidebar.caption(f"Products: {inventory['Product No'].nunique():,}")
st.sidebar.caption(f"Sales records: {len(sales):,}")

# Store selector; this project is intended for one supermarket but remains robust to multiple store labels.
stores = sorted(inventory["Store"].dropna().astype(str).unique().tolist())
selected_store = st.sidebar.selectbox("Store", stores, index=0)
inv = inventory[inventory["Store"].astype(str) == selected_store].copy()
sales_store = sales.copy()
if "Store" in sales_store.columns:
    sales_store = sales_store[sales_store["Store"].astype(str) == selected_store]

# Aggregate sales and join to inventory using the standardised Product No key.
sales_summary = sales_store.groupby("Product No", as_index=False).agg(
    Total_Units_Sold=("Net_Units", "sum"),
    Total_Sales_Value=("Sales_Amount", "sum")
)
analysis = inv.merge(sales_summary, on="Product No", how="left")
analysis[["Total_Units_Sold", "Total_Sales_Value"]] = analysis[
    ["Total_Units_Sold", "Total_Sales_Value"]
].fillna(0)
analysis["Total_Available_Stock"] = analysis["Qty_On_Hand"] + analysis["Warehouse_Qty"]
analysis["Estimated_Stock_Value"] = analysis["Total_Available_Stock"] * analysis["Stock_Unit_Cost_Price"]

abc_sales, abc_value_label = calculate_abc(sales_store)
# IMPORTANT: standardise the key on BOTH dataframes before the ABC merge.
abc_sales["Product No"] = abc_sales["Product No"].astype(str).str.strip()
analysis["Product No"] = analysis["Product No"].astype(str).str.strip()
analysis = analysis.merge(abc_sales, on="Product No", how="left")
analysis["ABC_Category"] = analysis["ABC_Category"].fillna("C")

# ----------------------------- Home -------------------------------------
if page == "Home":
    st.title("StockSense: Intelligent Inventory Replenishment DSS")
    st.write("A decision support dashboard for supermarket demand analysis, stock monitoring and replenishment planning.")
    st.info("This academic prototype uses a synthetic supermarket dataset. Recommendations are decision support outputs, not automatic purchase orders.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Products", f"{len(analysis):,}")
    c2.metric("Store stock", f"{analysis['Qty_On_Hand'].sum():,.0f} units")
    c3.metric("Warehouse stock", f"{analysis['Warehouse_Qty'].sum():,.0f} units")
    c4.metric("Total units sold", f"{analysis['Total_Units_Sold'].sum():,.0f}")
    st.subheader("Inventory status")
    low_stock = analysis[analysis["Qty_On_Hand"] <= 10]
    adequate = analysis[analysis["Qty_On_Hand"] > 10]
    a, b = st.columns(2)
    a.metric("Low store-stock products", len(low_stock))
    b.metric("Products above 10 units", len(adequate))
    chart_data = analysis.nlargest(12, "Total_Units_Sold").sort_values("Total_Units_Sold")
    fig = px.bar(chart_data, x="Total_Units_Sold", y="Product_Description", orientation="h",
                 title="Top products by units sold", template="plotly_dark")
    fig.update_layout(paper_bgcolor="#0b1220", plot_bgcolor="#111827")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Products requiring attention")
    attention = analysis[analysis["Qty_On_Hand"] <= 10][
        ["Product No", "Product_Description", "Qty_On_Hand", "Warehouse_Qty", "Supplier"]
    ].sort_values("Qty_On_Hand")
    st.dataframe(attention, use_container_width=True, hide_index=True)

# ----------------------------- Analysis ---------------------------------
elif page == "Analysis":
    st.title("Inventory & Sales Analysis")
    search = st.text_input("Search product by ID or description")
    view = analysis.copy()
    if search:
        mask = view["Product No"].str.contains(search, case=False, na=False) | view["Product_Description"].astype(str).str.contains(search, case=False, na=False)
        view = view[mask]
    c1, c2, c3 = st.columns(3)
    c1.metric("Total available stock", f"{view['Total_Available_Stock'].sum():,.0f}")
    c2.metric("Total sales value", money(view["Total_Sales_Value"].sum()))
    c3.metric("Average store stock", f"{view['Qty_On_Hand'].mean():,.1f}" if len(view) else "0")
    st.dataframe(view[[
        "Product No", "Product_Description", "Qty_On_Hand", "Warehouse_Qty",
        "Total_Available_Stock", "Total_Units_Sold", "Total_Sales_Value", "ABC_Category"
    ]].sort_values("Total_Units_Sold", ascending=False), use_container_width=True, hide_index=True)
    fig = px.scatter(view, x="Total_Units_Sold", y="Total_Available_Stock",
                     size="Total_Sales_Value", color="ABC_Category",
                     hover_name="Product_Description", title="Sales versus available inventory",
                     template="plotly_dark")
    fig.update_layout(paper_bgcolor="#0b1220", plot_bgcolor="#111827")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------- ABC Analysis ----------------------------
elif page == "ABC Analysis":
    st.title("ABC Inventory Classification")
    st.write("Products are prioritised using their cumulative contribution to total sales value. A ≈ first 70%, B ≈ next 20%, C ≈ remaining 10%.")
    st.caption(f"Classification basis: {abc_value_label}. Thresholds are approximate and intended for managerial prioritisation.")
    counts = analysis["ABC_Category"].value_counts()
    x1, x2, x3 = st.columns(3)
    x1.metric("A-class products", int(counts.get("A", 0)))
    x2.metric("B-class products", int(counts.get("B", 0)))
    x3.metric("C-class products", int(counts.get("C", 0)))
    abc_view = analysis[[
        "Product No", "Product_Description", "Total_Units_Sold", "Sales_Value",
        "Contribution_%", "Cumulative_%", "ABC_Category", "Qty_On_Hand", "Warehouse_Qty"
    ]].copy()
    abc_view = abc_view.sort_values("Sales_Value", ascending=False)
    st.dataframe(abc_view, use_container_width=True, hide_index=True)
    fig = px.bar(abc_view.head(20).sort_values("Sales_Value"), x="Sales_Value",
                 y="Product_Description", color="ABC_Category", orientation="h",
                 title="Top 20 products by ABC classification basis", template="plotly_dark")
    fig.update_layout(paper_bgcolor="#0b1220", plot_bgcolor="#111827")
    st.plotly_chart(fig, use_container_width=True)
    st.download_button("Download ABC classification CSV",
                       abc_view.to_csv(index=False).encode("utf-8"),
                       "StockSense_ABC_Classification.csv", "text/csv")

# ----------------------------- Forecast ---------------------------------
elif page == "Forecast":
    st.title("Product Demand Forecast")
    st.write("Compare a Moving Average baseline with ARIMA and weekly-seasonal SARIMA forecasts.")
    product_options = analysis[["Product No", "Product_Description"]].drop_duplicates()
    product_options["label"] = product_options.apply(lambda r: product_label(r), axis=1)
    selected_label = st.selectbox("Select product", product_options["label"].tolist())
    selected_product = product_options.loc[product_options["label"] == selected_label, "Product No"].iloc[0]
    horizon = st.slider("Forecast horizon (days)", 7, 30, 14)
    model_name = st.selectbox("Forecasting method", ["Moving Average", "ARIMA", "SARIMA"])
    series = daily_series(sales_store, selected_product)
    if len(series):
        st.line_chart(series.tail(90), height=250)
        st.caption(f"Daily history from {series.index.min().date()} to {series.index.max().date()} • {len(series)} observations")
        try:
            forecast_values = forecast_series(series, horizon, model_name)
            future_dates = pd.date_range(series.index.max() + timedelta(days=1), periods=horizon, freq="D")
            forecast_df = pd.DataFrame({"Date": future_dates, "Forecast_Units": forecast_values})
            st.metric("Average forecast per day", f"{forecast_df['Forecast_Units'].mean():.2f} units")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=series.tail(90).index, y=series.tail(90).values,
                                     mode="lines", name="Historical sales"))
            fig.add_trace(go.Scatter(x=future_dates, y=forecast_values,
                                     mode="lines+markers", name=f"{model_name} forecast"))
            fig.update_layout(title=f"{model_name} demand forecast", template="plotly_dark",
                              paper_bgcolor="#0b1220", plot_bgcolor="#111827",
                              xaxis_title="Date", yaxis_title="Units sold")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)
            st.download_button("Download forecast CSV", forecast_df.to_csv(index=False).encode("utf-8"),
                               "StockSense_Forecast.csv", "text/csv")
        except Exception as exc:
            st.error("Forecast could not be generated with the selected method.")
            st.exception(exc)
    else:
        st.warning("No sales history found for this product.")

# ----------------------------- What-if ----------------------------------
elif page == "What-If Analysis":
    st.title("What-If Replenishment Analysis")
    st.write("Test how a change in expected demand and supplier lead time affects replenishment decisions.")
    product_options = analysis[["Product No", "Product_Description"]].drop_duplicates()
    product_options["label"] = product_options.apply(lambda r: product_label(r), axis=1)
    selected_label = st.selectbox("Select product for scenario", product_options["label"].tolist())
    selected_product = product_options.loc[product_options["label"] == selected_label, "Product No"].iloc[0]
    row = analysis[analysis["Product No"] == selected_product].iloc[0]
    demand_change = st.slider("Expected demand change (%)", -50, 100, 20, 5)
    lead_time = st.slider("Supplier lead time (days)", 1, 30, 7)
    history = daily_series(sales_store, selected_product)
    avg_daily_demand = float(history.tail(min(30, len(history))).mean()) if len(history) else 0.0
    adjusted_daily_demand = max(avg_daily_demand * (1 + demand_change / 100), 0)
    safety_days = 3
    target_stock = adjusted_daily_demand * (lead_time + safety_days)
    store_stock = float(row["Qty_On_Hand"])
    warehouse_stock = float(row["Warehouse_Qty"])
    transfer_qty = min(max(target_stock - store_stock, 0), warehouse_stock)
    remaining_need = max(target_stock - store_stock - transfer_qty, 0)
    purchase_qty = int(np.ceil(remaining_need))
    c1, c2, c3 = st.columns(3)
    c1.metric("Adjusted daily demand", f"{adjusted_daily_demand:.2f} units")
    c2.metric("Target stock", f"{target_stock:.1f} units")
    c3.metric("Total available stock", f"{store_stock + warehouse_stock:.0f} units")
    st.subheader("Recommended decision")
    if store_stock >= target_stock:
        st.success("Current store stock is sufficient for this scenario. No replenishment is indicated.")
    else:
        if transfer_qty > 0:
            st.info(f"Transfer approximately {int(np.ceil(transfer_qty))} units from the warehouse to the store.")
        if purchase_qty > 0:
            st.warning(f"After the warehouse transfer, consider purchasing approximately {purchase_qty} units from {row['Supplier']}.")
        else:
            st.success("Warehouse stock is sufficient to cover the estimated store requirement.")
    result = pd.DataFrame([{
        "Product No": selected_product,
        "Product": row["Product_Description"],
        "Average Daily Demand": avg_daily_demand,
        "Adjusted Daily Demand": adjusted_daily_demand,
        "Supplier Lead Time Days": lead_time,
        "Store Stock": store_stock,
        "Warehouse Stock": warehouse_stock,
        "Suggested Transfer": int(np.ceil(transfer_qty)),
        "Suggested Purchase": purchase_qty
    }])
    st.dataframe(result, use_container_width=True, hide_index=True)
    st.caption("Scenario assumptions: recent average daily sales are used as demand; safety stock is set to three days of adjusted demand. Review recommendations before acting.")
