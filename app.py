import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

st.set_page_config(page_title="Semiconductor Supply Chain Intelligence", page_icon="🔗", layout="wide")

# -----------------------------
# Demo data and data preparation
# -----------------------------
@st.cache_data
def make_demo_data():
    rng = np.random.default_rng(19)
    products = [
        ("JEDEC IC Tray", "Semiconductor Packaging", 4200, 34, 82, 75),
        ("Waffle Tray", "Semiconductor Packaging", 3100, 28, 78, 68),
        ("Tray Cover & Clip", "Semiconductor Packaging", 2600, 22, 88, 45),
        ("Wafer Carrier Box", "Wafer Handling", 850, 52, 91, 105),
        ("Test Socket", "Test & Burn-in", 1200, 63, 72, 95),
        ("Quartz Process Boat", "Solar & Process Consumables", 540, 48, 66, 90),
        ("Graphite Boat", "Solar & Process Consumables", 680, 43, 70, 82),
        ("PVDF Wafer Carrier", "Solar Wafer Handling", 760, 39, 74, 88),
    ]
    suppliers = ["Taiwan Supplier A", "Japan Supplier B", "China Supplier C", "India Supplier D", "Korea Supplier E"]
    rows = []
    dates = pd.date_range(end=pd.Timestamp.today().normalize().replace(day=1), periods=30, freq="MS")
    for p_idx, (product, category, base, lt, imp, price) in enumerate(products):
        for i, date in enumerate(dates):
            season = 1 + 0.10 * np.sin(i * 2 * np.pi / 12)
            trend = 1 + 0.012 * i
            demand = max(20, int(base * season * trend * rng.normal(1, 0.10)))
            supplier = suppliers[(p_idx + i // 7) % len(suppliers)]
            lead = max(5, int(rng.normal(lt + (imp - 70) * 0.16, 7)))
            supply = max(0, int(demand * rng.normal(0.96 if p_idx in [0,1,4] else 1.02, 0.16)))
            inventory = max(0, int(demand * rng.uniform(0.35, 1.35)))
            on_time = float(np.clip(rng.normal(0.91 - (lead / 500), 0.08), 0.55, 0.99))
            defects = float(np.clip(rng.normal(1.8 + (100-on_time)*0.08, 0.7), 0.1, 6.0))
            rows.append({
                "date": date, "product": product, "category": category,
                "demand_qty": demand, "supply_qty": supply, "inventory_qty": inventory,
                "lead_time_days": lead, "supplier": supplier,
                "import_dependency_pct": float(np.clip(rng.normal(imp, 5), 20, 100)),
                "supplier_on_time_rate": on_time, "defect_rate_pct": defects,
                "unit_price_inr": price * 1000
            })
    return pd.DataFrame(rows)

REQUIRED = ["date", "product", "category", "demand_qty", "supply_qty", "inventory_qty",
            "lead_time_days", "supplier", "import_dependency_pct"]

def normalize_data(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    numeric = ["demand_qty", "supply_qty", "inventory_qty", "lead_time_days", "import_dependency_pct"]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c, default in [("supplier_on_time_rate", 0.90), ("defect_rate_pct", 2.0), ("unit_price_inr", 0)]:
        if c not in df.columns:
            df[c] = default
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(default)
    df = df.dropna(subset=["date", "product", "category", "supplier"] + numeric)
    df["supplier_on_time_rate"] = df["supplier_on_time_rate"].clip(0, 1)
    df["import_dependency_pct"] = df["import_dependency_pct"].clip(0, 100)
    return df.sort_values(["product", "date"]).reset_index(drop=True)

def build_features(df):
    x = df.copy().sort_values(["product", "date"])
    x["month_num"] = x["date"].dt.month
    x["year_num"] = x["date"].dt.year
    x["demand_lag_1"] = x.groupby("product")["demand_qty"].shift(1)
    x["demand_lag_2"] = x.groupby("product")["demand_qty"].shift(2)
    x["demand_lag_3"] = x.groupby("product")["demand_qty"].shift(3)
    x["demand_ma_3"] = x.groupby("product")["demand_qty"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    return x

def train_demand_model(df):
    feat = build_features(df).dropna(subset=["demand_lag_1", "demand_lag_2", "demand_lag_3"])
    if len(feat) < 12:
        return None, None, "At least 12 usable historical rows are needed for demand forecasting."
    enc = LabelEncoder()
    feat["product_code"] = enc.fit_transform(feat["product"].astype(str))
    cols = ["product_code", "month_num", "year_num", "demand_lag_1", "demand_lag_2", "demand_lag_3", "demand_ma_3"]
    X, y = feat[cols], feat["demand_qty"]
    split = max(1, int(len(feat) * 0.8))
    model = RandomForestRegressor(n_estimators=250, max_depth=8, min_samples_leaf=2, random_state=42)
    model.fit(X.iloc[:split], y.iloc[:split])
    pred = model.predict(X.iloc[split:]) if split < len(feat) else np.array([])
    mae = mean_absolute_error(y.iloc[split:], pred) if len(pred) else None
    return (model, enc, cols), mae, None

def forecast_product(df, model_bundle, product, periods=3):
    model, enc, cols = model_bundle
    hist = df[df["product"] == product].sort_values("date")
    if hist.empty:
        return pd.DataFrame()
    history = list(hist["demand_qty"].astype(float).values)
    last_date = hist["date"].max()
    rows = []
    for step in range(1, periods + 1):
        target_date = last_date + pd.offsets.MonthBegin(step)
        code = int(enc.transform([product])[0])
        row = pd.DataFrame([{
            "product_code": code, "month_num": target_date.month, "year_num": target_date.year,
            "demand_lag_1": history[-1], "demand_lag_2": history[-2] if len(history)>1 else history[-1],
            "demand_lag_3": history[-3] if len(history)>2 else history[-1],
            "demand_ma_3": float(np.mean(history[-3:]))
        }])[cols]
        value = max(0, float(model.predict(row)[0]))
        history.append(value)
        rows.append({"date": target_date, "forecast_demand": value})
    return pd.DataFrame(rows)

def risk_table(df):
    latest = df.sort_values("date").groupby("product").tail(1).copy()
    latest["coverage_ratio"] = (latest["inventory_qty"] + latest["supply_qty"]) / latest["demand_qty"].replace(0, np.nan)
    latest["gap_qty"] = (latest["demand_qty"] - latest["inventory_qty"] - latest["supply_qty"]).clip(lower=0)
    latest["risk_score"] = (
        0.40 * (1 - latest["coverage_ratio"].clip(0, 1)) +
        0.25 * latest["import_dependency_pct"] / 100 +
        0.20 * (latest["lead_time_days"] / max(1, latest["lead_time_days"].max())) +
        0.15 * (1 - latest["supplier_on_time_rate"])
    ) * 100
    latest["risk_level"] = pd.cut(latest["risk_score"], [-1, 35, 60, 101], labels=["Low", "Medium", "High"])
    return latest.sort_values("risk_score", ascending=False)

def train_stockout_model(df):
    d = df.copy()
    d["next_period_gap"] = (d["demand_qty"] - d["supply_qty"] - d["inventory_qty"]).clip(lower=0)
    d["stockout"] = (d["next_period_gap"] > 0).astype(int)
    d["product_code"] = LabelEncoder().fit_transform(d["product"].astype(str))
    features = ["product_code", "demand_qty", "supply_qty", "inventory_qty", "lead_time_days",
                "import_dependency_pct", "supplier_on_time_rate", "defect_rate_pct"]
    if len(d) < 20 or d["stockout"].nunique() < 2:
        return None, None, None
    X, y = d[features], d["stockout"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=220, max_depth=7, min_samples_leaf=2, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, (probs >= 0.5).astype(int))
    auc = roc_auc_score(y_test, probs) if y_test.nunique() > 1 else None
    return (model, features), acc, auc

# -----------------------------
# Sidebar and data source
# -----------------------------
st.sidebar.title("🔗 Supply Chain Intelligence")
st.sidebar.caption("Semiconductor & advanced manufacturing")
uploaded = st.sidebar.file_uploader("Upload supply chain CSV or Excel", type=["csv", "xlsx"])
if uploaded:
    try:
        raw = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
        data = normalize_data(raw)
        st.sidebar.success(f"Loaded {len(data):,} records")
    except Exception as e:
        st.sidebar.error(str(e))
        data = make_demo_data()
        st.sidebar.info("Using demo dataset until upload is corrected.")
else:
    data = make_demo_data()
    st.sidebar.info("Demo dataset active")
st.sidebar.markdown("---")
st.sidebar.caption("Demo figures are synthetic and are not actual market or company data.")

min_date, max_date = data["date"].min().date(), data["date"].max().date()
date_range = st.sidebar.date_input("Analysis period", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(date_range, tuple) and len(date_range) == 2:
    view = data[(data["date"].dt.date >= date_range[0]) & (data["date"].dt.date <= date_range[1])].copy()
else:
    view = data.copy()
if view.empty:
    view = data.copy()

st.title("Semiconductor Supply Chain Intelligence")
st.markdown("**Find supply gaps. Forecast demand. Anticipate shortages. Prioritise opportunities.**")
st.caption("Decision-support prototype | Models train on the selected dataset | Validate predictions before procurement or investment decisions.")

tabs = st.tabs(["Executive Overview", "Gap Analysis", "Demand Forecast", "Shortage & Supplier Risk", "Delva Opportunity Map", "Data & Model Notes"])

# Shared calculations
latest = risk_table(view)
total_demand = float(view["demand_qty"].sum())
total_supply = float(view["supply_qty"].sum())
gap_total = float((view["demand_qty"] - view["supply_qty"]).clip(lower=0).sum())
avg_import = float(view.groupby("product")["import_dependency_pct"].mean().mean())
avg_lead = float(view["lead_time_days"].mean())
coverage = (view["supply_qty"].sum() + view["inventory_qty"].sum()) / max(1, view["demand_qty"].sum())

with tabs[0]:
    st.subheader("Executive overview")
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Products tracked", f"{view['product'].nunique()}")
    k2.metric("Demand volume", f"{total_demand:,.0f} units")
    k3.metric("Unfilled demand", f"{gap_total:,.0f} units")
    k4.metric("Average import dependency", f"{avg_import:.1f}%")
    k5.metric("Average lead time", f"{avg_lead:.0f} days")
    left,right = st.columns([1.3,1])
    with left:
        monthly = view.groupby(pd.Grouper(key="date", freq="MS"))[["demand_qty","supply_qty"]].sum().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly["date"], y=monthly["demand_qty"], mode="lines+markers", name="Demand"))
        fig.add_trace(go.Scatter(x=monthly["date"], y=monthly["supply_qty"], mode="lines+markers", name="Supply"))
        fig.update_layout(title="Monthly demand vs supply", xaxis_title="", yaxis_title="Units", legend_title="")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.bar(latest, x="risk_score", y="product", color="risk_level", orientation="h",
                     title="Product-level supply risk", color_discrete_map={"Low":"#2ca02c","Medium":"#ffbf00","High":"#d62728"})
        fig.update_layout(yaxis={"categoryorder":"total ascending"}, xaxis_title="Indicative risk score (0–100)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### Highest-priority gaps")
    st.dataframe(latest[["product","category","gap_qty","coverage_ratio","import_dependency_pct","lead_time_days","risk_score","risk_level"]]
                 .rename(columns={"gap_qty":"Indicative gap (units)","coverage_ratio":"Supply + inventory / demand","risk_score":"Risk score"}),
                 use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("Supply chain gap analysis")
    st.write("A gap is estimated where current supply plus available inventory does not cover observed demand. Import dependency and lead time add strategic risk context.")
    cat = st.selectbox("Filter category", ["All"] + sorted(view["category"].unique().tolist()), key="gap_cat")
    gap_view = latest if cat == "All" else latest[latest["category"] == cat]
    c1,c2,c3 = st.columns(3)
    c1.metric("Products with a gap", int((gap_view["gap_qty"] > 0).sum()))
    c2.metric("High-risk products", int((gap_view["risk_level"].astype(str) == "High").sum()))
    c3.metric("Mean import dependency", f"{gap_view['import_dependency_pct'].mean():.1f}%")
    fig = px.scatter(gap_view, x="lead_time_days", y="import_dependency_pct", size="gap_qty", color="risk_level",
                     hover_name="product", title="Import dependency vs lead time (bubble size = gap quantity)",
                     labels={"lead_time_days":"Lead time (days)","import_dependency_pct":"Import dependency (%)"})
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(gap_view[["product","supplier","demand_qty","supply_qty","inventory_qty","gap_qty","lead_time_days","import_dependency_pct","supplier_on_time_rate","defect_rate_pct"]],
                 use_container_width=True, hide_index=True)
    st.download_button("Download gap analysis CSV", gap_view.to_csv(index=False).encode("utf-8"), "supply_chain_gap_analysis.csv", "text/csv")

with tabs[2]:
    st.subheader("Machine-learning demand forecast")
    st.write("Random Forest regression uses product identity, calendar month, and 1–3 month demand lags plus a rolling average. It is a practical baseline, not a substitute for a validated time-series forecasting programme.")
    bundle, mae, err = train_demand_model(view)
    if err:
        st.warning(err)
    else:
        products = sorted(view["product"].unique().tolist())
        selected = st.selectbox("Product to forecast", products)
        horizon = st.slider("Forecast horizon (months)", 1, 6, 3)
        forecast = forecast_product(view, bundle, selected, horizon)
        hist = view[view["product"] == selected].sort_values("date").tail(18)[["date","demand_qty"]].rename(columns={"demand_qty":"value"})
        hist["series"] = "Historical demand"
        fc = forecast.rename(columns={"forecast_demand":"value"})
        fc["series"] = "Model forecast"
        plot_df = pd.concat([hist,fc], ignore_index=True)
        fig = px.line(plot_df, x="date", y="value", color="series", markers=True, title=f"Demand history and {horizon}-month forecast: {selected}")
        fig.update_layout(xaxis_title="", yaxis_title="Units")
        st.plotly_chart(fig, use_container_width=True)
        a,b,c = st.columns(3)
        a.metric("Next-month forecast", f"{forecast.iloc[0]['forecast_demand']:,.0f} units")
        b.metric("Forecast horizon", f"{horizon} months")
        c.metric("Holdout MAE", f"{mae:,.0f} units" if mae is not None else "Not available")
        st.dataframe(forecast, use_container_width=True, hide_index=True)
        st.download_button("Download forecast CSV", forecast.to_csv(index=False).encode("utf-8"), "demand_forecast.csv", "text/csv")

with tabs[3]:
    st.subheader("Shortage and supplier risk")
    stock_bundle, acc, auc = train_stockout_model(view)
    if stock_bundle:
        model, features = stock_bundle
        latest_rows = view.sort_values("date").groupby("product").tail(1).copy()
        enc = LabelEncoder()
        enc.fit(view["product"].astype(str))
        latest_rows["product_code"] = enc.transform(latest_rows["product"].astype(str))
        latest_rows["shortage_probability"] = model.predict_proba(latest_rows[features])[:,1]
        latest_rows["shortage_band"] = pd.cut(latest_rows["shortage_probability"], [-0.01,0.35,0.65,1.01], labels=["Low","Medium","High"])
        x,y,z = st.columns(3)
        x.metric("Model accuracy (holdout)", f"{acc:.1%}")
        y.metric("ROC-AUC", f"{auc:.2f}" if auc is not None else "N/A")
        z.metric("Products at high predicted risk", int((latest_rows["shortage_band"].astype(str)=="High").sum()))
        fig = px.bar(latest_rows.sort_values("shortage_probability"), x="shortage_probability", y="product", color="shortage_band",
                     orientation="h", title="Predicted shortage probability by product",
                     labels={"shortage_probability":"Predicted probability"})
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(latest_rows[["product","supplier","shortage_probability","shortage_band","lead_time_days","import_dependency_pct","inventory_qty","demand_qty"]],
                     use_container_width=True, hide_index=True)
    else:
        st.warning("Not enough class variation to train the shortage model on this dataset. Add more history and both shortage/non-shortage examples.")
    st.markdown("#### Supplier scorecard")
    supplier = view.groupby("supplier").agg(
        records=("product","count"), avg_lead_time=("lead_time_days","mean"),
        on_time_rate=("supplier_on_time_rate","mean"), defect_rate=("defect_rate_pct","mean"),
        import_dependency=("import_dependency_pct","mean")
    ).reset_index()
    supplier["risk_index"] = (0.35*(1-supplier["on_time_rate"])*100 + 0.25*supplier["avg_lead_time"]/max(1,supplier["avg_lead_time"].max())*100 +
                              0.20*supplier["defect_rate"]/max(1,supplier["defect_rate"].max())*100 + 0.20*supplier["import_dependency"])
    supplier["risk_band"] = pd.cut(supplier["risk_index"], [-1,35,60,101], labels=["Low","Medium","High"])
    st.dataframe(supplier.sort_values("risk_index", ascending=False), use_container_width=True, hide_index=True)

with tabs[4]:
    st.subheader("Delva SemiPack opportunity map")
    st.write("This view translates observed supply-chain gaps into potential commercial entry points. It is a screening tool; validate demand, qualification requirements, customer intent and competitor capacity before committing.")
    opp = latest.copy()
    def opportunity(row):
        p = str(row["product"]).lower()
        if "jedec" in p or "waffle" in p or "tray" in p:
            return "Core semiconductor packaging consumables"
        if "carrier" in p or "wafer" in p:
            return "Wafer handling and carrier solutions"
        if "socket" in p:
            return "Test and burn-in ecosystem"
        if "quartz" in p or "graphite" in p:
            return "Solar / process consumables"
        return "Supplier localisation / distribution"
    opp["potential_entry_point"] = opp.apply(opportunity, axis=1)
    opp["priority"] = np.where((opp["risk_score"]>=60) | (opp["gap_qty"]>0), "Investigate", "Monitor")
    opp = opp.sort_values(["priority","risk_score"], ascending=[True,False])
    fig = px.scatter(opp, x="import_dependency_pct", y="gap_qty", size="risk_score", color="potential_entry_point",
                     hover_name="product", title="Potential entry points: import dependency and unmet demand",
                     labels={"import_dependency_pct":"Import dependency (%)","gap_qty":"Indicative gap (units)"})
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(opp[["product","category","potential_entry_point","gap_qty","import_dependency_pct","lead_time_days","risk_score","priority"]],
                 use_container_width=True, hide_index=True)
    st.info("Delva focus areas to validate: JEDEC IC trays, waffle trays, tray accessories, wafer carriers, and selected solar/process consumables. This mapping reflects product-category logic, not confirmed customer demand or a market-size estimate.")

with tabs[5]:
    st.subheader("Data requirements and model notes")
    st.markdown("""
**Required columns** (column names are case-insensitive; spaces are accepted):
- `date` — monthly or transaction date
- `product`, `category`, `supplier`
- `demand_qty`, `supply_qty`, `inventory_qty`
- `lead_time_days`, `import_dependency_pct`

**Optional columns**: `supplier_on_time_rate` (0–1), `defect_rate_pct`, `unit_price_inr`.

**Models included**
1. Random Forest Regressor — multi-step product demand forecast using lagged demand and calendar features.
2. Random Forest Classifier — indicative shortage probability using demand, supply, inventory, lead time and supplier indicators.
3. Transparent weighted risk score — combines supply coverage, import dependency, lead time and supplier on-time performance.

**Important limitations**
- Demo data is synthetic. It is only to demonstrate the workflow.
- The shortage label in this prototype is a simplified rule based on demand exceeding supply plus inventory; it is not a verified future stockout event.
- Random train/test splitting and the small demo dataset can overstate or understate model performance. For production, use chronological backtesting, more history, SKU-specific service levels, confirmed purchase orders, safety stock, MOQ, capacity, logistics and customs data.
- Import dependency should be based on reliable product-level sourcing data, not a generic industry estimate.
- Predictions are decision support, not procurement instructions.
""")
    st.markdown("#### Preview current dataset")
    st.dataframe(view.head(100), use_container_width=True, hide_index=True)
    st.download_button("Download current dataset", view.to_csv(index=False).encode("utf-8"), "supply_chain_data.csv", "text/csv")

st.markdown("---")
st.caption("Prototype for supply-chain analysis | Built with Streamlit, Pandas, Plotly and scikit-learn")
