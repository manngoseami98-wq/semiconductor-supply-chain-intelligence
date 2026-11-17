# Semiconductor Supply Chain Intelligence Dashboard

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install dependencies:
   `pip install -r requirements.txt`
4. Start the app:
   `streamlit run app.py`

## Data upload
Upload a CSV or XLSX file. Required columns:
`date, product, category, demand_qty, supply_qty, inventory_qty, lead_time_days, supplier, import_dependency_pct`

Optional:
`supplier_on_time_rate` (0–1), `defect_rate_pct`, `unit_price_inr`.

The app starts with a synthetic demo dataset. Do not interpret demo values as real market or company data.
