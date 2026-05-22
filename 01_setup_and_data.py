# =============================================================================
# 01_setup_and_data.py
# Forward Buying Analysis — Environment Setup & Data Loading
# =============================================================================
#
# PURPOSE:
#   Verify that all required libraries are installed, load input data from
#   CSV files, and print a summary of the parameters and demand scenarios
#   that will be used throughout the analysis.
#
# LIBRARIES INSTALLED (run once before starting):
#   pip install scipy numpy pandas
#
# HOW TO RUN:
#   python 01_setup_and_data.py
# =============================================================================

import sys

# -----------------------------------------------------------------------------
# STEP 1: Check library availability
# -----------------------------------------------------------------------------
print("=" * 60)
print("  ENVIRONMENT CHECK")
print("=" * 60)

required_libs = {
    "numpy": "Matrix operations for LP constraint formulation",
    "scipy": "Linear Programming solver (linprog / HiGHS)",
    "pandas": "Data manipulation and output formatting"
}

all_ok = True
for lib, purpose in required_libs.items():
    try:
        __import__(lib)
        print(f"  ✓ {lib:<10} — {purpose}")
    except ImportError:
        print(f"  ✗ {lib:<10} — NOT FOUND. Run: pip install {lib}")
        all_ok = False

if not all_ok:
    print("\n  Some libraries are missing. Please install them before continuing.")
    sys.exit(1)
else:
    print("\n  All libraries available. Ready to proceed.\n")

# -----------------------------------------------------------------------------
# STEP 2: Import libraries
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.optimize import linprog

print(f"  numpy version:  {np.__version__}")
import scipy
print(f"  scipy version:  {scipy.__version__}")
print(f"  pandas version: {pd.__version__}")
print()

# -----------------------------------------------------------------------------
# STEP 3: Load parameters from CSV
# -----------------------------------------------------------------------------
print("=" * 60)
print("  LOADING PARAMETERS")
print("=" * 60)

params_df = pd.read_csv("data/parameters.csv")
params = dict(zip(params_df["parameter"], params_df["value"]))

# Print key parameters
key_params = [
    ("sell_price",                "Selling Price"),
    ("initial_inventory",         "Initial Inventory"),
    ("initial_workers",           "Initial Workforce"),
    ("working_days_per_month",    "Working Days / Month"),
    ("regular_hours_per_day",     "Regular Hours / Day"),
    ("max_overtime_per_worker",   "Max OT Hours / Worker / Month"),
    ("labor_hours_per_unit",      "Labor Hours / Unit"),
    ("capacity_regular_per_worker","Regular Capacity / Worker"),
    ("capacity_overtime_per_worker","OT Capacity / Worker"),
]

print()
for key, label in key_params:
    print(f"  {label:<35} : {params[key]}")

print()
print("  COST PARAMETERS:")
cost_params = [
    ("cost_material",       "Material Cost"),
    ("cost_regular_labor",  "Regular Labor Cost / Unit"),
    ("cost_overtime_labor", "Overtime Labor Cost / Unit"),
    ("cost_subcontract",    "Subcontract Cost / Unit"),
    ("cost_holding",        "Holding Cost / Unit / Month"),
    ("cost_shortage",       "Shortage Penalty / Unit / Month"),
    ("cost_hiring",         "Hiring Cost / Worker"),
    ("cost_firing",         "Firing Cost / Worker"),
    ("effective_shortage_cost", "Effective Shortage Cost (incl. lost rev.)"),
]

for key, label in cost_params:
    print(f"  {label:<40} : ${params[key]:.0f}")

# -----------------------------------------------------------------------------
# STEP 4: Load demand scenarios from CSV
# -----------------------------------------------------------------------------
print()
print("=" * 60)
print("  LOADING DEMAND SCENARIOS")
print("=" * 60)

demand_df = pd.read_csv("data/demand_scenarios.csv")
print()
print(demand_df.to_string(index=False))
print()

# Summary stats
for col in ["scenario_1_baseline", "scenario_2_promo_jan", "scenario_3_promo_apr"]:
    data = demand_df[col].values
    cv = np.std(data) / np.mean(data)
    print(f"  {col}:")
    print(f"    Total demand : {sum(data):,} units")
    print(f"    Average      : {np.mean(data):,.1f} units/month")
    print(f"    Std Dev      : {np.std(data):,.1f} units")
    print(f"    CV           : {cv*100:.2f}%")
    print()

# -----------------------------------------------------------------------------
# STEP 5: Derived capacity calculations
# -----------------------------------------------------------------------------
print("=" * 60)
print("  DERIVED CAPACITY CALCULATIONS")
print("=" * 60)

reg_hrs_day    = int(params["regular_hours_per_day"])
work_days      = int(params["working_days_per_month"])
max_ot         = int(params["max_overtime_per_worker"])
labor_per_unit = int(params["labor_hours_per_unit"])
init_workers   = int(params["initial_workers"])

cap_reg = (reg_hrs_day * work_days) / labor_per_unit
cap_ot  = max_ot / labor_per_unit

print()
print(f"  Regular capacity / worker / month:")
print(f"    {reg_hrs_day} hrs/day × {work_days} days ÷ {labor_per_unit} hrs/unit = {cap_reg:.0f} units")
print()
print(f"  OT capacity / worker / month:")
print(f"    {max_ot} hrs ÷ {labor_per_unit} hrs/unit = {cap_ot:.1f} units")
print()
print(f"  Max regular production with {init_workers} workers:")
print(f"    {init_workers} × {cap_reg:.0f} = {init_workers * cap_reg:,.0f} units/month")
print()
print(f"  Max OT production with {init_workers} workers:")
print(f"    {init_workers} × {cap_ot:.1f} = {init_workers * cap_ot:,.0f} units/month")
print()
print(f"  Demand gap in April (S1 baseline): 3,800 - {init_workers * cap_reg:,.0f} = "
      f"{3800 - init_workers * cap_reg:,.0f} units")
print("  → Must be covered by inventory buffer built in Feb & Mar")
print()
print("=" * 60)
print("  Setup complete. Proceed to 02_lp_iteration1_baseline.py")
print("=" * 60)
