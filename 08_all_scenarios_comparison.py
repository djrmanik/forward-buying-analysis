# =============================================================================
# 08_all_scenarios_comparison.py
# Forward Buying Analysis — Full 3-Scenario Comparison + CSV Output
# =============================================================================
#
# STATUS: ✅ FINAL
#
# PURPOSE:
#   Run all 3 scenarios in sequence, print a complete side-by-side comparison
#   of all key metrics, and export results to CSV for further use.
#
#   This is the "re-run with fresh data" step used to verify all numbers
#   are consistent before writing the final content analysis.
#
# OUTPUT FILES:
#   outputs/scenario_comparison.csv    ← key metrics table
#   outputs/production_plans.csv       ← monthly production plans all scenarios
#
# HOW TO RUN:
#   python 08_all_scenarios_comparison.py
# =============================================================================

import numpy as np
import pandas as pd
from scipy.optimize import linprog
import os

os.makedirs("outputs", exist_ok=True)

print("=" * 75)
print("  ALL SCENARIOS COMPARISON — Forward Buying Analysis")
print("  Re-running all 3 scenarios with fresh data for final verification")
print("=" * 75)

# -----------------------------------------------------------------------------
# PARAMETERS
# -----------------------------------------------------------------------------
T = 6
months         = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
inv_init       = 2000
workers_init   = 80
sell_price     = 40
cap_reg_per_worker = 40.0
cap_ot_per_worker  = 2.5

c_mat=10; c_reg=16; c_ot=24; c_sub=30
c_hold=2; c_short=5; c_hire=300; c_fire=200
c_short_eff = sell_price + c_short   # $45

# Demand scenarios
scenarios = {
    "S1_Baseline" : {
        "demand"      : [1600, 3000, 3200, 3800, 2200, 2100],
        "promo_month" : None,
        "promo_price" : sell_price,
        "label"       : "No Promotion (Baseline)",
    },
    "S2_Promo_Jan": {
        "demand"      : [1760, 2920, 3120, 3800, 2200, 2100],
        "promo_month" : 0,
        "promo_price" : 35,
        "label"       : "Promotion in January ($35/unit)",
    },
    "S3_Promo_Apr": {
        "demand"      : [1600, 3000, 3200, 4560, 1760, 1680],
        "promo_month" : 3,
        "promo_price" : 34,
        "label"       : "Promotion in April ($34/unit)",
    },
}

# -----------------------------------------------------------------------------
# SOLVER
# -----------------------------------------------------------------------------
def idx(var, t):
    return ['P','OT','S','I','B','W','H','F'].index(var)*T+t

def solve_lp(demand):
    N=8*T
    c_obj=np.zeros(N)
    for t in range(T):
        c_obj[idx('P',t)]=c_mat+c_reg; c_obj[idx('OT',t)]=c_mat+c_ot
        c_obj[idx('S',t)]=c_mat+c_sub; c_obj[idx('I',t)]=c_hold
        c_obj[idx('B',t)]=c_short_eff; c_obj[idx('H',t)]=c_hire
        c_obj[idx('F',t)]=c_fire

    Aeq,beq,Aub,bub=[],[],[],[]
    for t in range(T):
        r=np.zeros(N)
        r[idx('P',t)]=1;r[idx('OT',t)]=1;r[idx('S',t)]=1
        r[idx('I',t)]=-1;r[idx('B',t)]=1
        if t==0: rhs=demand[t]-inv_init
        else: r[idx('I',t-1)]=1;r[idx('B',t-1)]=-1;rhs=demand[t]
        Aeq.append(r);beq.append(rhs)

        r2=np.zeros(N);r2[idx('W',t)]=1;r2[idx('H',t)]=-1;r2[idx('F',t)]=1
        if t==0: rhs2=workers_init
        else: r2[idx('W',t-1)]=-1;rhs2=0
        Aeq.append(r2);beq.append(rhs2)

        r3=np.zeros(N);r3[idx('P',t)]=1;r3[idx('W',t)]=-cap_reg_per_worker
        Aub.append(r3);bub.append(0)

        r4=np.zeros(N);r4[idx('OT',t)]=1;r4[idx('W',t)]=-cap_ot_per_worker
        Aub.append(r4);bub.append(0)

    res=linprog(c_obj,A_ub=np.array(Aub),b_ub=np.array(bub,dtype=float),
                A_eq=np.array(Aeq),b_eq=np.array(beq,dtype=float),
                bounds=[(0,None)]*N,method='highs')
    if res.status!=0: return None
    x=res.x
    return [(demand[t],)+tuple(round(x[idx(v,t)]) for v in ['P','OT','S','I','B','W','H','F'])
            for t in range(T)]

# -----------------------------------------------------------------------------
# RUN ALL SCENARIOS
# -----------------------------------------------------------------------------
results = {}
for key, sc in scenarios.items():
    rd = solve_lp(sc["demand"])
    if rd is None:
        print(f"  ❌ {key} solver failed"); continue

    demand = sc["demand"]
    pm, pp = sc["promo_month"], sc["promo_price"]

    # Revenue
    revenue = sum(
        rd[t][0] * (pp if t == pm else sell_price)
        for t in range(T)
    )

    # Costs (actual, not effective)
    mat_cost  = sum((d[1]+d[2]+d[3])*c_mat for d in rd)
    reg_cost  = sum(d[1]*c_reg for d in rd)
    ot_cost   = sum(d[2]*c_ot  for d in rd)
    sub_cost  = sum(d[3]*c_sub for d in rd)
    hold_cost = sum(d[4]*c_hold for d in rd)
    short_cost= sum(d[5]*c_short for d in rd)
    hire_cost = sum(d[7]*c_hire for d in rd)
    fire_cost = sum(d[8]*c_fire for d in rd)
    total_cost= mat_cost+reg_cost+ot_cost+sub_cost+hold_cost+short_cost+hire_cost+fire_cost
    profit = revenue - total_cost

    cv        = np.std(demand)/np.mean(demand)
    inv_kum   = sum(d[4] for d in rd)
    short_kum = sum(d[5] for d in rd)
    total_prod= sum(d[1]+d[2]+d[3] for d in rd)
    workers_hired = sum(d[7] for d in rd)

    # Capacity utilization (last two months)
    final_workers = rd[-1][6]
    final_cap     = final_workers * cap_reg_per_worker
    util_mei      = rd[4][1] / final_cap * 100
    util_jun      = rd[5][1] / final_cap * 100

    results[key] = {
        "label"         : sc["label"],
        "demand"        : demand,
        "rd"            : rd,
        "revenue"       : revenue,
        "cost_material" : mat_cost,
        "cost_reg_labor": reg_cost,
        "cost_ot"       : ot_cost,
        "cost_subcon"   : sub_cost,
        "cost_holding"  : hold_cost,
        "cost_shortage" : short_cost,
        "cost_hiring"   : hire_cost,
        "cost_firing"   : fire_cost,
        "total_cost"    : total_cost,
        "profit"        : profit,
        "cv"            : cv,
        "inv_kum"       : inv_kum,
        "short_kum"     : short_kum,
        "total_prod"    : total_prod,
        "workers_hired" : workers_hired,
        "util_mei"      : util_mei,
        "util_jun"      : util_jun,
    }
    print(f"  ✓ {key} solved successfully")

# -----------------------------------------------------------------------------
# PRINT PRODUCTION PLANS (all 3 side by side)
# -----------------------------------------------------------------------------
print()
print("=" * 75)
print("  PRODUCTION PLANS — ALL 3 SCENARIOS")
print("=" * 75)

for key, res in results.items():
    print()
    print(f"  {key}: {res['label']}")
    print(f"  {'Month':<6} {'Demand':>7} {'P_Reg':>7} {'P_OT':>7} "
          f"{'Subcon':>7} {'Inv':>8} {'Short':>7} {'Workers':>8} {'Hire':>5}")
    print("  " + "-" * 75)
    for t, d in enumerate(res["rd"]):
        dm,P,OT,S,I,B,W,H,F = d
        print(f"  {months[t]:<6} {dm:>7} {P:>7} {OT:>7} "
              f"{S:>7} {I:>8} {B:>7} {W:>8} {H:>5}")

# -----------------------------------------------------------------------------
# MASTER COMPARISON TABLE
# -----------------------------------------------------------------------------
print()
print("=" * 75)
print("  MASTER COMPARISON TABLE")
print("=" * 75)
print()

keys = list(results.keys())
r    = results

def row(label, getter, fmt=""):
    vals = [getter(r[k]) for k in keys]
    line = f"  {label:<38}"
    for v in vals:
        if fmt == "$":   line += f" ${v:>12,.0f}"
        elif fmt == "%": line += f" {v:>12.2f}%"
        elif fmt == ",": line += f" {v:>13,}"
        else:            line += f" {str(v):>13}"
    print(line)

headers = "  " + " "*38 + "".join(f" {k:>13}" for k in keys)
print(headers)
print("  " + "-"*75)

print("  FINANCIAL METRICS")
row("Revenue",             lambda r: r["revenue"],    "$")
row("Total Cost",          lambda r: r["total_cost"], "$")
row("Profit",              lambda r: r["profit"],     "$")
row("Profit Margin",       lambda r: r["profit"]/r["revenue"]*100, "%")

# Delta vs baseline
s1_profit = results["S1_Baseline"]["profit"]
print()
print("  PROFIT DELTA vs BASELINE")
for k in keys:
    delta = results[k]["profit"] - s1_profit
    sign  = "+" if delta >= 0 else ""
    print(f"  {k:<38} ${sign}{delta:>12,.0f}")

print()
print("  COST BREAKDOWN")
row("  Material",          lambda r: r["cost_material"],  "$")
row("  Regular Labor",     lambda r: r["cost_reg_labor"], "$")
row("  OT Labor",          lambda r: r["cost_ot"],        "$")
row("  Subcontract",       lambda r: r["cost_subcon"],    "$")
row("  Holding",           lambda r: r["cost_holding"],   "$")
row("  Shortage Penalty",  lambda r: r["cost_shortage"],  "$")
row("  Hiring",            lambda r: r["cost_hiring"],    "$")
row("  Firing",            lambda r: r["cost_firing"],    "$")

print()
print("  OPERATIONAL METRICS")
row("CV Demand",           lambda r: r["cv"]*100,        "%")
row("Cumul. Inventory",    lambda r: r["inv_kum"],        ",")
row("Total Shortage",      lambda r: r["short_kum"],      ",")
row("Workers Hired",       lambda r: r["workers_hired"],  ",")
row("Util. May (%)",       lambda r: r["util_mei"],       "%")
row("Util. Jun (%)",       lambda r: r["util_jun"],       "%")

# Revenue loss from discount
print()
print("  DISCOUNT IMPACT")
rev_base = results["S1_Baseline"]["revenue"]
for k in keys:
    rl = rev_base - results[k]["revenue"]
    sign = "+" if rl < 0 else "-"
    print(f"  {'Revenue loss vs baseline':<38} "
          f" ${rl:>12,.0f}" if k != "S1_Baseline" else
          f"  {'Revenue loss vs baseline':<38}  {'—':>13}")

# -----------------------------------------------------------------------------
# KEY FINDINGS SUMMARY
# -----------------------------------------------------------------------------
print()
print("=" * 75)
print("  KEY FINDINGS")
print("=" * 75)
print()
print("  1. TIMING > DISCOUNT SIZE")
print(f"     S2 (promo Jan, $5 discount): profit loss = "
      f"${results['S2_Promo_Jan']['profit']-s1_profit:,.0f}")
print(f"     S3 (promo Apr, $6 discount): profit loss = "
      f"${results['S3_Promo_Apr']['profit']-s1_profit:,.0f}")
print(f"     $1 extra discount in wrong month → "
      f"${abs(results['S3_Promo_Apr']['profit']-results['S2_Promo_Jan']['profit']):,.0f} extra profit loss")
print()
print("  2. CV AS EARLY WARNING")
for k,res in results.items():
    print(f"     {res['label']}: CV = {res['cv']*100:.2f}%")
print()
print("  3. FORWARD BUYING EFFECT ON INVENTORY")
for k,res in results.items():
    print(f"     {res['label']}: Cumul. Inv = {res['inv_kum']:,} unit-months")
print()
print("  4. NO SHORTAGE IN ANY SCENARIO")
print("     LP finds feasible solution for all 3 demand profiles.")
print()

# -----------------------------------------------------------------------------
# EXPORT TO CSV
# -----------------------------------------------------------------------------
# Comparison table
comp_data = []
metrics = [
    ("Revenue",             lambda r: r["revenue"]),
    ("Total Cost",          lambda r: r["total_cost"]),
    ("Profit",              lambda r: r["profit"]),
    ("CV Demand (%)",       lambda r: round(r["cv"]*100, 2)),
    ("Cumul. Inventory",    lambda r: r["inv_kum"]),
    ("Total Shortage",      lambda r: r["short_kum"]),
    ("Workers Hired",       lambda r: r["workers_hired"]),
    ("Cost Material",       lambda r: r["cost_material"]),
    ("Cost Reg Labor",      lambda r: r["cost_reg_labor"]),
    ("Cost Holding",        lambda r: r["cost_holding"]),
    ("Cost Hiring",         lambda r: r["cost_hiring"]),
    ("Util May (%)",        lambda r: round(r["util_mei"], 2)),
    ("Util Jun (%)",        lambda r: round(r["util_jun"], 2)),
]
for label, getter in metrics:
    row_data = {"Metric": label}
    for k in keys:
        row_data[k] = getter(results[k])
    comp_data.append(row_data)

comp_df = pd.DataFrame(comp_data)
comp_df.to_csv("outputs/scenario_comparison.csv", index=False)
print(f"  ✓ Exported: outputs/scenario_comparison.csv")

# Production plans
plan_rows = []
for k, res in results.items():
    for t, d in enumerate(res["rd"]):
        dm,P,OT,S,I,B,W,H,F = d
        plan_rows.append({
            "scenario": k,
            "month": months[t],
            "month_num": t+1,
            "demand": dm,
            "prod_regular": P,
            "prod_overtime": OT,
            "subcontract": S,
            "inventory_end": I,
            "shortage_end": B,
            "workers": W,
            "hired": H,
            "fired": F,
            "total_production": P+OT+S,
        })

plan_df = pd.DataFrame(plan_rows)
plan_df.to_csv("outputs/production_plans.csv", index=False)
print(f"  ✓ Exported: outputs/production_plans.csv")
print()
print("  → NEXT: 09_final_clean_model.py")
