# =============================================================================
# 09_final_clean_model.py
# Forward Buying Analysis — Production-Ready Clean Version
# =============================================================================
#
# PURPOSE:
#   This is the clean, refactored, production-ready version of the entire
#   analysis. All development iterations (02-08) are consolidated here
#   into well-structured, documented, reusable code.
#
# WHAT'S DIFFERENT FROM EARLIER SCRIPTS:
#   - Parameters loaded from CSV (not hardcoded)
#   - Solver wrapped in a clean function with type hints
#   - Revenue calculation handles multiple price tiers
#   - All outputs exported cleanly
#   - Designed for reuse: change data files, get new results
#
# HOW TO RUN:
#   python 09_final_clean_model.py
#
# HOW TO ADAPT:
#   1. Edit data/demand_scenarios.csv to change demand profiles
#   2. Edit data/parameters.csv to change cost/capacity parameters
#   3. Run this script — all outputs update automatically
# =============================================================================

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from typing import Optional
import os

os.makedirs("outputs", exist_ok=True)

# =============================================================================
# DATA LOADING
# =============================================================================

def load_parameters(filepath: str = "data/parameters.csv") -> dict:
    """Load model parameters from CSV. Returns dict of {parameter: value}."""
    df = pd.read_csv(filepath)
    return dict(zip(df["parameter"], df["value"]))

def load_demand_scenarios(filepath: str = "data/demand_scenarios.csv") -> dict:
    """Load demand scenarios from CSV. Returns dict of {col_name: [values]}."""
    df = pd.read_csv(filepath)
    return {
        col: df[col].tolist()
        for col in df.columns
        if col.startswith("scenario")
    }

# =============================================================================
# LP MODEL
# =============================================================================

class AggregateProductionLP:
    """
    Linear Programming model for Aggregate Production Planning.

    Solves the cost-minimization problem:
        minimize: total operational cost over T periods

    Subject to:
        - Inventory balance constraints
        - Worker balance constraints
        - Regular production capacity constraints
        - Overtime production capacity constraints
        - Non-negativity constraints

    Decision variables per period:
        P[t]   = regular production (units)
        OT[t]  = overtime production (units)
        S[t]   = subcontracted production (units)
        I[t]   = end-of-period inventory (units)
        B[t]   = end-of-period shortage/backlog (units)
        W[t]   = active workforce (workers)
        H[t]   = workers hired (workers)
        F[t]   = workers fired (workers)
    """

    VARS = ['P', 'OT', 'S', 'I', 'B', 'W', 'H', 'F']

    def __init__(self, params: dict):
        self.T                   = 6
        self.inv_init            = int(params["initial_inventory"])
        self.workers_init        = int(params["initial_workers"])
        self.cap_reg_per_worker  = float(params["capacity_regular_per_worker"])
        self.cap_ot_per_worker   = float(params["capacity_overtime_per_worker"])
        self.sell_price          = float(params["sell_price"])

        self.c_mat   = float(params["cost_material"])
        self.c_reg   = float(params["cost_regular_labor"])
        self.c_ot    = float(params["cost_overtime_labor"])
        self.c_sub   = float(params["cost_subcontract"])
        self.c_hold  = float(params["cost_holding"])
        self.c_short = float(params["cost_shortage"])
        self.c_hire  = float(params["cost_hiring"])
        self.c_fire  = float(params["cost_firing"])

        # Effective shortage cost = penalty + lost revenue
        self.c_short_eff = self.sell_price + self.c_short

        self.N = 8 * self.T

    def _idx(self, var: str, t: int) -> int:
        """Return flat index for variable var at period t."""
        return self.VARS.index(var) * self.T + t

    def _build_objective(self) -> np.ndarray:
        """Build objective function coefficient vector."""
        c = np.zeros(self.N)
        for t in range(self.T):
            c[self._idx('P',  t)] = self.c_mat + self.c_reg   # $26/unit
            c[self._idx('OT', t)] = self.c_mat + self.c_ot    # $34/unit
            c[self._idx('S',  t)] = self.c_mat + self.c_sub   # $40/unit
            c[self._idx('I',  t)] = self.c_hold                # $2/unit/month
            c[self._idx('B',  t)] = self.c_short_eff           # $45/unit (incl. lost rev)
            c[self._idx('H',  t)] = self.c_hire                # $300/worker
            c[self._idx('F',  t)] = self.c_fire                # $200/worker
        return c

    def _build_constraints(self, demand: list[int]):
        """Build equality and inequality constraint matrices."""
        A_eq, b_eq, A_ub, b_ub = [], [], [], []

        for t in range(self.T):
            # --- Inventory balance ---
            # I[t-1] - B[t-1] + P[t] + OT[t] + S[t] - I[t] + B[t] = D[t]
            r = np.zeros(self.N)
            r[self._idx('P',t)] = 1
            r[self._idx('OT',t)] = 1
            r[self._idx('S',t)] = 1
            r[self._idx('I',t)] = -1
            r[self._idx('B',t)] = 1
            if t == 0:
                rhs = demand[t] - self.inv_init
            else:
                r[self._idx('I', t-1)] = 1
                r[self._idx('B', t-1)] = -1
                rhs = demand[t]
            A_eq.append(r); b_eq.append(rhs)

            # --- Worker balance ---
            # W[t] - W[t-1] - H[t] + F[t] = 0
            r2 = np.zeros(self.N)
            r2[self._idx('W',t)] = 1
            r2[self._idx('H',t)] = -1
            r2[self._idx('F',t)] = 1
            if t == 0:
                rhs2 = self.workers_init
            else:
                r2[self._idx('W', t-1)] = -1
                rhs2 = 0
            A_eq.append(r2); b_eq.append(rhs2)

            # --- Regular capacity: P[t] <= W[t] * cap_reg ---
            r3 = np.zeros(self.N)
            r3[self._idx('P',t)] = 1
            r3[self._idx('W',t)] = -self.cap_reg_per_worker
            A_ub.append(r3); b_ub.append(0)

            # --- OT capacity: OT[t] <= W[t] * cap_ot ---
            r4 = np.zeros(self.N)
            r4[self._idx('OT',t)] = 1
            r4[self._idx('W',t)]  = -self.cap_ot_per_worker
            A_ub.append(r4); b_ub.append(0)

        return (np.array(A_eq), np.array(b_eq, dtype=float),
                np.array(A_ub), np.array(b_ub, dtype=float))

    def solve(self, demand: list[int]) -> Optional[list[tuple]]:
        """
        Solve the LP for given demand profile.

        Returns:
            List of tuples (demand, P, OT, S, I, B, W, H, F) per period,
            or None if solver fails.
        """
        c = self._build_objective()
        A_eq, b_eq, A_ub, b_ub = self._build_constraints(demand)

        result = linprog(
            c,
            A_ub=A_ub, b_ub=b_ub,
            A_eq=A_eq, b_eq=b_eq,
            bounds=[(0, None)] * self.N,
            method='highs'
        )

        if result.status != 0:
            return None

        x = result.x
        return [
            (demand[t],) + tuple(round(x[self._idx(v, t)]) for v in self.VARS)
            for t in range(self.T)
        ]

# =============================================================================
# FINANCIAL ANALYSIS
# =============================================================================

def compute_financials(
    rd: list[tuple],
    params: dict,
    promo_month: Optional[int] = None,
    promo_price: Optional[float] = None
) -> dict:
    """Compute full financial and operational metrics from LP solution."""
    sp   = float(params["sell_price"])
    cm   = float(params["cost_material"])
    cr   = float(params["cost_regular_labor"])
    co   = float(params["cost_overtime_labor"])
    cs   = float(params["cost_subcontract"])
    ch   = float(params["cost_holding"])
    csh  = float(params["cost_shortage"])
    chi  = float(params["cost_hiring"])
    cfi  = float(params["cost_firing"])

    # Revenue (handles promo pricing)
    revenue = sum(
        rd[t][0] * (promo_price if t == promo_month and promo_month is not None else sp)
        for t in range(len(rd))
    )

    # Costs
    mat   = sum((d[1]+d[2]+d[3])*cm  for d in rd)
    reg   = sum(d[1]*cr               for d in rd)
    ot    = sum(d[2]*co               for d in rd)
    sub   = sum(d[3]*cs               for d in rd)
    hold  = sum(d[4]*ch               for d in rd)
    short = sum(d[5]*csh              for d in rd)
    hire  = sum(d[7]*chi              for d in rd)
    fire  = sum(d[8]*cfi              for d in rd)
    total = mat+reg+ot+sub+hold+short+hire+fire

    demand = [d[0] for d in rd]
    cv = np.std(demand) / np.mean(demand)

    return {
        "revenue"       : revenue,
        "cost_material" : mat,
        "cost_reg_labor": reg,
        "cost_overtime" : ot,
        "cost_subcon"   : sub,
        "cost_holding"  : hold,
        "cost_shortage" : short,
        "cost_hiring"   : hire,
        "cost_firing"   : fire,
        "total_cost"    : total,
        "profit"        : revenue - total,
        "profit_margin" : (revenue - total) / revenue * 100,
        "cv_demand"     : cv * 100,
        "inv_cumul"     : sum(d[4] for d in rd),
        "shortage_total": sum(d[5] for d in rd),
        "prod_total"    : sum(d[1]+d[2]+d[3] for d in rd),
        "workers_hired" : sum(d[7] for d in rd),
        "workers_fired" : sum(d[8] for d in rd),
    }

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  FORWARD BUYING ANALYSIS — Final Clean Model")
    print("=" * 70)

    # Load data
    params   = load_parameters()
    demand_d = load_demand_scenarios()

    MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']

    # Scenario configuration
    scenario_config = [
        {
            "key"         : "S1_Baseline",
            "demand_col"  : "scenario_1_baseline",
            "label"       : "No Promotion (Baseline)",
            "promo_month" : None,
            "promo_price" : None,
        },
        {
            "key"         : "S2_Promo_Jan",
            "demand_col"  : "scenario_2_promo_jan",
            "label"       : "Promotion January ($35/unit)",
            "promo_month" : 0,
            "promo_price" : 35.0,
        },
        {
            "key"         : "S3_Promo_Apr",
            "demand_col"  : "scenario_3_promo_apr",
            "label"       : "Promotion April ($34/unit)",
            "promo_month" : 3,
            "promo_price" : 34.0,
        },
    ]

    # Initialize model
    model = AggregateProductionLP(params)

    all_results = {}
    all_rd      = {}

    # Solve each scenario
    print()
    for sc in scenario_config:
        demand = demand_d[sc["demand_col"]]
        rd     = model.solve(demand)

        if rd is None:
            print(f"  ❌ {sc['key']} — solver failed"); continue

        fin = compute_financials(rd, params, sc["promo_month"], sc["promo_price"])
        all_results[sc["key"]] = {"config": sc, "financials": fin, "demand": demand}
        all_rd[sc["key"]]      = rd
        print(f"  ✓ {sc['key']} — profit: ${fin['profit']:,.0f}")

    # Print production plans
    print()
    print("=" * 70)
    print("  PRODUCTION PLANS")
    print("=" * 70)

    for sc in scenario_config:
        key = sc["key"]
        rd  = all_rd[key]
        print()
        print(f"  {key}: {sc['label']}")
        print(f"  {'Mo':<5} {'Demand':>7} {'P_Reg':>7} {'P_OT':>6} "
              f"{'Sub':>6} {'Inv':>7} {'Short':>6} {'Wrkr':>6} {'Hire':>5}")
        print("  " + "-"*68)
        for t, d in enumerate(rd):
            dm,P,OT,S,I,B,W,H,F = d
            print(f"  {MONTHS[t]:<5} {dm:>7} {P:>7} {OT:>6} "
                  f"{S:>6} {I:>7} {B:>6} {W:>6} {H:>5}")

    # Comparison table
    print()
    print("=" * 70)
    print("  SUMMARY COMPARISON")
    print("=" * 70)
    print()

    keys   = [sc["key"] for sc in scenario_config]
    header = f"  {'Metric':<32}" + "".join(f" {k:>18}" for k in keys)
    print(header)
    print("  " + "-" * 90)

    baseline_profit = all_results["S1_Baseline"]["financials"]["profit"]

    metrics_to_show = [
        ("Revenue",         "revenue",        "$"),
        ("Total Cost",      "total_cost",     "$"),
        ("Profit",          "profit",         "$"),
        ("Profit Margin",   "profit_margin",  "%"),
        ("CV Demand",       "cv_demand",      "%"),
        ("Cumul. Inventory","inv_cumul",      ","),
        ("Total Shortage",  "shortage_total", ","),
        ("Workers Hired",   "workers_hired",  ","),
    ]

    for label, field, fmt in metrics_to_show:
        line = f"  {label:<32}"
        for k in keys:
            v = all_results[k]["financials"][field]
            if fmt == "$": line += f" ${v:>17,.0f}"
            elif fmt == "%": line += f" {v:>17.2f}%"
            else: line += f" {v:>18,}"
        print(line)

    print()
    print("  PROFIT DELTA vs BASELINE")
    for k in keys:
        delta = all_results[k]["financials"]["profit"] - baseline_profit
        sign  = "+" if delta >= 0 else ""
        print(f"  {k:<32} ${sign}{delta:>17,.0f}")

    # Export
    print()
    rows = []
    for label, field, _ in metrics_to_show:
        row = {"Metric": label}
        for k in keys:
            row[k] = all_results[k]["financials"][field]
        rows.append(row)
    pd.DataFrame(rows).to_csv("outputs/final_comparison.csv", index=False)
    print(f"  ✓ Exported: outputs/final_comparison.csv")

    # Monthly production plans export
    plan_rows = []
    for sc in scenario_config:
        key = sc["key"]
        for t, d in enumerate(all_rd[key]):
            dm,P,OT,S,I,B,W,H,F = d
            plan_rows.append({
                "scenario":key, "month":MONTHS[t], "month_num":t+1,
                "demand":dm, "prod_regular":P, "prod_overtime":OT,
                "subcontract":S, "inventory_end":I, "shortage":B,
                "workers":W, "hired":H, "fired":F, "total_prod":P+OT+S
            })
    pd.DataFrame(plan_rows).to_csv("outputs/final_production_plans.csv", index=False)
    print(f"  ✓ Exported: outputs/final_production_plans.csv")

    print()
    print("=" * 70)
    print("  Analysis complete.")
    print("  Key finding: Promotion timing > promotion size.")
    print(f"  Promo Jan profit loss: "
          f"${all_results['S2_Promo_Jan']['financials']['profit']-baseline_profit:,.0f}")
    print(f"  Promo Apr profit loss: "
          f"${all_results['S3_Promo_Apr']['financials']['profit']-baseline_profit:,.0f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
