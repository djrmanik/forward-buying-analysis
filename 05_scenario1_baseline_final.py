# =============================================================================
# 05_scenario1_baseline_final.py
# Forward Buying Analysis — Scenario 1: No Promotion (Baseline)
# =============================================================================
#
# STATUS: ✅ FINAL
#
# PURPOSE:
#   Run the corrected LP model on Scenario 1 (no promotion / baseline demand).
#   This establishes the reference point for comparing Scenarios 2 and 3.
#
# SCENARIO 1 SETUP:
#   - No promotional discount
#   - Selling price: $40/unit (constant)
#   - Demand follows natural pattern: low in Jan, peaks in Apr, falls in May-Jun
#
# KEY FINDINGS (documented here, proved by the model output):
#   - January: production = 0 (initial inventory of 2,000 covers demand of 1,600)
#   - Feb-Mar: full capacity production (3,200/month) to build buffer for April
#   - April: buffer (600 units) + production (3,200) = 3,800 = demand. No shortage.
#   - May-Jun: production exactly matches demand. No excess.
#   - No overtime, subcontracting, hiring, or firing needed.
#
# HOW TO RUN:
#   python 05_scenario1_baseline_final.py
# =============================================================================

import numpy as np
from scipy.optimize import linprog

print("=" * 70)
print("  SCENARIO 1 — No Promotion (Baseline)")
print("  Reference point for forward buying analysis")
print("=" * 70)

# -----------------------------------------------------------------------------
# PARAMETERS (final, correct version)
# -----------------------------------------------------------------------------
T = 6
months         = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
inv_init       = 2000
workers_init   = 80
sell_price     = 40

cap_reg_per_worker = 40.0
cap_ot_per_worker  = 2.5

c_mat   = 10
c_reg   = 16
c_ot    = 24
c_sub   = 30
c_hold  = 2
c_short = 5
c_hire  = 300
c_fire  = 200
c_short_eff = sell_price + c_short  # $45

demand = [1600, 3000, 3200, 3800, 2200, 2100]

print()
print(f"  Demand (Jan-Jun): {demand}")
print(f"  Total demand    : {sum(demand):,} units")
print(f"  Selling price   : ${sell_price}/unit")
print()

# -----------------------------------------------------------------------------
# SOLVER FUNCTION (reusable — same function used in scripts 05, 06, 07, 08)
# -----------------------------------------------------------------------------
def idx(var, t, T=6):
    order = ['P', 'OT', 'S', 'I', 'B', 'W', 'H', 'F']
    return order.index(var) * T + t

def solve_lp(demand, T, inv_init, workers_init,
             cap_reg_per_worker, cap_ot_per_worker,
             c_mat, c_reg, c_ot, c_sub, c_hold,
             c_short_eff, c_hire, c_fire):
    """
    Solve the aggregate production planning LP.

    Returns list of tuples: (demand, P, OT, S, I, B, W, H, F) per period.
    Returns None if solver fails.
    """
    N = 8 * T
    _idx = lambda v, t: ['P','OT','S','I','B','W','H','F'].index(v) * T + t

    c_obj = np.zeros(N)
    for t in range(T):
        c_obj[_idx('P', t)]  = c_mat + c_reg
        c_obj[_idx('OT', t)] = c_mat + c_ot
        c_obj[_idx('S', t)]  = c_mat + c_sub
        c_obj[_idx('I', t)]  = c_hold
        c_obj[_idx('B', t)]  = c_short_eff
        c_obj[_idx('H', t)]  = c_hire
        c_obj[_idx('F', t)]  = c_fire

    A_eq, b_eq, A_ub, b_ub = [], [], [], []

    for t in range(T):
        # Inventory balance
        r = np.zeros(N)
        r[_idx('P',t)]=1; r[_idx('OT',t)]=1; r[_idx('S',t)]=1
        r[_idx('I',t)]=-1; r[_idx('B',t)]=1
        if t==0: rhs = demand[t] - inv_init
        else:
            r[_idx('I',t-1)]=1; r[_idx('B',t-1)]=-1
            rhs = demand[t]
        A_eq.append(r); b_eq.append(rhs)

        # Worker balance
        r2 = np.zeros(N)
        r2[_idx('W',t)]=1; r2[_idx('H',t)]=-1; r2[_idx('F',t)]=1
        if t==0: rhs2 = workers_init
        else: r2[_idx('W',t-1)]=-1; rhs2=0
        A_eq.append(r2); b_eq.append(rhs2)

        # Regular capacity
        r3 = np.zeros(N)
        r3[_idx('P',t)]=1; r3[_idx('W',t)]=-cap_reg_per_worker
        A_ub.append(r3); b_ub.append(0)

        # OT capacity
        r4 = np.zeros(N)
        r4[_idx('OT',t)]=1; r4[_idx('W',t)]=-cap_ot_per_worker
        A_ub.append(r4); b_ub.append(0)

    result = linprog(c_obj,
                     A_ub=np.array(A_ub), b_ub=np.array(b_ub, dtype=float),
                     A_eq=np.array(A_eq), b_eq=np.array(b_eq, dtype=float),
                     bounds=[(0, None)]*N, method='highs')

    if result.status != 0:
        return None

    x = result.x
    return [(demand[t],) + tuple(round(x[_idx(v,t)]) for v in ['P','OT','S','I','B','W','H','F'])
            for t in range(T)]

# -----------------------------------------------------------------------------
# RUN SCENARIO 1
# -----------------------------------------------------------------------------
rd = solve_lp(demand, T, inv_init, workers_init,
              cap_reg_per_worker, cap_ot_per_worker,
              c_mat, c_reg, c_ot, c_sub, c_hold,
              c_short_eff, c_hire, c_fire)

if rd is None:
    print("  ❌ Solver failed.")
    exit()

# -----------------------------------------------------------------------------
# PRINT PRODUCTION PLAN TABLE
# -----------------------------------------------------------------------------
print("  AGGREGATE PRODUCTION PLAN:")
print()
print(f"  {'Month':<6} {'Demand':>7} {'P_Reg':>7} {'P_OT':>7} "
      f"{'Subcon':>7} {'Inv':>8} {'Short':>7} {'Workers':>8} {'Hire':>5} {'Fire':>5}")
print("  " + "-" * 80)

for t, d in enumerate(rd):
    dm, P, OT, S, I, B, W, H, F = d
    print(f"  {months[t]:<6} {dm:>7} {P:>7} {OT:>7} "
          f"{S:>7} {I:>8} {B:>7} {W:>8} {H:>5} {F:>5}")

print("  " + "-" * 80)
total_prod = sum(d[1]+d[2]+d[3] for d in rd)
total_inv  = sum(d[4] for d in rd)
total_short= sum(d[5] for d in rd)
print(f"  {'TOTAL':<6} {sum(d[0] for d in rd):>7} {total_prod:>7} {'0':>7} "
      f"{'0':>7} {total_inv:>8} {total_short:>7}")

# -----------------------------------------------------------------------------
# SUPPLY CHAIN FLOW DETAIL
# -----------------------------------------------------------------------------
print()
print("  SUPPLY CHAIN FLOW DETAIL (per month):")
print()
print(f"  {'Month':<6} {'Inv_Start':>10} {'Production':>11} "
      f"{'Total Supply':>13} {'Demand':>8} {'Inv_End':>9} {'Shortage':>9}")
print("  " + "-" * 75)

inv_prev, short_prev = inv_init, 0
for t, d in enumerate(rd):
    dm, P, OT, S, I, B, W, H, F = d
    prod      = P + OT + S
    inv_start = inv_prev - short_prev
    supply    = inv_start + prod
    print(f"  {months[t]:<6} {inv_start:>10} {prod:>11} "
          f"{supply:>13} {dm:>8} {I:>9} {B:>9}")
    inv_prev, short_prev = I, B

# -----------------------------------------------------------------------------
# COST BREAKDOWN
# -----------------------------------------------------------------------------
mat_cost   = sum((d[1]+d[2]+d[3])*c_mat for d in rd)
reg_cost   = sum(d[1]*c_reg for d in rd)
ot_cost    = sum(d[2]*c_ot  for d in rd)
sub_cost   = sum(d[3]*c_sub for d in rd)
hold_cost  = sum(d[4]*c_hold for d in rd)
short_cost = sum(d[5]*c_short for d in rd)
hire_cost  = sum(d[7]*c_hire for d in rd)
fire_cost  = sum(d[8]*c_fire for d in rd)
total_cost = mat_cost+reg_cost+ot_cost+sub_cost+hold_cost+short_cost+hire_cost+fire_cost

revenue = sum(d[0] for d in rd) * sell_price
profit  = revenue - total_cost

print()
print("  COST BREAKDOWN:")
print()
print(f"  {'Material ($10/unit)':<40} ${mat_cost:>10,.0f}")
print(f"  {'Regular Labor ($16/unit)':<40} ${reg_cost:>10,.0f}")
print(f"  {'Overtime Labor ($24/unit)':<40} ${ot_cost:>10,.0f}")
print(f"  {'Subcontract ($40/unit)':<40} ${sub_cost:>10,.0f}")
print(f"  {'Holding ($2/unit/month)':<40} ${hold_cost:>10,.0f}")
print(f"  {'Shortage Penalty ($5/unit/month)':<40} ${short_cost:>10,.0f}")
print(f"  {'Hiring ($300/worker)':<40} ${hire_cost:>10,.0f}")
print(f"  {'Firing ($200/worker)':<40} ${fire_cost:>10,.0f}")
print(f"  {'─'*52}")
print(f"  {'TOTAL COST':<40} ${total_cost:>10,.0f}")
print(f"  {'TOTAL REVENUE ($40/unit)':<40} ${revenue:>10,.0f}")
print(f"  {'PROFIT':<40} ${profit:>10,.0f}")
print(f"  {'Profit Margin':<40} {profit/revenue*100:>9.2f}%")

# -----------------------------------------------------------------------------
# STATISTICAL METRICS
# -----------------------------------------------------------------------------
cv = np.std(demand) / np.mean(demand)
print()
print("  STATISTICAL METRICS:")
print()
print(f"  CV Demand                    : {cv*100:.2f}%")
print(f"  Cumulative Inventory         : {total_inv:,} unit-months")
print(f"  Total Shortage               : {total_short:,} units")
print(f"  Total Units Produced         : {total_prod:,} units")
print(f"  Avg Production/Month         : {total_prod/T:,.1f} units")
print(f"  Max Workers                  : {max(d[6] for d in rd)} workers")

# -----------------------------------------------------------------------------
# KEY INSIGHTS
# -----------------------------------------------------------------------------
print()
print("=" * 70)
print("  KEY INSIGHTS — SCENARIO 1")
print("=" * 70)
print()
print("  1. January: No production needed.")
print(f"     Initial inventory (2,000) covers demand (1,600). Surplus: 400 units.")
print()
print("  2. Feb-Mar: Full capacity production (3,200 units/month).")
print(f"     Excess over demand: Feb +200, Mar +600 → buffer built = 600 units")
print()
print("  3. April: Buffer + production exactly meets peak demand.")
print(f"     600 (buffer) + 3,200 (prod) = 3,800 = demand. No overtime needed.")
print()
print("  4. May-Jun: Production exactly matches falling demand. No excess.")
print()
print("  5. No overtime, subcontracting, hiring, or firing required.")
print(f"     Largest cost: Regular labor = ${reg_cost:,.0f} ({reg_cost/total_cost*100:.1f}% of total)")
print()
print(f"  → BASELINE PROFIT: ${profit:,.0f}")
print("  → This is the benchmark for Scenarios 2 and 3.")
print()
print("  → NEXT: 06_scenario2_promo_jan.py")
