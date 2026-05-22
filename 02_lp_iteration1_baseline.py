# =============================================================================
# 02_lp_iteration1_baseline.py
# Forward Buying Analysis — LP Iteration 1: Naive Baseline Model
# =============================================================================
#
# STATUS: ❌ FLAWED — documented for learning purposes
#
# WHAT HAPPENED:
#   First attempt at building the LP model. The model was structurally correct
#   (variables, constraints) but had a critical cost imbalance:
#
#   PROBLEM:
#     Shortage cost  = $5/unit/month
#     Production cost = $26/unit (regular)
#
#   Because shortage is so much cheaper than production, the LP solver
#   chose to let products be in shortage rather than produce them.
#   Result: production = 0, shortage = everything. Economically absurd.
#
#   ROOT CAUSE:
#     The model minimizes cost but doesn't account for the REVENUE LOST
#     when a shortage occurs. A shortage doesn't just cost $5 in penalty —
#     it also means the company doesn't earn $40 in revenue from that unit.
#     The model didn't know this, so it "preferred" shortage.
#
# FIX:
#   See 04_lp_fix_shortage_cost.py — effective shortage cost = $5 + $40 = $45
#
# HOW TO RUN:
#   python 02_lp_iteration1_baseline.py
# =============================================================================

import numpy as np
from scipy.optimize import linprog

print("=" * 65)
print("  LP ITERATION 1 — Naive Baseline (FLAWED)")
print("  This script documents the first attempt and its failure.")
print("=" * 65)

# -----------------------------------------------------------------------------
# PARAMETERS
# -----------------------------------------------------------------------------
T = 6                   # number of periods (months)
inv_init = 2000         # initial inventory
workers_init = 80       # initial workforce
days_per_month = 20     # working days per month
reg_hrs_day = 8         # regular hours per day
max_ot_month = 10       # max overtime hours per worker per month
labor_hrs_unit = 4      # labor hours per unit

# Derived capacity
cap_reg_per_worker = reg_hrs_day * days_per_month / labor_hrs_unit  # 40 units
cap_ot_per_worker  = max_ot_month / labor_hrs_unit                   # 2.5 units

# Cost parameters
c_mat   = 10    # material cost per unit
c_reg   = 16    # regular labor cost per unit (4 hrs x $4/hr)
c_ot    = 24    # OT labor cost per unit (4 hrs x $6/hr)
c_sub   = 30    # subcontract cost per unit
c_hold  = 2     # holding cost per unit per month
c_short = 5     # ← THE PROBLEM: only $5, much cheaper than producing ($26)
c_hire  = 300   # hiring cost per worker

# Demand baseline
demand = [1600, 3000, 3200, 3800, 2200, 2100]
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']

# -----------------------------------------------------------------------------
# VARIABLE INDEX HELPER
# Decision variables: [P, OT, S, I, B, W, H, F] x 6 months = 48 variables
# P  = regular production
# OT = overtime production
# S  = subcontract
# I  = inventory (end of period)
# B  = backlog/shortage (end of period)
# W  = workforce
# H  = hired workers
# F  = fired workers
# -----------------------------------------------------------------------------
def idx(var, t):
    order = ['P', 'OT', 'S', 'I', 'B', 'W', 'H', 'F']
    return order.index(var) * T + t

N = 8 * T  # total variables = 48

# -----------------------------------------------------------------------------
# OBJECTIVE FUNCTION (minimize total cost)
# FLAW: shortage cost is only $5, making shortage "cheaper" than production
# -----------------------------------------------------------------------------
c_obj = np.zeros(N)
for t in range(T):
    c_obj[idx('P', t)]   = c_mat + c_reg   # $26/unit regular
    c_obj[idx('OT', t)]  = c_mat + c_ot    # $34/unit OT
    c_obj[idx('S', t)]   = c_mat + c_sub   # $40/unit subcontract
    c_obj[idx('I', t)]   = c_hold          # $2/unit/month holding
    c_obj[idx('B', t)]   = c_short         # ← $5/unit — TOO CHEAP vs $26 production
    c_obj[idx('H', t)]   = c_hire          # $300/worker hired

# NOTE: W and F have zero cost (firing cost not included in this iteration)

# -----------------------------------------------------------------------------
# CONSTRAINTS
# -----------------------------------------------------------------------------
A_eq_rows, b_eq_rows = [], []
A_ub_rows, b_ub_rows = [], []

for t in range(T):
    # EQUALITY 1: Inventory balance
    # I[t-1] - B[t-1] + P[t] + OT[t] + S[t] - I[t] + B[t] = D[t]
    row = np.zeros(N)
    row[idx('P', t)]  =  1
    row[idx('OT', t)] =  1
    row[idx('S', t)]  =  1
    row[idx('I', t)]  = -1
    row[idx('B', t)]  =  1
    if t == 0:
        rhs = demand[t] - inv_init      # inventory_0 = 2000
    else:
        row[idx('I', t-1)] =  1
        row[idx('B', t-1)] = -1
        rhs = demand[t]
    A_eq_rows.append(row)
    b_eq_rows.append(rhs)

    # EQUALITY 2: Worker balance
    # W[t] = W[t-1] + H[t] - F[t]
    row2 = np.zeros(N)
    row2[idx('W', t)] =  1
    row2[idx('H', t)] = -1
    row2[idx('F', t)] =  1
    if t == 0:
        rhs2 = workers_init
    else:
        row2[idx('W', t-1)] = -1
        rhs2 = 0
    A_eq_rows.append(row2)
    b_eq_rows.append(rhs2)

    # INEQUALITY 1: Regular capacity
    # P[t] <= W[t] * cap_reg_per_worker
    row3 = np.zeros(N)
    row3[idx('P', t)] =  1
    row3[idx('W', t)] = -cap_reg_per_worker
    A_ub_rows.append(row3)
    b_ub_rows.append(0)

    # INEQUALITY 2: OT capacity
    # OT[t] <= W[t] * cap_ot_per_worker
    row4 = np.zeros(N)
    row4[idx('OT', t)] =  1
    row4[idx('W', t)]  = -cap_ot_per_worker
    A_ub_rows.append(row4)
    b_ub_rows.append(0)

bounds = [(0, None)] * N

# -----------------------------------------------------------------------------
# SOLVE
# -----------------------------------------------------------------------------
result = linprog(
    c_obj,
    A_ub=np.array(A_ub_rows),
    b_ub=np.array(b_ub_rows, dtype=float),
    A_eq=np.array(A_eq_rows),
    b_eq=np.array(b_eq_rows, dtype=float),
    bounds=bounds,
    method='highs'
)

# -----------------------------------------------------------------------------
# SHOW RESULT + EXPLAIN THE PROBLEM
# -----------------------------------------------------------------------------
print()
if result.status != 0:
    print(f"  Solver failed: {result.message}")
else:
    x = result.x
    print(f"  Solver status: {result.message}")
    print()
    print(f"  {'Month':<6} {'Demand':>7} {'P_Reg':>7} {'P_OT':>7} "
          f"{'Subcon':>7} {'Inv':>7} {'Short':>7} {'Workers':>8}")
    print("  " + "-" * 65)

    for t in range(T):
        P  = round(x[idx('P', t)])
        OT = round(x[idx('OT', t)])
        S  = round(x[idx('S', t)])
        I  = round(x[idx('I', t)])
        B  = round(x[idx('B', t)])
        W  = round(x[idx('W', t)])
        print(f"  {months[t]:<6} {demand[t]:>7} {P:>7} {OT:>7} "
              f"{S:>7} {I:>7} {B:>7} {W:>8}")

    print()
    total_prod     = sum(round(x[idx('P', t)]) + round(x[idx('OT', t)]) +
                        round(x[idx('S', t)]) for t in range(T))
    total_shortage = sum(round(x[idx('B', t)]) for t in range(T))

    print(f"  Total production : {total_prod:,} units")
    print(f"  Total shortage   : {total_shortage:,} units")
    print()

    # -------------------------------------------------------------------------
    # EXPLAIN THE FAILURE
    # -------------------------------------------------------------------------
    print("=" * 65)
    print("  ❌ WHY THIS MODEL IS WRONG")
    print("=" * 65)
    print()
    print("  The solver chose shortage over production because:")
    print()
    print(f"    Cost to PRODUCE 1 unit (regular) : ${c_mat + c_reg}")
    print(f"    Cost to SHORTAGE 1 unit          : ${c_short}")
    print()
    print("  From the model's perspective, it's 'cheaper' to let")
    print("  demand go unfulfilled ($5) than to produce ($26).")
    print()
    print("  But this is WRONG in reality, because a shortage doesn't")
    print("  just cost $5 — the company also LOSES the $40 revenue")
    print("  from that unit that was never sold.")
    print()
    print("  Effective cost of 1 unit shortage:")
    print(f"    Penalty           : ${c_short}")
    print(f"    Lost revenue      : $40")
    print(f"    TOTAL EFFECTIVE   : ${c_short + 40}")
    print()
    print("  When shortage costs $45, production at $26 becomes")
    print("  the preferred choice — which is economically correct.")
    print()
    print("  → FIX: See 04_lp_fix_shortage_cost.py")
    print()
    print("  NOTE: 03_lp_iteration2_improve.py adds firing cost as")
    print("  an intermediate improvement before the shortage fix.")
