# =============================================================================
# 03_lp_iteration2_improve.py
# Forward Buying Analysis — LP Iteration 2: Adding Firing Cost
# =============================================================================
#
# STATUS: ⚠️ PARTIAL IMPROVEMENT — still has the shortage cost flaw
#
# WHAT CHANGED FROM ITERATION 1:
#   Added firing cost ($200/worker) to the objective function.
#   In iteration 1, there was no cost for firing workers — meaning the solver
#   could freely reduce workforce without penalty. This is not realistic.
#
# WHAT STILL NEEDS FIXING:
#   The shortage cost is still only $5/unit, which is still much cheaper
#   than production ($26/unit). The solver still prefers shortage.
#   This will be fixed in the next iteration.
#
# WHY DOCUMENT THIS INTERMEDIATE STEP:
#   - Shows the iterative nature of model building
#   - Firing cost is a real and important constraint
#   - Demonstrates that model improvement is rarely one big jump
#
# HOW TO RUN:
#   python 03_lp_iteration2_improve.py
# =============================================================================

import numpy as np
from scipy.optimize import linprog

print("=" * 65)
print("  LP ITERATION 2 — Added Firing Cost (Partial Improvement)")
print("  Still has shortage cost flaw — see iteration 4 for final fix")
print("=" * 65)

# -----------------------------------------------------------------------------
# PARAMETERS (same as iteration 1)
# -----------------------------------------------------------------------------
T = 6
inv_init       = 2000
workers_init   = 80
cap_reg_per_worker = 40.0   # (8 hrs/day x 20 days) / 4 hrs/unit
cap_ot_per_worker  = 2.5    # 10 hrs / 4 hrs/unit

c_mat   = 10
c_reg   = 16
c_ot    = 24
c_sub   = 30
c_hold  = 2
c_short = 5     # still flawed — too cheap vs production
c_hire  = 300
c_fire  = 200   # ← NEW in this iteration: firing cost added

demand = [1600, 3000, 3200, 3800, 2200, 2100]
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']

def idx(var, t):
    order = ['P', 'OT', 'S', 'I', 'B', 'W', 'H', 'F']
    return order.index(var) * T + t

N = 8 * T

# -----------------------------------------------------------------------------
# OBJECTIVE FUNCTION
# Key change: c_fire is now included (was 0 in iteration 1)
# -----------------------------------------------------------------------------
c_obj = np.zeros(N)
for t in range(T):
    c_obj[idx('P', t)]  = c_mat + c_reg    # $26/unit
    c_obj[idx('OT', t)] = c_mat + c_ot     # $34/unit
    c_obj[idx('S', t)]  = c_mat + c_sub    # $40/unit
    c_obj[idx('I', t)]  = c_hold           # $2/unit/month
    c_obj[idx('B', t)]  = c_short          # $5/unit — still flawed
    c_obj[idx('H', t)]  = c_hire           # $300/worker
    c_obj[idx('F', t)]  = c_fire           # $200/worker ← NEW

print()
print("  Objective function coefficients (cost per unit/worker):")
print(f"    Regular production : ${c_mat + c_reg}/unit")
print(f"    OT production      : ${c_mat + c_ot}/unit")
print(f"    Subcontract        : ${c_mat + c_sub}/unit")
print(f"    Holding            : ${c_hold}/unit/month")
print(f"    Shortage           : ${c_short}/unit/month  ← still too low")
print(f"    Hiring             : ${c_hire}/worker")
print(f"    Firing             : ${c_fire}/worker  ← NEW (was 0 in iter 1)")
print()

# -----------------------------------------------------------------------------
# CONSTRAINTS (identical to iteration 1)
# -----------------------------------------------------------------------------
A_eq_rows, b_eq_rows = [], []
A_ub_rows, b_ub_rows = [], []

for t in range(T):
    # Inventory balance
    row = np.zeros(N)
    row[idx('P', t)] = 1; row[idx('OT', t)] = 1; row[idx('S', t)] = 1
    row[idx('I', t)] = -1; row[idx('B', t)] = 1
    if t == 0:
        rhs = demand[t] - inv_init
    else:
        row[idx('I', t-1)] = 1; row[idx('B', t-1)] = -1
        rhs = demand[t]
    A_eq_rows.append(row); b_eq_rows.append(rhs)

    # Worker balance
    row2 = np.zeros(N)
    row2[idx('W', t)] = 1; row2[idx('H', t)] = -1; row2[idx('F', t)] = 1
    if t == 0: rhs2 = workers_init
    else: row2[idx('W', t-1)] = -1; rhs2 = 0
    A_eq_rows.append(row2); b_eq_rows.append(rhs2)

    # Regular capacity
    row3 = np.zeros(N)
    row3[idx('P', t)] = 1; row3[idx('W', t)] = -cap_reg_per_worker
    A_ub_rows.append(row3); b_ub_rows.append(0)

    # OT capacity
    row4 = np.zeros(N)
    row4[idx('OT', t)] = 1; row4[idx('W', t)] = -cap_ot_per_worker
    A_ub_rows.append(row4); b_ub_rows.append(0)

bounds = [(0, None)] * N

result = linprog(
    c_obj,
    A_ub=np.array(A_ub_rows), b_ub=np.array(b_ub_rows, dtype=float),
    A_eq=np.array(A_eq_rows), b_eq=np.array(b_eq_rows, dtype=float),
    bounds=bounds, method='highs'
)

# -----------------------------------------------------------------------------
# RESULTS + ANALYSIS
# -----------------------------------------------------------------------------
print(f"  Solver: {result.message}")
print()

if result.status == 0:
    x = result.x
    print(f"  {'Month':<6} {'Demand':>7} {'P_Reg':>7} {'P_OT':>7} "
          f"{'Subcon':>7} {'Inv':>7} {'Short':>7} {'Workers':>8} {'Hire':>5} {'Fire':>5}")
    print("  " + "-" * 75)

    for t in range(T):
        vals = {v: round(x[idx(v, t)]) for v in ['P','OT','S','I','B','W','H','F']}
        print(f"  {months[t]:<6} {demand[t]:>7} {vals['P']:>7} {vals['OT']:>7} "
              f"{vals['S']:>7} {vals['I']:>7} {vals['B']:>7} "
              f"{vals['W']:>8} {vals['H']:>5} {vals['F']:>5}")

    total_prod = sum(round(x[idx('P',t)]+x[idx('OT',t)]+x[idx('S',t)]) for t in range(T))
    total_short = sum(round(x[idx('B', t)]) for t in range(T))
    print()
    print(f"  Total production : {total_prod:,} units")
    print(f"  Total shortage   : {total_short:,} units")

# -----------------------------------------------------------------------------
# COMPARISON: What changed vs iteration 1?
# -----------------------------------------------------------------------------
print()
print("=" * 65)
print("  WHAT CHANGED vs ITERATION 1?")
print("=" * 65)
print()
print("  Added : c_fire = $200/worker in the objective function")
print()
print("  Effect: The solver now has a reason to NOT fire workers")
print("          unnecessarily. In iteration 1, firing was 'free'.")
print()
print("  However, the core flaw remains:")
print(f"    Shortage cost = ${c_short}/unit")
print(f"    Production cost = ${c_mat + c_reg}/unit")
print()
print("  The solver STILL prefers shortage over production.")
print("  Firing cost alone does not fix the economic imbalance.")
print()
print("  → NEXT: 04_lp_fix_shortage_cost.py")
print("    Effective shortage cost = penalty + lost revenue")
print(f"    = ${c_short} + $40 = $45/unit")
print()
print("  At $45, shortage becomes MORE expensive than production ($26)")
print("  → solver will correctly choose to produce.")
