# =============================================================================
# 06_scenario2_promo_jan.py
# Forward Buying Analysis — Scenario 2: Promotion in January (Low Demand Month)
# =============================================================================
#
# STATUS: ✅ FINAL
#
# SCENARIO 2 SETUP:
#   - Promotional discount in January only
#   - Price: $35/unit in Jan (down from $40), $40/unit in Feb-Jun
#   - Forward buying mechanism:
#       Jan demand: +10% (genuine new demand from lower price)
#       Feb demand: -80 units (shifted forward to Jan)
#       Mar demand: -80 units (shifted forward to Jan)
#       Apr-Jun demand: unchanged (too far from promo period)
#   - Total demand: still 15,900 units (redistributed, not increased)
#
# KEY QUESTION:
#   Does promoting in the lowest-demand month improve supply chain efficiency?
#
# KEY FINDING (proved by LP output):
#   - Production plan almost identical to baseline
#   - CV demand DECREASES from 28.22% to 26.30% (more stable)
#   - Cumulative inventory DECREASES (more efficient)
#   - Profit loss: only -$8,320 (entirely from discount given)
#   - No additional workforce, overtime, or subcontracting needed
#
# HOW TO RUN:
#   python 06_scenario2_promo_jan.py
# =============================================================================

import numpy as np
from scipy.optimize import linprog

print("=" * 70)
print("  SCENARIO 2 — Promotion in January ($35/unit)")
print("  Forward buying: demand shifts from Feb-Mar to Jan")
print("=" * 70)

# -----------------------------------------------------------------------------
# PARAMETERS
# -----------------------------------------------------------------------------
T = 6
months         = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
inv_init       = 2000
workers_init   = 80
sell_price     = 40
promo_price    = 35      # discounted price in January
promo_month    = 0       # January = index 0

cap_reg_per_worker = 40.0
cap_ot_per_worker  = 2.5

c_mat   = 10; c_reg = 16; c_ot = 24; c_sub = 30
c_hold  = 2;  c_short = 5; c_hire = 300; c_fire = 200
c_short_eff = sell_price + c_short   # $45

# Demand with forward buying effect
demand_base = [1600, 3000, 3200, 3800, 2200, 2100]
demand_s2   = [1760, 2920, 3120, 3800, 2200, 2100]

print()
print("  FORWARD BUYING MECHANISM:")
print()
print(f"  {'Month':<6} {'Baseline':>10} {'Promo S2':>10} {'Change':>10} {'Reason'}")
print("  " + "-" * 65)
for t in range(T):
    delta = demand_s2[t] - demand_base[t]
    if t == 0:
        reason = "+10% genuine new demand (price incentive)"
    elif t in [1, 2]:
        reason = f"forward buying (-{abs(delta)} units shifted to Jan)"
    else:
        reason = "unchanged (too far from promo period)"
    sign = "+" if delta >= 0 else ""
    print(f"  {months[t]:<6} {demand_base[t]:>10} {demand_s2[t]:>10} "
          f"{sign}{delta:>9} {reason}")
print()
print(f"  Total demand (base): {sum(demand_base):,} units")
print(f"  Total demand (S2)  : {sum(demand_s2):,} units  ← same total, redistributed")

# -----------------------------------------------------------------------------
# SOLVER (same function as script 05)
# -----------------------------------------------------------------------------
def idx(var, t, T=6):
    return ['P','OT','S','I','B','W','H','F'].index(var) * T + t

def solve_lp(demand):
    N = 8 * T
    _i = lambda v, t: idx(v, t, T)

    c_obj = np.zeros(N)
    for t in range(T):
        c_obj[_i('P',t)]  = c_mat+c_reg;  c_obj[_i('OT',t)] = c_mat+c_ot
        c_obj[_i('S',t)]  = c_mat+c_sub;  c_obj[_i('I',t)]  = c_hold
        c_obj[_i('B',t)]  = c_short_eff;  c_obj[_i('H',t)]  = c_hire
        c_obj[_i('F',t)]  = c_fire

    Aeq, beq, Aub, bub = [], [], [], []
    for t in range(T):
        r = np.zeros(N)
        r[_i('P',t)]=1; r[_i('OT',t)]=1; r[_i('S',t)]=1
        r[_i('I',t)]=-1; r[_i('B',t)]=1
        if t==0: rhs = demand[t] - inv_init
        else: r[_i('I',t-1)]=1; r[_i('B',t-1)]=-1; rhs=demand[t]
        Aeq.append(r); beq.append(rhs)

        r2 = np.zeros(N)
        r2[_i('W',t)]=1; r2[_i('H',t)]=-1; r2[_i('F',t)]=1
        if t==0: rhs2=workers_init
        else: r2[_i('W',t-1)]=-1; rhs2=0
        Aeq.append(r2); beq.append(rhs2)

        r3 = np.zeros(N)
        r3[_i('P',t)]=1; r3[_i('W',t)]=-cap_reg_per_worker
        Aub.append(r3); bub.append(0)

        r4 = np.zeros(N)
        r4[_i('OT',t)]=1; r4[_i('W',t)]=-cap_ot_per_worker
        Aub.append(r4); bub.append(0)

    res = linprog(c_obj,
                  A_ub=np.array(Aub), b_ub=np.array(bub, dtype=float),
                  A_eq=np.array(Aeq), b_eq=np.array(beq, dtype=float),
                  bounds=[(0,None)]*N, method='highs')
    if res.status != 0: return None
    x = res.x
    return [(demand[t],)+tuple(round(x[_i(v,t)]) for v in ['P','OT','S','I','B','W','H','F'])
            for t in range(T)]

# -----------------------------------------------------------------------------
# RUN
# -----------------------------------------------------------------------------
rd1 = solve_lp(demand_base)   # baseline for comparison
rd2 = solve_lp(demand_s2)     # scenario 2

print()
print("  AGGREGATE PRODUCTION PLAN — SCENARIO 2:")
print()
print(f"  {'Month':<6} {'Demand':>7} {'P_Reg':>7} {'P_OT':>7} "
      f"{'Subcon':>7} {'Inv':>8} {'Short':>7} {'Workers':>8} {'Hire':>5} {'Fire':>5}")
print("  " + "-" * 80)
for t, d in enumerate(rd2):
    dm,P,OT,S,I,B,W,H,F = d
    print(f"  {months[t]:<6} {dm:>7} {P:>7} {OT:>7} "
          f"{S:>7} {I:>8} {B:>7} {W:>8} {H:>5} {F:>5}")
print("  " + "-" * 80)

# -----------------------------------------------------------------------------
# INVENTORY COMPARISON: S1 vs S2
# -----------------------------------------------------------------------------
print()
print("  INVENTORY COMPARISON (Baseline vs Promo Jan):")
print()
print(f"  {'Month':<6} {'Inv S1':>8} {'Inv S2':>8} {'Delta':>8} {'Explanation'}")
print("  " + "-" * 65)
explanations = [
    "S2 demand higher → more stock consumed",
    "Carries over lower inv from Jan",
    "LP fills back to 600 for April buffer",
    "Both: buffer fully used for April",
    "Both: produce exactly to demand",
    "Both: produce exactly to demand",
]
for t in range(T):
    delta = rd2[t][4] - rd1[t][4]
    sign = "+" if delta >= 0 else ""
    print(f"  {months[t]:<6} {rd1[t][4]:>8} {rd2[t][4]:>8} "
          f"{sign}{delta:>7}  {explanations[t]}")

# -----------------------------------------------------------------------------
# REVENUE CALCULATION (two prices involved)
# -----------------------------------------------------------------------------
rev_jan  = demand_s2[0] * promo_price         # 1,760 × $35
rev_rest = sum(demand_s2[1:]) * sell_price     # 14,140 × $40
revenue  = rev_jan + rev_rest

rev_base      = sum(demand_base) * sell_price
revenue_loss  = rev_base - revenue
discount_loss = demand_s2[0] * (sell_price - promo_price)

# Costs
mat_cost  = sum((d[1]+d[2]+d[3])*c_mat for d in rd2)
reg_cost  = sum(d[1]*c_reg for d in rd2)
hold_cost = sum(d[4]*c_hold for d in rd2)
hire_cost = sum(d[7]*c_hire for d in rd2)
fire_cost = sum(d[8]*c_fire for d in rd2)
total_cost = mat_cost+reg_cost+hold_cost+hire_cost+fire_cost
profit = revenue - total_cost

# Baseline comparison
mat1  = sum((d[1]+d[2]+d[3])*c_mat for d in rd1)
reg1  = sum(d[1]*c_reg for d in rd1)
hold1 = sum(d[4]*c_hold for d in rd1)
cost1 = mat1+reg1+hold1
rev1  = sum(d[0] for d in rd1)*sell_price
prof1 = rev1-cost1

print()
print("  FINANCIAL ANALYSIS:")
print()
print(f"  Revenue breakdown:")
print(f"    Jan: {demand_s2[0]:,} units × ${promo_price} (promo)   = ${rev_jan:>10,.0f}")
print(f"    Feb-Jun: {sum(demand_s2[1:]):,} units × ${sell_price} (normal) = ${rev_rest:>10,.0f}")
print(f"    TOTAL REVENUE S2                     = ${revenue:>10,.0f}")
print(f"    Revenue loss vs baseline             = ${revenue_loss:>10,.0f}")
print(f"    (= {demand_s2[0]:,} units × $5 discount = ${discount_loss:,.0f})")
print()
print(f"  Cost comparison:")
print(f"    {'Item':<35} {'S1 Baseline':>13} {'S2 Promo Jan':>13} {'Delta':>10}")
print("    " + "-" * 75)
print(f"    {'Material':<35} ${mat1:>12,.0f} ${mat_cost:>12,.0f} ${mat_cost-mat1:>+9,.0f}")
print(f"    {'Regular Labor':<35} ${reg1:>12,.0f} ${reg_cost:>12,.0f} ${reg_cost-reg1:>+9,.0f}")
print(f"    {'Holding':<35} ${hold1:>12,.0f} ${hold_cost:>12,.0f} ${hold_cost-hold1:>+9,.0f}")
print(f"    {'Hiring + Firing':<35} ${'0':>12} ${'0':>12} ${'0':>+9}")
print("    " + "-" * 75)
print(f"    {'TOTAL COST':<35} ${cost1:>12,.0f} ${total_cost:>12,.0f} ${total_cost-cost1:>+9,.0f}")
print(f"    {'REVENUE':<35} ${rev1:>12,.0f} ${revenue:>12,.0f} ${revenue-rev1:>+9,.0f}")
print(f"    {'PROFIT':<35} ${prof1:>12,.0f} ${profit:>12,.0f} ${profit-prof1:>+9,.0f}")

# CV
cv1 = np.std(demand_base)/np.mean(demand_base)
cv2 = np.std(demand_s2)/np.mean(demand_s2)
inv1 = sum(d[4] for d in rd1)
inv2 = sum(d[4] for d in rd2)

print()
print("  OPERATIONAL METRICS:")
print()
print(f"  {'Metric':<35} {'S1 Baseline':>13} {'S2 Promo Jan':>13} {'Delta':>10}")
print("  " + "-" * 75)
print(f"  {'CV Demand':<35} {cv1*100:>12.2f}% {cv2*100:>12.2f}% {(cv2-cv1)*100:>+9.2f}%")
print(f"  {'Cumulative Inventory (unit-mo)':<35} {inv1:>13,} {inv2:>13,} {inv2-inv1:>+10,}")
print(f"  {'Total Shortage':<35} {'0':>13} {'0':>13} {'0':>10}")
print(f"  {'Workers Hired':<35} {'0':>13} {'0':>13} {'0':>10}")

print()
print("=" * 70)
print("  KEY INSIGHTS — SCENARIO 2")
print("=" * 70)
print()
print("  1. Production plan almost IDENTICAL to baseline.")
print("     The small demand shifts in Jan-Mar don't require strategy change.")
print()
print("  2. CV demand DECREASED: 28.22% → 26.30%")
print("     Promotion in a low-demand month makes demand MORE stable.")
print()
print("  3. Cumulative inventory DECREASED: 1,600 → 1,360 unit-months")
print("     Lower inventory in Jan & Feb → lower holding cost.")
print()
print(f"  4. Profit loss: ${profit-prof1:,.0f} (only from discount given)")
print(f"     = {demand_s2[0]:,} units × $5 discount = ${discount_loss:,.0f} revenue loss")
print(f"     Partially offset by $480 holding cost savings")
print()
print("  5. The promotion 'buys' demand stability at a cost of $8,320.")
print("     Whether that's worth it depends on business context.")
print()
print("  → NEXT: 07_scenario3_promo_apr.py")
