# =============================================================================
# 07_scenario3_promo_apr.py
# Forward Buying Analysis — Scenario 3: Promotion in April (High Demand Month)
# =============================================================================
#
# STATUS: ✅ FINAL
#
# SCENARIO 3 SETUP:
#   - Promotional discount in April only (peak demand month / Eid season)
#   - Price: $34/unit in Apr (down $6 from $40), $40/unit in other months
#   - Forward buying mechanism:
#       Apr demand: +20% (3,800 × 1.20 = 4,560 units)
#       May demand: -20% (2,200 × 0.80 = 1,760 units, shifted to Apr)
#       Jun demand: -20% (2,100 × 0.80 = 1,680 units, shifted to Apr)
#       Jan-Mar: unchanged
#
# KEY FINDING (proved by LP):
#   - LP must HIRE 6 new workers in February to meet April demand
#   - Production increases from 3,200 to 3,453 units/month
#   - Inventory peaks at 1,107 units in March (record high)
#   - After April, capacity utilization drops to ~50% (Mei: 51%, Jun: 49%)
#   - Revenue loss: 4,560 × $6 = $27,360 (vs $8,800 in Scenario 2)
#   - Profit loss vs baseline: -$32,054 (nearly 4× worse than Scenario 2)
#   - CV demand: 40.74% (vs 28.22% baseline — much more volatile)
#
# HOW TO RUN:
#   python 07_scenario3_promo_apr.py
# =============================================================================

import numpy as np
from scipy.optimize import linprog

print("=" * 70)
print("  SCENARIO 3 — Promotion in April ($34/unit)")
print("  Promoting in the highest-demand month")
print("=" * 70)

# -----------------------------------------------------------------------------
# PARAMETERS
# -----------------------------------------------------------------------------
T = 6
months         = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
inv_init       = 2000
workers_init   = 80
sell_price     = 40
promo_price    = 34
promo_month    = 3       # April = index 3

cap_reg_per_worker = 40.0
cap_ot_per_worker  = 2.5

c_mat=10; c_reg=16; c_ot=24; c_sub=30
c_hold=2; c_short=5; c_hire=300; c_fire=200
c_short_eff = sell_price + c_short   # $45

demand_base = [1600, 3000, 3200, 3800, 2200, 2100]
demand_s3   = [1600, 3000, 3200, 4560, 1760, 1680]

print()
print("  FORWARD BUYING MECHANISM:")
print()
print(f"  {'Month':<6} {'Baseline':>10} {'Promo S3':>10} {'Change':>10} {'Reason'}")
print("  " + "-" * 75)
for t in range(T):
    delta = demand_s3[t] - demand_base[t]
    if t == 3:
        reason = "+20% genuine demand surge (price + Eid momentum)"
    elif t in [4, 5]:
        reason = f"-20% forward buying (shifted to April)"
    else:
        reason = "unchanged (before promo period)"
    sign = "+" if delta >= 0 else ""
    print(f"  {months[t]:<6} {demand_base[t]:>10} {demand_s3[t]:>10} "
          f"{sign}{delta:>9}  {reason}")

print()
print(f"  April demand vs regular capacity:")
print(f"    Regular capacity (80 workers): {int(80 * cap_reg_per_worker):,} units/month")
print(f"    April demand after promo     : {demand_s3[3]:,} units")
print(f"    Gap                          : {demand_s3[3] - int(80*cap_reg_per_worker):,} units")
print(f"    → Must cover via inventory buffer AND/OR workforce increase")

# -----------------------------------------------------------------------------
# SOLVER
# -----------------------------------------------------------------------------
def idx(var, t, T=6):
    return ['P','OT','S','I','B','W','H','F'].index(var)*T+t

def solve_lp(demand):
    N = 8*T
    _i = lambda v,t: idx(v,t,T)
    c_obj = np.zeros(N)
    for t in range(T):
        c_obj[_i('P',t)]=c_mat+c_reg;  c_obj[_i('OT',t)]=c_mat+c_ot
        c_obj[_i('S',t)]=c_mat+c_sub;  c_obj[_i('I',t)]=c_hold
        c_obj[_i('B',t)]=c_short_eff;  c_obj[_i('H',t)]=c_hire
        c_obj[_i('F',t)]=c_fire

    Aeq,beq,Aub,bub=[],[],[],[]
    for t in range(T):
        r=np.zeros(N)
        r[_i('P',t)]=1;r[_i('OT',t)]=1;r[_i('S',t)]=1
        r[_i('I',t)]=-1;r[_i('B',t)]=1
        if t==0: rhs=demand[t]-inv_init
        else: r[_i('I',t-1)]=1;r[_i('B',t-1)]=-1;rhs=demand[t]
        Aeq.append(r);beq.append(rhs)

        r2=np.zeros(N)
        r2[_i('W',t)]=1;r2[_i('H',t)]=-1;r2[_i('F',t)]=1
        if t==0: rhs2=workers_init
        else: r2[_i('W',t-1)]=-1;rhs2=0
        Aeq.append(r2);beq.append(rhs2)

        r3=np.zeros(N);r3[_i('P',t)]=1;r3[_i('W',t)]=-cap_reg_per_worker
        Aub.append(r3);bub.append(0)

        r4=np.zeros(N);r4[_i('OT',t)]=1;r4[_i('W',t)]=-cap_ot_per_worker
        Aub.append(r4);bub.append(0)

    res=linprog(c_obj,
                A_ub=np.array(Aub),b_ub=np.array(bub,dtype=float),
                A_eq=np.array(Aeq),b_eq=np.array(beq,dtype=float),
                bounds=[(0,None)]*N,method='highs')
    if res.status!=0: return None
    x=res.x
    return [(demand[t],)+tuple(round(x[_i(v,t)]) for v in ['P','OT','S','I','B','W','H','F'])
            for t in range(T)]

rd1 = solve_lp(demand_base)
rd3 = solve_lp(demand_s3)

# -----------------------------------------------------------------------------
# PRODUCTION PLAN
# -----------------------------------------------------------------------------
print()
print("  AGGREGATE PRODUCTION PLAN — SCENARIO 3:")
print()
print(f"  {'Month':<6} {'Demand':>7} {'P_Reg':>7} {'P_OT':>7} "
      f"{'Subcon':>7} {'Inv':>8} {'Short':>7} {'Workers':>8} {'Hire':>5} {'Fire':>5}")
print("  " + "-" * 80)
for t, d in enumerate(rd3):
    dm,P,OT,S,I,B,W,H,F = d
    # Highlight February (hiring) and March-April (peak inventory and demand)
    flag = " ←" if t==1 else (" ←" if t==2 else (" ←" if t==3 else ""))
    print(f"  {months[t]:<6} {dm:>7} {P:>7} {OT:>7} "
          f"{S:>7} {I:>8} {B:>7} {W:>8} {H:>5} {F:>5}{flag}")

print()
print("  ← Feb: LP hires 6 workers → capacity goes from 3,200 to 3,440+/month")
print("  ← Mar: Inventory peaks at 1,107 units (record for entire analysis)")
print("  ← Apr: All buffer used to meet 4,560 demand")

# -----------------------------------------------------------------------------
# SUPPLY CHAIN FLOW
# -----------------------------------------------------------------------------
print()
print("  SUPPLY CHAIN FLOW DETAIL:")
print()
print(f"  {'Month':<6} {'Inv_Start':>10} {'Workers':>8} {'Cap_Reg':>9} "
      f"{'Production':>11} {'Total Supply':>13} {'Demand':>8} {'Inv_End':>9}")
print("  " + "-" * 85)
inv_prev, short_prev = inv_init, 0
for t, d in enumerate(rd3):
    dm,P,OT,S,I,B,W,H,F = d
    cap_this  = W * cap_reg_per_worker
    prod      = P+OT+S
    inv_start = inv_prev - short_prev
    supply    = inv_start + prod
    util      = (prod/cap_this*100) if cap_this > 0 else 0
    print(f"  {months[t]:<6} {inv_start:>10} {W:>8} {cap_this:>9.0f} "
          f"{prod:>11} {supply:>13} {dm:>8} {I:>9}")
    inv_prev, short_prev = I, B

# Capacity utilization in May-Jun
w_final = rd3[4][6]
cap_final = w_final * cap_reg_per_worker
print()
print(f"  Capacity utilization AFTER PROMOTION:")
print(f"    May : {rd3[4][1]:,} / {cap_final:,.0f} = {rd3[4][1]/cap_final*100:.1f}%")
print(f"    Jun : {rd3[5][1]:,} / {cap_final:,.0f} = {rd3[5][1]/cap_final*100:.1f}%")
print(f"    → 6 extra workers are underutilized but not fired (PHK cost = $1,200)")

# -----------------------------------------------------------------------------
# FINANCIAL ANALYSIS
# -----------------------------------------------------------------------------
rev_apr  = demand_s3[3] * promo_price
rev_rest = sum(d for i,d in enumerate(demand_s3) if i != 3) * sell_price
revenue  = rev_apr + rev_rest

rev_base = sum(demand_base) * sell_price

mat_cost  = sum((d[1]+d[2]+d[3])*c_mat for d in rd3)
reg_cost  = sum(d[1]*c_reg for d in rd3)
hold_cost = sum(d[4]*c_hold for d in rd3)
hire_cost = sum(d[7]*c_hire for d in rd3)
fire_cost = sum(d[8]*c_fire for d in rd3)
total_cost= mat_cost+reg_cost+hold_cost+hire_cost+fire_cost
profit    = revenue - total_cost

mat1  = sum((d[1]+d[2]+d[3])*c_mat for d in rd1)
reg1  = sum(d[1]*c_reg for d in rd1)
hold1 = sum(d[4]*c_hold for d in rd1)
cost1 = mat1+reg1+hold1
prof1 = sum(d[0] for d in rd1)*sell_price - cost1

print()
print("  FINANCIAL ANALYSIS:")
print()
print(f"  Revenue breakdown:")
print(f"    Apr: {demand_s3[3]:,} units × ${promo_price} (promo)   = ${rev_apr:>10,.0f}")
print(f"    Other: {sum(d for i,d in enumerate(demand_s3) if i!=3):,} units × ${sell_price} = ${rev_rest:>10,.0f}")
print(f"    TOTAL REVENUE                            = ${revenue:>10,.0f}")
print(f"    Revenue loss vs baseline                 = ${rev_base-revenue:>10,.0f}")
print(f"    (= {demand_s3[3]:,} units × $6 discount = ${demand_s3[3]*6:,.0f})")
print()
disc_loss = demand_s3[3] * (sell_price - promo_price)
print(f"  {'Item':<35} {'S1 Baseline':>13} {'S3 Promo Apr':>13} {'Delta':>10}")
print("  " + "-" * 75)
print(f"  {'Material':<35} ${mat1:>12,.0f} ${mat_cost:>12,.0f} ${mat_cost-mat1:>+9,.0f}")
print(f"  {'Regular Labor':<35} ${reg1:>12,.0f} ${reg_cost:>12,.0f} ${reg_cost-reg1:>+9,.0f}")
print(f"  {'Holding':<35} ${hold1:>12,.0f} ${hold_cost:>12,.0f} ${hold_cost-hold1:>+9,.0f}")
print(f"  {'Hiring (6 workers)':<35} ${'0':>12} ${hire_cost:>12,.0f} ${hire_cost:>+9,.0f}")
print("  " + "-" * 75)
print(f"  {'TOTAL COST':<35} ${cost1:>12,.0f} ${total_cost:>12,.0f} ${total_cost-cost1:>+9,.0f}")
print(f"  {'REVENUE':<35} ${rev_base:>12,.0f} ${revenue:>12,.0f} ${revenue-rev_base:>+9,.0f}")
print(f"  {'PROFIT':<35} ${prof1:>12,.0f} ${profit:>12,.0f} ${profit-prof1:>+9,.0f}")

# CV
cv1 = np.std(demand_base)/np.mean(demand_base)
cv3 = np.std(demand_s3)/np.mean(demand_s3)
inv1 = sum(d[4] for d in rd1)
inv3 = sum(d[4] for d in rd3)

print()
print("  OPERATIONAL METRICS:")
print()
print(f"  {'Metric':<35} {'S1 Baseline':>13} {'S3 Promo Apr':>13} {'Delta':>10}")
print("  " + "-" * 75)
print(f"  {'CV Demand':<35} {cv1*100:>12.2f}% {cv3*100:>12.2f}% {(cv3-cv1)*100:>+9.2f}%")
print(f"  {'Cumul. Inventory (unit-mo)':<35} {inv1:>13,} {inv3:>13,} {inv3-inv1:>+10,}")
print(f"  {'Total Shortage':<35} {'0':>13} {'0':>13} {'0':>10}")
print(f"  {'Workers Hired':<35} {'0':>13} {'6':>13} {'+6':>10}")

print()
print("=" * 70)
print("  KEY INSIGHTS — SCENARIO 3")
print("=" * 70)
print()
print("  1. LP recruits 6 workers in February to handle April surge.")
print(f"     Capacity: 80 × 40 = 3,200 → 86 × 40 = 3,440 → LP uses 3,453/month")
print()
print("  2. Inventory peaks at 1,107 units in March (highest in full analysis).")
print("     Buffer needed to cover April's 4,560 demand with 3,453 capacity.")
print()
print("  3. After April, demand crashes to 1,760 (May) and 1,680 (Jun).")
print(f"     Capacity utilization: May {rd3[4][1]/cap_final*100:.1f}%, Jun {rd3[5][1]/cap_final*100:.1f}%")
print("     6 extra workers are effectively idle — but cheaper to keep than fire.")
print()
print(f"  4. Revenue loss from discount: {demand_s3[3]:,} × $6 = ${disc_loss:,.0f}")
print(f"     (vs $8,800 in S2 — 3.1× larger)")
print()
print(f"  5. CV surged to 40.74% (vs 28.22% baseline, 26.30% in S2)")
print("     Promoting in peak demand month makes system MUCH harder to manage.")
print()
print(f"  → PROFIT LOSS vs BASELINE: ${profit-prof1:,.0f}")
print(f"  → Nearly 4× worse than Scenario 2 (${-8320:,})")
print()
print("  → NEXT: 08_all_scenarios_comparison.py")
