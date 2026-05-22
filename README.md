# Forward Buying & Promotion Effect Analysis
### Supply Chain Management | Aggregate Production Planning with Linear Programming

---

## 📌 Project Overview

This project analyzes the effect of promotional pricing on supply chain demand patterns —
a phenomenon known as **Forward Buying**: when customers purchase more than their
immediate need due to a temporary price incentive, effectively shifting future demand
to the present.

The analysis uses **Linear Programming (LP)** to build an optimal aggregate production
plan under three scenarios:

| Scenario | Description |
|----------|-------------|
| **S1 — Baseline** | No promotion. Demand follows its natural pattern. |
| **S2 — Promo January** | Price discount in the lowest-demand month ($40 → $35/unit) |
| **S3 — Promo April** | Price discount in the highest-demand month ($40 → $34/unit) |

The key finding: **timing of promotion matters far more than the size of the discount.**

---

## 📂 Repository Structure

```
forward-buying-analysis/
│
├── README.md
│
├── data/
│   ├── demand_scenarios.csv        ← Demand data for all 3 scenarios
│   └── parameters.csv              ← Cost & capacity parameters
│
├── outputs/
│   └── (generated when scripts are run)
│
├── 01_setup_and_data.py            ← Environment check + parameter loading
├── 02_lp_iteration1_baseline.py    ← First LP attempt (naive model)
├── 03_lp_iteration2_improve.py     ← Improved model with firing cost
├── 04_lp_fix_shortage_cost.py      ← Fix: effective shortage cost
├── 05_scenario1_baseline_final.py  ← Scenario 1: clean final run + analysis
├── 06_scenario2_promo_jan.py       ← Scenario 2: promo January + analysis
├── 07_scenario3_promo_apr.py       ← Scenario 3: promo April + analysis
├── 08_all_scenarios_comparison.py  ← Full 3-scenario comparison + CSV output
└── 09_final_clean_model.py         ← Production-ready clean version
```

---

## 🧮 The Model

### Decision Variables (per month t, 6 months total)

| Variable | Description | Unit |
|----------|-------------|------|
| `P[t]` | Regular production | units |
| `OT[t]` | Overtime production | units |
| `S[t]` | Subcontracted production | units |
| `I[t]` | End-of-period inventory | units |
| `B[t]` | End-of-period shortage/backlog | units |
| `W[t]` | Active workforce | workers |
| `H[t]` | Workers hired | workers |
| `F[t]` | Workers fired | workers |

**Total: 8 variables × 6 months = 48 decision variables**

### Objective Function

Minimize total operational cost:

```
Z = Σ [ 26·P[t] + 34·OT[t] + 40·S[t]     ← production costs
      + 2·I[t] + 45·B[t]                   ← inventory & shortage
      + 300·H[t] + 200·F[t] ]              ← workforce costs
```

*Note: shortage cost = $5 (penalty) + $40 (lost revenue) = $45 effective*

### Constraints (72 total)

1. **Inventory balance**: `I[t-1] - B[t-1] + P[t] + OT[t] + S[t] - I[t] + B[t] = D[t]`
2. **Worker balance**: `W[t] - W[t-1] - H[t] + F[t] = 0`
3. **Regular capacity**: `P[t] ≤ W[t] × 40`
4. **OT capacity**: `OT[t] ≤ W[t] × 2.5`
5. **Non-negativity**: all variables ≥ 0

---

## 📊 Key Parameters

| Parameter | Value |
|-----------|-------|
| Selling price | $40/unit |
| Initial inventory | 2,000 units |
| Initial workforce | 80 workers |
| Regular hours/day | 8 hours |
| Working days/month | 20 days |
| Max overtime/worker/month | 10 hours |
| Labor hours per unit | 4 hours |
| Material cost | $10/unit |
| Regular labor cost | $4/hour → $16/unit |
| Overtime labor cost | $6/hour → $24/unit |
| Subcontract cost | $30/unit |
| Holding cost | $2/unit/month |
| Shortage cost | $5/unit/month |
| Hiring cost | $300/worker |
| Firing cost | $200/worker |

---

## 📈 Results Summary

| Metric | S1: Baseline | S2: Promo Jan | S3: Promo Apr |
|--------|-------------|---------------|---------------|
| CV Demand | 28.22% | **26.30%** ✓ | 40.74% ✗ |
| Revenue | $636,000 | $627,200 | $604,640 |
| Total Cost | $364,600 | $364,120 | $365,294 |
| **Profit** | **$271,400** | **$263,080** | **$239,346** |
| Δ Profit vs S1 | — | -$8,320 | -$32,054 |
| Cumulative Inventory | 1,600 | **1,360** ✓ | 2,360 ✗ |
| Total Shortage | 0 | 0 | 0 |
| Workers Added | 0 | 0 | +6 |

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install scipy numpy pandas
```

### 2. Run scripts in order
```bash
python 01_setup_and_data.py
python 02_lp_iteration1_baseline.py
python 03_lp_iteration2_improve.py
python 04_lp_fix_shortage_cost.py
python 05_scenario1_baseline_final.py
python 06_scenario2_promo_jan.py
python 07_scenario3_promo_apr.py
python 08_all_scenarios_comparison.py
python 09_final_clean_model.py
```

### 3. Or run only the final clean model
```bash
python 09_final_clean_model.py
```

---

## 🔍 Iteration History

This repo documents the full development process — including failed attempts and fixes.
Each script corresponds to a real iteration in the analysis:

| Script | Status | Key Issue Addressed |
|--------|--------|---------------------|
| `02_lp_iteration1_baseline.py` | ❌ Flawed | Solver chose shortage over production (cost imbalance) |
| `03_lp_iteration2_improve.py` | ⚠️ Partial | Added firing cost, but shortage cost still underpriced |
| `04_lp_fix_shortage_cost.py` | ✅ Fixed | Effective shortage cost = penalty + lost revenue ($45) |
| `05–08` | ✅ Final | Full 3-scenario analysis with correct model |
| `09` | ✅ Clean | Production-ready refactored version |

---

## 📚 Context

This analysis is part of a larger **Supply Chain Management** content series
(@povsupplychain on Instagram), exploring how concepts like demand forecasting,
aggregate planning, and forward buying work in real-world scenarios.

**Key concepts covered:**
- Forward Buying & Demand Shifting
- Aggregate Production Planning
- Linear Programming for Operations
- Coefficient of Variation (CV) as demand stability metric
- Sales & Operations Planning (S&OP)

---

## 🛠️ Tech Stack

- **Python 3.x**
- **scipy** — `linprog` with HiGHS solver for LP optimization
- **numpy** — matrix operations for constraint formulation
- **pandas** — data manipulation and output formatting

---

*Analysis developed iteratively — see individual scripts for full development history.*
