# Interactive q-Voter Model Simulation for Customer Churn

## 1. STREAMLIT APP LINK

https://q-voter-app-app-54xgh5gxxrr4pukoxwfmwe.streamlit.app/

https://q-voter-app-app-54xgh5gxxrr4pukoxwfmwe.streamlit.app/

https://q-voter-app-app-54xgh5gxxrr4pukoxwfmwe.streamlit.app/

## 2. Project Description

This project presents an interactive web application designed to simulate and visualize customer churn dynamics. It uses an Agent-Based Modeling (ABM) approach via a modified **q-voter model** mapped onto a **Watts-Strogatz small-world network**. 

The primary goal of the simulation is to explore how customers churn in the face of episodic (un)successful ad campaigns, which are sent into the network at specific intervals. The application allows users to interactively adjust network topology, agent influences, and ad campaign effectiveness to see real-time Monte Carlo simulation results.

The project features:
- Agent-Based Modeling and Monte Carlo simulations.
- Interactive parameter tuning via a Streamlit web interface.
- High-performance parallelized computations using Numba.
- Network generation and topological adjustments (NetworkX).
- Real-time calculations of Mean and 95% Confidence Intervals for churn rates.

## 3. Model & Theoretical Background

Instead of a static dataset, this project generates complex dynamic data based on theoretical socio-physics and marketing research.

**Underlying Mechanics:**
- **Watts-Strogatz Graph:** Represents the social network of customers. The rewiring probability ($\beta$) allows exploration from regular ring lattices to random networks.
- **Modified q-Voter Model:** - An agent looks at $q$ random neighbors. If all $q$ neighbors share the same opinion, the agent adopts it.
  - If opinions differ, the agent acts independently and may "flip" based on their personal baseline sensitivity.
- **Advertising Impact:** Based on literature regarding online ad campaigns, episodic ads impact a random subset of agents (5-20% success rate). Exposure to a successful ad dynamically alters an agent's flipping threshold, making them harder to churn or easier to win back.

## 4. Tech Stack and Methodology

### Tech Stack

- **Python**
- **Streamlit** — Web application framework for interactive UI.
- **Numba** — JIT compilation and parallelization (`@njit(parallel=True)`) for high-speed Monte Carlo simulations.
- **NetworkX** — Graph theory and complex network generation.
- **NumPy** — Fast numerical operations and array manipulations.
- **Matplotlib** — Plotting and data visualization.

### Methodology

The workflow follows these main steps:

1. **Network Initialization:** Generation of a Watts-Strogatz graph based on user-defined parameters ($N$, $k$, $\beta$).
2. **Translation for Numba:** Converting the NetworkX graph into 2D NumPy arrays (neighbor matrices and degree arrays) to allow C-level execution speeds.
3. **Agent State Initialization:** Assigning baseline ad sensitivities and initial churn states to $N$ agents.
4. **Monte Carlo Simulation:** - Episodic ad injection at 10% step intervals.
   - Individual agent updates based on $q$ neighbors' states and dynamic flip probabilities.
5. **Parallel Execution:** Running $X$ number of simulations simultaneously to build statistical robustness.
6. **Metrics Aggregation:** Calculating the mean churn rate and empirical 95% Confidence Intervals (2.5th and 97.5th percentiles) across all parallel runs.
7. **Interactive Visualization:** Rendering the aggregated data back to the user via Streamlit.

## 5. Repository Contents

```text
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── q-voter_projekt.ipynb   # Original exploratory Jupyter Notebook
└── README.md               # Project documentation
