import streamlit as st
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import networkx as nx
from numba import njit, prange

st.set_page_config(page_title="Interactive q-voter Model", layout="wide") # streamlit

@njit
def mod_qvoter(neighbors_matrix, degree_array, states, q, p_std, p_inc, steps):
    N = len(states)
    churn_rates = np.empty(steps)

    ad_interval = max(1, steps // 10)
    ad_impacted = np.zeros(N, dtype=np.bool_)

    for step in range(steps):
        if step % ad_interval == 0:
            impact_rate = 0.05 + np.random.rand() * 0.15
            for i in range(N):
                ad_impacted[i] = (np.random.rand() < impact_rate)
        else:
            for i in range(N):
                ad_impacted[i] = False
        
        for _ in range(N):
            node = np.random.randint(0, N)
            deg = degree_array[node]
            if deg == 0:
                continue

            if ad_impacted[node]: 
                if states[node] == 1: 
                    current_p = p_std[node] / 2.0 
                else: 
                    current_p = p_inc 
            else:
                current_p = p_std[node] 

            if deg <= q:
                all_same = True
                first_state = states[neighbors_matrix[node, 0]]
                for i in range(1, deg):
                    if states[neighbors_matrix[node, i]] != first_state:
                        all_same = False
                        break
                        
                if all_same:
                    states[node] = first_state
                else:
                    if np.random.rand() < current_p:
                        states[node] = 1 - states[node]
            else:
                sampled = np.empty(q, dtype=np.int64)
                count = 0
                while count < q:
                    idx = np.random.randint(0, deg)
                    val = neighbors_matrix[node, idx]
                    
                    already_picked = False
                    for i in range(count):
                        if sampled[i] == val:
                            already_picked = True
                            break
                            
                    if not already_picked:
                        sampled[count] = val
                        count += 1
                        
                all_same = True
                first_state = states[sampled[0]]
                for i in range(1, q):
                    if states[sampled[i]] != first_state:
                        all_same = False
                        break
                        
                if all_same:
                    states[node] = first_state
                else:
                    if np.random.rand() < current_p:
                        states[node] = 1 - states[node]
                        
        churns = 0
        for i in range(N):
            if states[i] == 0:
                churns += 1
        churn_rates[step] = churns / N
        
    return churn_rates

@njit(parallel=True)
def parallel_mc_runs(neighbors_matrix, degree_array, q, p_inc, steps, num_simulations, N, initial_churn):
    all_churn_rates = np.empty((num_simulations, steps), dtype=np.float64)
   
    for sim in prange(num_simulations):
        states = np.empty(N, dtype=np.int64)
        for i in range(N):
            states[i] = 1 if np.random.rand() > initial_churn else 0
            
        p_std = np.random.uniform(0.05, 0.15, N)
        all_churn_rates[sim, :] = mod_qvoter(neighbors_matrix, degree_array, states, q, p_std, p_inc, steps)
        
    return all_churn_rates

@st.cache_data(show_spinner=False)
def get_graph_arrays(N, k, beta):
    G = nx.watts_strogatz_graph(N, k, beta)
    max_deg = max([d for n, d in G.degree()])
    
    neighbors_matrix = np.full((N, max_deg), -1, dtype=np.int64)
    degree_array = np.zeros(N, dtype=np.int64)
    
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        deg = len(neighbors)
        degree_array[node] = deg
        for i in range(deg):
            neighbors_matrix[node, i] = neighbors[i]
            
    return neighbors_matrix, degree_array

# UI on streamlit webpage
st.title("Interactive q-Voter Model Simulation")
st.markdown("""
Explore how customers churn in the face of episodic ad campaigns done by a company. This Monte Carlo simulation uses a modified q-voter model mapped onto a Watts-Strogatz graph.
* Adjust parameters in the sidebar to run custom simulations.
""")

st.sidebar.header("Simulation Parameters")
N = st.sidebar.number_input("Number of Nodes (N)", min_value=1000, max_value=100000, value=20000, step=5000)
k = st.sidebar.slider("Nearest Neighbors (k)", min_value=2, max_value=20, value=8, step=2)
q = st.sidebar.slider("q Neighbours (q)", min_value=1, max_value=10, value=4, step=1)
p_inc = st.sidebar.slider("Boosted Flipping Prob (p_inc)", min_value=0.0, max_value=1.0, value=0.40, step=0.05)
steps = st.sidebar.slider("Monte Carlo Steps (steps)", min_value=50, max_value=1000, value=300, step=50)

st.sidebar.subheader("Tab 1 Settings")
initial_churn = st.sidebar.slider("Initial Churn", min_value=0.0, max_value=1.0, value=0.30, step=0.05)
num_simulations = st.sidebar.slider("Num MC Simulations", min_value=5, max_value=200, value=50, step=5)

st.sidebar.subheader("Tab 2 Settings")
num_mc_runs = st.sidebar.slider("Runs per Concentration Line", min_value=1, max_value=50, value=5, step=1)

st.sidebar.subheader("Network Rewiring Probabilities (beta)")
default_betas = [0.0, 0.05, 0.2, 0.35, 0.5, 1.0]
beta_input = st.sidebar.text_input("Enter comma-separated beta values", value="0.0, 0.05, 0.2, 0.35, 0.5, 1.0")
try:
    betas = [float(b.strip()) for b in beta_input.split(',')]
except ValueError:
    st.sidebar.error("Invalid beta input")
    betas = default_betas


# plots
if st.sidebar.button("▶ Run Simulations", type="primary"):
    
    tab1, tab2 = st.tabs(["Churn rate per beta", "Loyal clients concentration"])
    
    with tab1:
        st.subheader(f"Customer Churn for different beta values")
        st.write(f"Mean & 95% confidence interval over {num_simulations} MC simulations")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        
        for idx, beta in enumerate(betas):
            status_text.text(f"Running simulation for beta = {beta}...")
            start_time = time.time()
            neighbors_matrix, degree_array = get_graph_arrays(N, k, beta)
            all_churn_rates = parallel_mc_runs(
                neighbors_matrix, degree_array, q, p_inc, steps, num_simulations, N, initial_churn
            )
            
            mean_churn = np.mean(all_churn_rates, axis=0)
            ci_lower = np.percentile(all_churn_rates, 2.5, axis=0)
            ci_upper = np.percentile(all_churn_rates, 97.5, axis=0)
            comp_time = time.time() - start_time
            
            p = ax1.plot(mean_churn, label=f'$\\beta={beta}$ (Time: {comp_time:.2f}s)', linewidth=2)
            color = p[0].get_color() 
            ax1.fill_between(range(steps), ci_lower, ci_upper, color=color, alpha=0.2)
            
            progress_bar.progress((idx + 1) / len(betas))
            
        ax1.set_xlabel('MC simulation steps', fontsize=12)
        ax1.set_ylabel('Churn rate', fontsize=12)
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig1)
        
        status_text.text("Churn simulations")
        
    with tab2:
        st.subheader("Concentration of loyal customers")
        initial_loyal_fractions = np.linspace(0, 1.0, 11)
        cols = st.columns(2) 
        
        for idx, beta in enumerate(betas):
            with cols[idx % 2]:
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                neighbors_matrix, degree_array = get_graph_arrays(N, k, beta)
                
                for c0 in initial_loyal_fractions:
                    initial_churn_val = 1.0 - c0 
                    all_churn_rates = parallel_mc_runs(
                        neighbors_matrix, degree_array, q, p_inc, steps, num_mc_runs, N, initial_churn_val
                    )
                    
                    mean_churn = np.mean(all_churn_rates, axis=0)
                    loyal_rates = 1.0 - mean_churn
                    
                    ax2.plot(range(steps), loyal_rates, 'ro--', markersize=5, markevery=steps//5, linewidth=2)

                ax2.set_title(f'Concentration of Loyal Clients ($\\beta={beta}$)\nMean of {num_mc_runs} MC runs', fontsize=14)
                ax2.set_xlabel('MCS', fontsize=12)
                ax2.set_ylabel('Concentration', fontsize=12)
                ax2.set_xlim(0, steps)
                ax2.set_ylim(0, 1.0)
                st.pyplot(fig2)