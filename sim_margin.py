import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# Fixed sample size
SAMPLE_SIZE = 1000

# Function to calculate margins
@st.cache_data
def calculate_margins(eval_price, discount_pct, purchase_to_payout_rate, avg_payout):
    revenue_eval = eval_price * SAMPLE_SIZE
    discounted_eval_price = eval_price * (1 - discount_pct)
    revenue_discounted_eval = discounted_eval_price * SAMPLE_SIZE
    cost = purchase_to_payout_rate * avg_payout * SAMPLE_SIZE
    net_revenue = revenue_eval - cost
    net_discounted_revenue = revenue_discounted_eval - cost
    price_margin = net_revenue / revenue_eval if revenue_eval > 0 else 0
    discounted_price_margin = net_discounted_revenue / revenue_discounted_eval if revenue_discounted_eval > 0 else 0
    return price_margin, discounted_price_margin

# Function to compute margins for a variable
@st.cache_data
def compute_margins_for_variable(var_name, var_values, eval_price, discount_pct, purchase_to_payout_rate, avg_payout):
    price_margins = []
    discounted_margins = []
    for value in var_values:
        if var_name == "Eval Price":
            pm, dpm = calculate_margins(value, discount_pct, purchase_to_payout_rate, avg_payout)
        elif var_name == "Discount %":
            pm, dpm = calculate_margins(eval_price, value, purchase_to_payout_rate, avg_payout)
        elif var_name == "Purchase to Payout Rate":
            pm, dpm = calculate_margins(eval_price, discount_pct, value, avg_payout)
        elif var_name == "Avg Payout":
            pm, dpm = calculate_margins(eval_price, discount_pct, purchase_to_payout_rate, value)
        price_margins.append(pm)
        discounted_margins.append(dpm)
    return price_margins, discounted_margins

# Function to compute aggregated margins for Purchase to Payout Rate and Avg Payout
@st.cache_data
def compute_aggregated_margins(purchase_to_payout_vars, avg_payout_vars, eval_price, discount_pct):
    price_margins = []
    discounted_margins = []
    labels = []
    for ptr, ap in zip(purchase_to_payout_vars, avg_payout_vars):
        pm, dpm = calculate_margins(eval_price, discount_pct, ptr, ap)
        price_margins.append(pm)
        discounted_margins.append(dpm)
        labels.append(f"Ptr: {ptr*100:.2f}%, Avg Payout: ${ap:.2f}")
    return price_margins, discounted_margins, labels

# Function to find the first point where margin falls below 50%
def find_margin_threshold(var_name, var_values, price_margins, discounted_margins):
    price_threshold = None
    discounted_threshold = None
    for i, (pm, dpm) in enumerate(zip(price_margins, discounted_margins)):
        if pm <= 0.5 and price_threshold is None:
            price_threshold = var_values[i]
        if dpm <= 0.5 and discounted_threshold is None:
            discounted_threshold = var_values[i]
        if price_threshold is not None and discounted_threshold is not None:
            break
    return price_threshold, discounted_threshold

# Streamlit app
st.set_page_config(page_title="Margin Simulator", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .stSlider > div > div > div > div {
        background-color: #488BF8;
    }
    .stButton > button {
        background-color: #F2CB80;
        color: #0E1117;
    }
    .stDownloadButton > button {
        background-color: #F2CB80;
        color: #0E1117;
    }
    h1, h2, h3 {
        color: #488BF8;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #262730;
        border-radius: 20px;
        padding: 8px 16px;
        color: #FAFAFA;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #488BF8;
        color: #FAFAFA;
    }
    </style>
""", unsafe_allow_html=True)

# Introduction
with st.expander("Welcome to the Margin Simulator", expanded=True):
    st.markdown("""
        This app helps you simulate profit margins for different account sizes by adjusting key variables like Discount %, Eval Pass Rate, Sim Funded Rate, and Avg Payout. 
        - Select an account size to set the Eval Price and default Avg Payout.
        - Adjust other parameters in the sidebar.
        - Explore individual and aggregated simulations to understand margin impacts using the tabs below.
        - Download results for further analysis.
    """)

# Account size profiles with fixed Eval Price and preset Avg Payout
account_sizes = {
    "25k": {"eval_price": 150.0, "avg_payout": 750.0},
    "50k": {"eval_price": 170.0, "avg_payout": 1250.0},
    "75k": {"eval_price": 245.0, "avg_payout": 2000.0},
    "100k": {"eval_price": 330.0, "avg_payout": 2500.0},
    "150k": {"eval_price": 360.0, "avg_payout": 3750.0}
}

st.sidebar.header("Input Parameters")
st.sidebar.markdown("**Account Size**: Select the account size to set the Eval Price and default Avg Payout.")
account_size = st.sidebar.selectbox("Account Size", list(account_sizes.keys()), index=0, key="account_size")
eval_price = account_sizes[account_size]["eval_price"]  # Fixed Eval Price based on selection
default_avg_payout = account_sizes[account_size]["avg_payout"]  # Default Avg Payout based on selection
st.sidebar.write(f"Eval Price (fixed): ${eval_price:.2f}")

st.sidebar.markdown("**Discount %**: Percentage discount applied to Eval Price (0-100%).")
discount_pct = st.sidebar.number_input("Discount % (e.g., 30 for 30%)", min_value=0.0, max_value=100.0, value=30.0, step=1.0) / 100

st.sidebar.markdown("**Eval Pass Rate**: Percentage of evaluations that pass (0-100%).")
eval_pass_rate = st.sidebar.number_input("Eval Pass Rate (e.g., 27.01 for 27.01%)", min_value=0.0, max_value=100.0, value=27.01, step=0.01) / 100

st.sidebar.markdown("**Sim Funded Rate**: Percentage of passed evaluations that lead to payouts (0-100%).")
sim_funded_rate = st.sidebar.number_input("Sim Funded to Payout Rate (e.g., 4.8 for 4.8%)", min_value=0.0, max_value=100.0, value=4.8, step=0.01) / 100

st.sidebar.markdown("**Avg Payout**: Average payout amount per funded account (default set by account size).")
avg_payout = st.sidebar.number_input("Avg. Payout Amount", min_value=0.0, value=default_avg_payout, step=1.0)

# Calculate base values
discounted_eval_price = eval_price * (1 - discount_pct)
purchase_to_payout_rate = eval_pass_rate * sim_funded_rate
base_pm, base_dpm = calculate_margins(eval_price, discount_pct, purchase_to_payout_rate, avg_payout)

# Display base calculation
st.header("Base Calculation")
col1, col2, col3 = st.columns(3)
with col1:
    st.write(f"**Account Size:** {account_size}")
    st.write(f"**Eval Price:** ${eval_price:.2f}")
with col2:
    st.write(f"**Calculated Discounted Eval Price:** ${discounted_eval_price:.2f}")
    st.write(f"**Purchase to Payout Rate:** {purchase_to_payout_rate*100:.4f}%")
with col3:
    st.write(f"**Price Margin:** {base_pm:.4f} ({base_pm*100:.2f}%)")
    st.write(f"**Discounted Price Margin:** {base_dpm:.4f} ({base_dpm*100:.2f}%)")

# Dynamic range selection
st.sidebar.header("Simulation Ranges")
max_eval_price_decrease = st.sidebar.slider("Max Eval Price Decrease (%)", 10, 100, 50) / 100
max_discount_increase = st.sidebar.slider("Max Discount % Increase (absolute)", 10, 70, 50) / 100
max_ptr_increase = st.sidebar.slider("Max Purchase to Payout Rate Increase (%)", 100, 1000, 500) / 100
max_avg_payout_increase = st.sidebar.slider("Max Avg Payout Increase (%)", 100, 500, 200) / 100

# Variation ranges with 20 steps
eval_price_vars = [eval_price * (1 - x) for x in np.linspace(0, max_eval_price_decrease, 20)]
discount_pct_vars = [max(0, min(1, discount_pct + (max_discount_increase * x))) for x in np.linspace(0, 1, 20)]
purchase_to_payout_rate_vars = [max(0, min(1, purchase_to_payout_rate * (1 + max_ptr_increase * x))) for x in np.linspace(0, 1, 20)]
avg_payout_vars = [avg_payout * (1 + max_avg_payout_increase * x) for x in np.linspace(0, 1, 20)]

variables = [
    ("Eval Price", eval_price_vars),
    ("Discount %", discount_pct_vars),
    ("Purchase to Payout Rate", purchase_to_payout_rate_vars),
    ("Avg Payout", avg_payout_vars)
]

# Tabbed interface for simulations
st.header("Simulations")
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Eval Price", "Discount %", "Purchase to Payout Rate", "Avg Payout", 
    "Aggregated (Ptr vs Avg Payout)", "Combined", "Extreme Cases"
])

# Individual simulations with Plotly (Tabs 1-4)
for idx, (tab, (var_name, var_values)) in enumerate(zip([tab1, tab2, tab3, tab4], variables)):
    with tab:
        st.subheader(f"{var_name} Simulation")
        price_margins, discounted_margins = compute_margins_for_variable(
            var_name, var_values, eval_price, discount_pct, purchase_to_payout_rate, avg_payout
        )
        
        # Plotly figure (Line Chart with Dual Axes)
        fig = go.Figure()
        x_values = [v * 100 if var_name in ["Discount %", "Purchase to Payout Rate"] else v for v in var_values]
        unit = "%" if var_name in ["Discount %", "Purchase to Payout Rate"] else "$"
        fig.add_trace(go.Scatter(
            x=x_values, y=price_margins, mode='lines+markers', name=var_name,
            line=dict(color="#488BF8"), marker=dict(size=8, line=dict(width=2, color='DarkSlateGrey')),
            hovertemplate=f'{var_name}: {unit}%{{x:.2f}}<br>Margin: %{{y:.4f}} (%{{y:.2%}})'
        ))
        fig.add_trace(go.Scatter(
            x=x_values, y=discounted_margins, mode='lines+markers', name="Discounted Margin",
            line=dict(color="#A3C1FA"), marker=dict(size=8, line=dict(width=2, color='DarkSlateGrey')),
            hovertemplate=f'{var_name}: {unit}%{{x:.2f}}<br>Margin: %{{y:.4f}} (%{{y:.2%}})'
        ))
        fig.add_hline(y=0.5, line_dash="dash", line_color="#F2CB80", annotation_text="50% Threshold")
        # Add vertical lines for thresholds
        price_threshold, discounted_threshold = find_margin_threshold(var_name, x_values, price_margins, discounted_margins)
        if price_threshold is not None:
            fig.add_vline(x=price_threshold, line_dash="dash", line_color="#F2CB80", annotation_text=f"{var_name} < 50%")
        if discounted_threshold is not None:
            fig.add_vline(x=discounted_threshold, line_dash="dash", line_color="#F9E4B7", annotation_text="Discounted < 50%")
        fig.update_layout(
            title=f"Effect of {var_name} on Margins",
            xaxis_title=f"{var_name} ({'%' if var_name in ['Discount %', 'Purchase to Payout Rate'] else '$'})",
            yaxis_title="Margin",
            yaxis_range=[-1.5, 1.2],
            hovermode="x unified",
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#FAFAFA"),
            xaxis=dict(gridcolor='LightGrey'),
            yaxis=dict(gridcolor='LightGrey')
        )
        st.plotly_chart(fig, use_container_width=True)

        # Display threshold points
        if price_threshold is not None:
            st.write(f"**{var_name} Margin falls to or below 50% at:** {price_threshold:.2f}{' %' if var_name in ['Discount %', 'Purchase to Payout Rate'] else ''}")
        else:
            st.write(f"**{var_name} Margin stays above 50% across the range.**")
        if discounted_threshold is not None:
            st.write(f"**Discounted Margin falls to or below 50% at:** {discounted_threshold:.2f}{' %' if var_name in ['Discount %', 'Purchase to Payout Rate'] else ''}")
        else:
            st.write(f"**Discounted Margin stays above 50% across the range.**")

        # Download simulation data
        df = pd.DataFrame({
            f"{var_name} ({'%' if var_name in ['Discount %', 'Purchase to Payout Rate'] else ''})": x_values,
            "Price Margin": price_margins,
            "Discounted Margin": discounted_margins
        })
        csv = df.to_csv(index=False)
        st.download_button(f"Download {var_name} Simulation Data", csv, f"{var_name.lower().replace(' ', '_')}_simulation.csv", "text/csv")

# Aggregated simulation for Purchase to Payout Rate and Avg Payout (Tab 5)
with tab5:
    st.subheader("Aggregated Simulation: Purchase to Payout Rate vs Avg Payout")
    # Use the same variation ranges as individual simulations
    price_margins, discounted_margins, labels = compute_aggregated_margins(
        purchase_to_payout_rate_vars, avg_payout_vars, eval_price, discount_pct
    )

    # Grouped Bar Chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=price_margins,
        name="Price Margin",
        marker_color="#488BF8",
        hovertemplate='%{x}<br>Margin: %{y:.4f} (%{y:.2%})'
    ))
    fig.add_trace(go.Bar(
        x=labels,
        y=discounted_margins,
        name="Discounted Margin",
        marker_color="#A3C1FA",
        hovertemplate='%{x}<br>Margin: %{y:.4f} (%{y:.2%})'
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="#F2CB80", annotation_text="50% Threshold")
    fig.update_layout(
        title="Purchase to Payout Rate vs Avg Payout",
        xaxis_title="Purchase to Payout Rate and Avg Payout Combinations",
        yaxis_title="Margin",
        yaxis_range=[-1.5, 1.2],
        barmode='group',
        hovermode="x unified",
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#FAFAFA"),
        xaxis=dict(gridcolor='LightGrey', tickangle=45),
        yaxis=dict(gridcolor='LightGrey')
    )
    st.plotly_chart(fig, use_container_width=True)

    # Find and display threshold points for aggregated simulation
    progression_values = np.linspace(0, 100, 20)
    price_threshold, discounted_threshold = find_margin_threshold("Combined (Ptr & Avg Payout)", progression_values, price_margins, discounted_margins)
    if price_threshold is not None:
        st.write(f"**Combined (Ptr & Avg Payout) Price Margin falls to or below 50% at progression:** {price_threshold:.2f}%")
    else:
        st.write(f"**Combined (Ptr & Avg Payout) Price Margin stays above 50% across the range.**")
    if discounted_threshold is not None:
        st.write(f"**Combined (Ptr & Avg Payout) Discounted Margin falls to or below 50% at progression:** {discounted_threshold:.2f}%")
    else:
        st.write(f"**Combined (Ptr & Avg Payout) Discounted Margin stays above 50% across the range.**")

    # Download aggregated simulation data
    df_agg = pd.DataFrame({
        "Combination": labels,
        "Price Margin": price_margins,
        "Discounted Margin": discounted_margins
    })
    csv_agg = df_agg.to_csv(index=False)
    st.download_button("Download Aggregated Simulation Data", csv_agg, "aggregated_simulation.csv", "text/csv")

# Combined simulation with user-selected variables (Tab 6)
with tab6:
    st.subheader("Combined Simulation")
    st.write("Select percentage changes for each variable (0% for no change):")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        eval_price_change = -st.slider("Eval Price Decrease (%)", 0, 50, 0, step=5) / 100
    with col2:
        discount_pct_change = st.slider("Discount % Increase (%)", 0, 50, 0, step=5) / 100
    with col3:
        eval_pass_rate_change = st.slider("Eval Pass Rate Increase (%)", 0, 200, 0, step=10) / 100
    with col4:
        sim_funded_rate_change = st.slider("Sim Funded Rate Increase (%)", 0, 200, 0, step=10) / 100
    with col5:
        avg_payout_change = st.slider("Avg Payout Increase (%)", 0, 200, 0, step=10) / 100

    # Calculate combined scenario
    combined_eval_price = eval_price * (1 + eval_price_change)
    combined_discount_pct = min(1.0, discount_pct * (1 + discount_pct_change))
    combined_purchase_to_payout_rate = min(1.0, eval_pass_rate * (1 + eval_pass_rate_change) * sim_funded_rate * (1 + sim_funded_rate_change))
    combined_avg_payout = avg_payout * (1 + avg_payout_change)

    combined_pm, combined_dpm = calculate_margins(
        combined_eval_price, combined_discount_pct, combined_purchase_to_payout_rate, combined_avg_payout
    )

    st.subheader("Combined Scenario Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"Eval Price: ${combined_eval_price:.2f}")
        st.write(f"Discount %: {combined_discount_pct*100:.2f}%")
    with col2:
        st.write(f"Purchase to Payout Rate: {combined_purchase_to_payout_rate*100:.4f}%")
        st.write(f"Avg Payout: ${combined_avg_payout:.2f}")
    with col3:
        st.write(f"**Price Margin:** {combined_pm:.4f} ({combined_pm*100:.2f}%)")
        st.write(f"**Discounted Price Margin:** {combined_dpm:.4f} ({combined_dpm*100:.2f}%)")
    if combined_pm >= 0.5 and combined_dpm >= 0.5:
        st.success("Both margins are above 50%!")
    elif combined_pm < 0.5 or combined_dpm < 0.5:
        st.warning("One or both margins are below 50%.")

# Extreme cases for each variable (Tab 7)
with tab7:
    st.subheader("Extreme Case Scenarios")
    st.write("Each scenario uses the extreme value for one variable while keeping others at base values:")
    extreme_scenarios = [
        ("Eval Price (-50%)", eval_price * 0.5, discount_pct, purchase_to_payout_rate, avg_payout),
        ("Discount % (+50%)", eval_price, min(1.0, discount_pct + 0.5), purchase_to_payout_rate, avg_payout),
        ("Purchase to Payout Rate (+500%)", eval_price, discount_pct, min(1.0, purchase_to_payout_rate * 6), avg_payout),
        ("Avg Payout (+200%)", eval_price, discount_pct, purchase_to_payout_rate, avg_payout * 3)
    ]

    data = []
    for name, ep, dp, ptr, ap in extreme_scenarios:
        pm, dpm = calculate_margins(ep, dp, ptr, ap)
        data.append([name, f"${ep:.2f}", f"{dp*100:.2f}%", f"{ptr*100:.4f}%", f"${ap:.2f}", f"{pm:.4f} ({pm*100:.2f}%)", f"{dpm:.4f} ({dpm*100:.2f}%)"])

    st.table({
        "Scenario": [row[0] for row in data],
        "Eval Price": [row[1] for row in data],
        "Discount %": [row[2] for row in data],
        "Purchase to Payout Rate": [row[3] for row in data],
        "Avg Payout": [row[4] for row in data],
        "Price Margin": [row[5] for row in data],
        "Discounted Price Margin": [row[6] for row in data]
    })
