import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Fixed sample size
SAMPLE_SIZE = 1000

# Function to calculate margins
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
def compute_aggregated_margins(purchase_to_payout_vars, avg_payout_vars, eval_price, discount_pct):
    price_margins = []
    discounted_margins = []
    for ptr, ap in zip(purchase_to_payout_vars, avg_payout_vars):
        pm, dpm = calculate_margins(eval_price, discount_pct, ptr, ap)
        price_margins.append(pm)
        discounted_margins.append(dpm)
    return price_margins, discounted_margins

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
st.title("Interactive Margin Simulator")

# Account size profiles with fixed Eval Price
account_sizes = {
    "25k": 150.0,
    "50k": 170.0,
    "75k": 245.0,
    "100k": 330.0,
    "150k": 360.0
}

st.sidebar.header("Input Parameters")
account_size = st.sidebar.selectbox("Account Size", list(account_sizes.keys()), index=0)
eval_price = account_sizes[account_size]  # Fixed Eval Price based on selection
st.sidebar.write(f"Eval Price (fixed): ${eval_price:.2f}")
discount_pct = st.sidebar.number_input("Discount % (e.g., 30 for 30%)", min_value=0.0, max_value=100.0, value=30.0, step=1.0) / 100
eval_pass_rate = st.sidebar.number_input("Eval Pass Rate (e.g., 27.01 for 27.01%)", min_value=0.0, max_value=100.0, value=27.01, step=0.01) / 100
sim_funded_rate = st.sidebar.number_input("Sim Funded to Payout Rate (e.g., 4.8 for 4.8%)", min_value=0.0, max_value=100.0, value=4.8, step=0.01) / 100
avg_payout = st.sidebar.number_input("Avg. Payout Amount", min_value=0.0, value=750.0, step=1.0)

# Calculate base values
discounted_eval_price = eval_price * (1 - discount_pct)
purchase_to_payout_rate = eval_pass_rate * sim_funded_rate
base_pm, base_dpm = calculate_margins(eval_price, discount_pct, purchase_to_payout_rate, avg_payout)

# Display base calculation
st.header("Base Calculation")
st.write(f"**Account Size:** {account_size}")
st.write(f"**Eval Price:** ${eval_price:.2f}")
st.write(f"**Calculated Discounted Eval Price:** ${discounted_eval_price:.2f}")
st.write(f"**Purchase to Payout Rate:** {purchase_to_payout_rate*100:.4f}% (Eval Pass Rate * Sim Funded Rate)")
st.write(f"**Price Margin:** {base_pm:.4f} ({base_pm*100:.2f}%)")
st.write(f"**Discounted Price Margin:** {base_dpm:.4f} ({base_dpm*100:.2f}%)")

# Variation ranges with 20 steps
eval_price_vars = [eval_price * (1 - x) for x in np.linspace(0, 0.5, 20)]  # Base to -50%
discount_pct_vars = [max(0, min(1, discount_pct + (0.5 * x))) for x in np.linspace(0, 1, 20)]  # Base to 80%
purchase_to_payout_rate_vars = [max(0, min(1, purchase_to_payout_rate * (1 + 5 * x))) for x in np.linspace(0, 1, 20)]  # Base to 500% increase
avg_payout_vars = [avg_payout * (1 + 2 * x) for x in np.linspace(0, 1, 20)]  # Base to 200% increase

variables = [
    ("Eval Price", eval_price_vars),
    ("Discount %", discount_pct_vars),
    ("Purchase to Payout Rate", purchase_to_payout_rate_vars),
    ("Avg Payout", avg_payout_vars)
]

# Individual simulations with Plotly
st.header("Individual Variable Simulations")
for var_name, var_values in variables:
    price_margins, discounted_margins = compute_margins_for_variable(
        var_name, var_values, eval_price, discount_pct, purchase_to_payout_rate, avg_payout
    )
    st.subheader(f"{var_name} Simulation")
    
    # Plotly figure
    fig = go.Figure()
    x_values = [v * 100 if var_name in ["Discount %", "Purchase to Payout Rate"] else v for v in var_values]
    fig.add_trace(go.Scatter(x=x_values, y=price_margins, mode='lines+markers', name=var_name,
                             hovertemplate='%{x:.2f}<br>Margin: %{y:.4f} (%{y:.2%})'))
    fig.add_trace(go.Scatter(x=x_values, y=discounted_margins, mode='lines+markers', name="Discounted Margin",
                             hovertemplate='%{x:.2f}<br>Margin: %{y:.4f} (%{y:.2%})'))
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="50% Threshold")
    fig.update_layout(
        title=f"Effect of {var_name} on Margins",
        xaxis_title=f"{var_name} ({'%' if var_name in ['Discount %', 'Purchase to Payout Rate'] else ''})",
        yaxis_title="Margin",
        yaxis_range=[0, 1.2],
        hovermode="x unified"
    )
    st.plotly_chart(fig)

    # Find and display threshold points
    price_threshold, discounted_threshold = find_margin_threshold(var_name, x_values, price_margins, discounted_margins)
    if price_threshold is not None:
        st.write(f"**{var_name} Margin falls to or below 50% at:** {price_threshold:.2f}{' %' if var_name in ['Discount %', 'Purchase to Payout Rate'] else ''}")
    else:
        st.write(f"**{var_name} Margin stays above 50% across the range.**")
    if discounted_threshold is not None:
        st.write(f"**Discounted Margin falls to or below 50% at:** {discounted_threshold:.2f}{' %' if var_name in ['Discount %', 'Purchase to Payout Rate'] else ''}")
    else:
        st.write(f"**Discounted Margin stays above 50% across the range.**")

# Aggregated simulation for Purchase to Payout Rate and Avg Payout
st.header("Aggregated Simulation: Purchase to Payout Rate and Avg Payout")
# Use the same variation ranges as individual simulations
price_margins, discounted_margins = compute_aggregated_margins(
    purchase_to_payout_rate_vars, avg_payout_vars, eval_price, discount_pct
)
# Create an index for the x-axis (0 to 100% to represent the progression)
x_values = np.linspace(0, 100, 20)  # 0% to 100% progression
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_values, y=price_margins, mode='lines+markers', name="Combined (Ptr & Avg Payout)",
                         hovertemplate='Progression: %{x:.2f}%<br>Margin: %{y:.4f} (%{y:.2%})'))
fig.add_trace(go.Scatter(x=x_values, y=discounted_margins, mode='lines+markers', name="Discounted Margin",
                         hovertemplate='Progression: %{x:.2f}%<br>Margin: %{y:.4f} (%{y:.2%})'))
fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="50% Threshold")
fig.update_layout(
    title="Effect of Combined Purchase to Payout Rate and Avg Payout on Margins",
    xaxis_title="Progression (%)",
    yaxis_title="Margin",
    yaxis_range=[0, 1.2],
    hovermode="x unified"
)
st.plotly_chart(fig)

# Find and display threshold points for aggregated simulation
price_threshold, discounted_threshold = find_margin_threshold("Combined (Ptr & Avg Payout)", x_values, price_margins, discounted_margins)
if price_threshold is not None:
    st.write(f"**Combined (Ptr & Avg Payout) Margin falls to or below 50% at progression:** {price_threshold:.2f}%")
else:
    st.write(f"**Combined (Ptr & Avg Payout) Margin stays above 50% across the range.**")
if discounted_threshold is not None:
    st.write(f"**Discounted Margin falls to or below 50% at progression:** {discounted_threshold:.2f}%")
else:
    st.write(f"**Discounted Margin stays above 50% across the range.**")

# Combined simulation with user-selected variables
st.header("Combined Simulation")
st.write("Select percentage changes for each variable (0% for no change):")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    eval_price_change = -st.slider("Eval Price Decrease (%)", 0, 50, 0, step=5) / 100  # Negative for decrease
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
st.write(f"Eval Price: ${combined_eval_price:.2f}")
st.write(f"Discount %: {combined_discount_pct*100:.2f}%")
st.write(f"Purchase to Payout Rate: {combined_purchase_to_payout_rate*100:.4f}%")
st.write(f"Avg Payout: ${combined_avg_payout:.2f}")
st.write(f"**Price Margin:** {combined_pm:.4f} ({combined_pm*100:.2f}%)")
st.write(f"**Discounted Price Margin:** {combined_dpm:.4f} ({combined_dpm*100:.2f}%)")
if combined_pm >= 0.5 and combined_dpm >= 0.5:
    st.success("Both margins are above 50%!")
elif combined_pm < 0.5 or combined_dpm < 0.5:
    st.warning("One or both margins are below 50%.")

# Extreme cases for each variable
st.header("Extreme Case Scenarios (Individual Variables)")
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
