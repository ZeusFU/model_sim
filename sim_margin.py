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

# Streamlit app
st.title("Interactive Margin Simulator")

st.sidebar.header("Input Parameters")
eval_price = st.sidebar.number_input("Eval Price", min_value=0.0, value=150.0, step=1.0)
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
st.write(f"**Calculated Discounted Eval Price:** {discounted_eval_price:.2f}")
st.write(f"**Purchase to Payout Rate:** {purchase_to_payout_rate*100:.4f}% (Eval Pass Rate * Sim Funded Rate)")
st.write(f"**Price Margin:** {base_pm:.4f} ({base_pm*100:.2f}%)")
st.write(f"**Discounted Price Margin:** {base_dpm:.4f} ({base_dpm*100:.2f}%)")

# Variation ranges with 20 steps
eval_price_vars = [eval_price * (1 - x) for x in np.linspace(0, 0.5, 20)]  # 150 to 75
discount_pct_vars = [max(0, min(1, discount_pct + (0.5 * x))) for x in np.linspace(0, 1, 20)]  # 30% to 80%
purchase_to_payout_rate_vars = [max(0, min(1, purchase_to_payout_rate * (1 + 2 * x))) for x in np.linspace(0, 1, 20)]  # Base to 200% increase
avg_payout_vars = [avg_payout * (1 + 2 * x) for x in np.linspace(0, 1, 20)]  # Base to 200% increase

# Uncomment below if needed later
# eval_pass_rate_vars = [max(0, min(1, eval_pass_rate * (1 + x))) for x in np.linspace(0, 0.5, 20)]  # 0% to +50% in 20 steps
# sim_funded_rate_vars = [max(0, min(1, sim_funded_rate * (1 + x))) for x in np.linspace(0, 0.5, 20)]  # 0% to +50% in 20 steps

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
    fig.add_trace(go.Scatter(x=x_values, y=price_margins, mode='lines+markers', name='Price Margin',
                             hovertemplate='%{x:.2f}<br>Margin: %{y:.4f} (%{y:.2%})'))
    fig.add_trace(go.Scatter(x=x_values, y=discounted_margins, mode='lines+markers', name='Discounted Price Margin',
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
st.write(f"Eval Price: {combined_eval_price:.2f}")
st.write(f"Discount %: {combined_discount_pct*100:.2f}%")
st.write(f"Purchase to Payout Rate: {combined_purchase_to_payout_rate*100:.4f}%")
st.write(f"Avg Payout: {combined_avg_payout:.2f}")
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
    ("Discount % (+50%)", eval_price, min(1.0, discount_pct + 0.5), purchase_to_payout_rate, avg_payout),  # +50% absolute increase
    ("Purchase to Payout Rate (+200%)", eval_price, discount_pct, min(1.0, purchase_to_payout_rate * 3), avg_payout),
    ("Avg Payout (+200%)", eval_price, discount_pct, purchase_to_payout_rate, avg_payout * 3)
]

data = []
for name, ep, dp, ptr, ap in extreme_scenarios:
    pm, dpm = calculate_margins(ep, dp, ptr, ap)
    data.append([name, f"{ep:.2f}", f"{dp*100:.2f}%", f"{ptr*100:.4f}%", f"{ap:.2f}", f"{pm:.4f} ({pm*100:.2f}%)", f"{dpm:.4f} ({dpm*100:.2f}%)"])

st.table({
    "Scenario": [row[0] for row in data],
    "Eval Price": [row[1] for row in data],
    "Discount %": [row[2] for row in data],
    "Purchase to Payout Rate": [row[3] for row in data],
    "Avg Payout": [row[4] for row in data],
    "Price Margin": [row[5] for row in data],
    "Discounted Price Margin": [row[6] for row in data]
})
