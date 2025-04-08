import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Fixed sample size
SAMPLE_SIZE = 1000

# Function to calculate margins
def calculate_margins(eval_price, discount_pct, eval_pass_rate, sim_funded_rate, avg_payout):
    revenue_eval = eval_price * SAMPLE_SIZE
    discounted_eval_price = eval_price * (1 - discount_pct)
    revenue_discounted_eval = discounted_eval_price * SAMPLE_SIZE
    purchase_to_payout_rate = eval_pass_rate * sim_funded_rate
    cost = purchase_to_payout_rate * avg_payout * SAMPLE_SIZE
    net_revenue = revenue_eval - cost
    net_discounted_revenue = revenue_discounted_eval - cost
    price_margin = net_revenue / revenue_eval if revenue_eval > 0 else 0
    discounted_price_margin = net_discounted_revenue / revenue_discounted_eval if revenue_discounted_eval > 0 else 0
    return price_margin, discounted_price_margin

# Function to compute margins for a variable while keeping others constant
def compute_margins_for_variable(var_name, var_values, eval_price, discount_pct, eval_pass_rate, sim_funded_rate, avg_payout):
    price_margins = []
    discounted_margins = []
    for value in var_values:
        if var_name == "Eval Price":
            pm, dpm = calculate_margins(value, discount_pct, eval_pass_rate, sim_funded_rate, avg_payout)
        elif var_name == "Discount %":
            pm, dpm = calculate_margins(eval_price, value, eval_pass_rate, sim_funded_rate, avg_payout)
        elif var_name == "Eval Pass Rate":
            pm, dpm = calculate_margins(eval_price, discount_pct, value, sim_funded_rate, avg_payout)
        elif var_name == "Sim Funded Rate":
            pm, dpm = calculate_margins(eval_price, discount_pct, eval_pass_rate, value, avg_payout)
        elif var_name == "Avg Payout":
            pm, dpm = calculate_margins(eval_price, discount_pct, eval_pass_rate, sim_funded_rate, value)
        price_margins.append(pm)
        discounted_margins.append(dpm)
    return price_margins, discounted_margins

# Streamlit app
st.title("Margin Simulator")

st.write("Enter the following values from your sheet:")

# Input fields
eval_price = st.number_input("Eval Price", min_value=0.0, value=150.0, step=1.0)
discount_pct = st.number_input("Discount % (e.g., 30 for 30%)", min_value=0.0, max_value=100.0, value=30.0, step=1.0) / 100
eval_pass_rate = st.number_input("Eval Pass Rate (e.g., 27.01 for 27.01%)", min_value=0.0, max_value=100.0, value=27.01, step=0.01) / 100
sim_funded_rate = st.number_input("Sim Funded to Payout Rate (e.g., 4.8 for 4.8%)", min_value=0.0, max_value=100.0, value=4.8, step=0.01) / 100
avg_payout = st.number_input("Avg. Payout Amount", min_value=0.0, value=750.0, step=1.0)

# Calculate Discounted Eval Price
discounted_eval_price = eval_price * (1 - discount_pct)
st.write(f"Calculated Discounted Eval Price: {discounted_eval_price:.2f}")

# Variation ranges based on user input
eval_price_vars = [eval_price * (1 + x) for x in [-0.4, -0.2, 0, 0.2, 0.4]]  # ±20%, ±40%
discount_pct_vars = [max(0, min(1, discount_pct + x)) for x in [-0.2, -0.1, 0, 0.1, 0.2]]  # ±10%, ±20%, clamped 0-1
eval_pass_rate_vars = [max(0, min(1, eval_pass_rate * (1 + x))) for x in [-0.5, -0.25, 0, 0.25, 0.5]]  # ±25%, ±50%, clamped 0-1
sim_funded_rate_vars = [max(0, min(1, sim_funded_rate * (1 + x))) for x in [-0.5, -0.25, 0, 0.25, 0.5]]  # ±25%, ±50%, clamped 0-1
avg_payout_vars = [avg_payout * (1 + x) for x in [-0.4, -0.2, 0, 0.2, 0.4]]  # ±20%, ±40%

# Variables to analyze
variables = [
    ("Eval Price", eval_price_vars),
    ("Discount %", discount_pct_vars),
    ("Eval Pass Rate", eval_pass_rate_vars),
    ("Sim Funded Rate", sim_funded_rate_vars),
    ("Avg Payout", avg_payout_vars)
]

# Display results and plots
for var_name, var_values in variables:
    price_margins, discounted_margins = compute_margins_for_variable(
        var_name, var_values, eval_price, discount_pct, eval_pass_rate, sim_funded_rate, avg_payout
    )
    st.subheader(f"{var_name} Variations")
    for i, (val, pm, dpm) in enumerate(zip(var_values, price_margins, discounted_margins)):
        if var_name in ["Discount %", "Eval Pass Rate", "Sim Funded Rate"]:
            st.write(f"{var_name}: {val*100:.2f}%, Price Margin: {pm:.4f} ({pm*100:.2f}%), Discounted Margin: {dpm:.4f} ({dpm*100:.2f}%)")
        else:
            st.write(f"{var_name}: {val:.4f}, Price Margin: {pm:.4f} ({pm*100:.2f}%), Discounted Margin: {dpm:.4f} ({dpm*100:.2f}%)")

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    if var_name in ["Discount %", "Eval Pass Rate", "Sim Funded Rate"]:
        ax.plot([v * 100 for v in var_values], price_margins, label="Price Margin", marker='o')
        ax.plot([v * 100 for v in var_values], discounted_margins, label="Discounted Price Margin", marker='o')
        ax.set_xlabel(f"{var_name} (%)")
    else:
        ax.plot(var_values, price_margins, label="Price Margin", marker='o')
        ax.plot(var_values, discounted_margins, label="Discounted Price Margin", marker='o')
        ax.set_xlabel(var_name)
    ax.axhline(y=0.5, color='r', linestyle='--', label="50% Threshold")
    ax.set_ylabel("Margin")
    ax.set_title(f"Effect of {var_name} on Price and Discounted Price Margins")
    ax.legend()
    ax.grid(True)
    ax.set_ylim(0, 1.2)
    st.pyplot(fig)

# Extreme case
st.subheader("Extreme Case (High Cost Scenario)")
extreme_pm, extreme_dpm = calculate_margins(
    eval_price,
    min(1.0, discount_pct + 0.2),
    min(1.0, eval_pass_rate * 1.5),
    min(1.0, sim_funded_rate * 1.5),
    avg_payout * 1.4
)
st.write(f"Price Margin: {extreme_pm:.4f} ({extreme_pm*100:.2f}%), Discounted Price Margin: {extreme_dpm:.4f} ({extreme_dpm*100:.2f}%)")
