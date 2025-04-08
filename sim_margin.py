import numpy as np
import matplotlib.pyplot as plt

# Fixed sample size
SAMPLE_SIZE = 1000

# Function to get user input with validation (for regular floats)
def get_float_input(prompt, min_val=None, max_val=None):
    while True:-
        try:
            value = float(input(prompt))
            if (min_val is not None and value < min_val) or (max_val is not None and value > max_val):
                print(f"Please enter a value between {min_val} and {max_val}.")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

# Function to get percentage input with validation (returns decimal)
def get_percentage_input(prompt):
    while True:
        try:
            value = float(input(prompt))
            if value < 0 or value > 100:
                print("Please enter a percentage between 0 and 100.")
                continue
            return value / 100  # Convert to decimal
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

# Prompt user for base values
print("Please enter the following values from your sheet:")
BASE_EVAL_PRICE = get_float_input("Eval Price: ", min_val=0)
BASE_DISCOUNT_PCT = get_percentage_input("Discount % (e.g., 30 for 30%): ")
BASE_DISCOUNTED_EVAL_PRICE = get_float_input("Discounted Eval Price: ", min_val=0)
BASE_EVAL_PASS_RATE = get_percentage_input("Eval Pass Rate (e.g., 27.01 for 27.01%): ")
BASE_SIM_FUNDED_RATE = get_percentage_input("Sim Funded to Payout Rate (e.g., 4.8 for 4.8%): ")
BASE_AVG_PAYOUT = get_float_input("Avg. Payout Amount: ", min_val=0)

# Validate Discounted Eval Price consistency
expected_discounted_price = BASE_EVAL_PRICE * (1 - BASE_DISCOUNT_PCT)
if abs(BASE_DISCOUNTED_EVAL_PRICE - expected_discounted_price) > 0.01:  # Allow small rounding difference
    print(f"Warning: Discounted Eval Price ({BASE_DISCOUNTED_EVAL_PRICE}) does not match "
          f"Eval Price ({BASE_EVAL_PRICE}) with Discount % ({BASE_DISCOUNT_PCT*100:.2f}%). Expected: {expected_discounted_price:.2f}")

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

# Variation ranges based on user input
eval_price_vars = [BASE_EVAL_PRICE * (1 + x) for x in [-0.4, -0.2, 0, 0.2, 0.4]]  # ±20%, ±40%
discount_pct_vars = [max(0, min(1, BASE_DISCOUNT_PCT + x)) for x in [-0.2, -0.1, 0, 0.1, 0.2]]  # ±10%, ±20%, clamped 0-1
eval_pass_rate_vars = [max(0, min(1, BASE_EVAL_PASS_RATE * (1 + x))) for x in [-0.5, -0.25, 0, 0.25, 0.5]]  # ±25%, ±50%, clamped 0-1
sim_funded_rate_vars = [max(0, min(1, BASE_SIM_FUNDED_RATE * (1 + x))) for x in [-0.5, -0.25, 0, 0.25, 0.5]]  # ±25%, ±50%, clamped 0-1
avg_payout_vars = [BASE_AVG_PAYOUT * (1 + x) for x in [-0.4, -0.2, 0, 0.2, 0.4]]  # ±20%, ±40%

# Function to compute margins for a variable while keeping others constant
def compute_margins_for_variable(var_name, var_values):
    price_margins = []
    discounted_margins = []
    for value in var_values:
        if var_name == "Eval Price":
            pm, dpm = calculate_margins(value, BASE_DISCOUNT_PCT, BASE_EVAL_PASS_RATE, BASE_SIM_FUNDED_RATE, BASE_AVG_PAYOUT)
        elif var_name == "Discount %":
            pm, dpm = calculate_margins(BASE_EVAL_PRICE, value, BASE_EVAL_PASS_RATE, BASE_SIM_FUNDED_RATE, BASE_AVG_PAYOUT)
        elif var_name == "Eval Pass Rate":
            pm, dpm = calculate_margins(BASE_EVAL_PRICE, BASE_DISCOUNT_PCT, value, BASE_SIM_FUNDED_RATE, BASE_AVG_PAYOUT)
        elif var_name == "Sim Funded Rate":
            pm, dpm = calculate_margins(BASE_EVAL_PRICE, BASE_DISCOUNT_PCT, BASE_EVAL_PASS_RATE, value, BASE_AVG_PAYOUT)
        elif var_name == "Avg Payout":
            pm, dpm = calculate_margins(BASE_EVAL_PRICE, BASE_DISCOUNT_PCT, BASE_EVAL_PASS_RATE, BASE_SIM_FUNDED_RATE, value)
        price_margins.append(pm)
        discounted_margins.append(dpm)
    return price_margins, discounted_margins

# Plotting function (display percentages for relevant variables)
def plot_margins(var_name, var_values, price_margins, discounted_margins):
    plt.figure(figsize=(10, 6))
    if var_name in ["Discount %", "Eval Pass Rate", "Sim Funded Rate"]:
        plt.plot([v * 100 for v in var_values], price_margins, label="Price Margin", marker='o')
        plt.plot([v * 100 for v in var_values], discounted_margins, label="Discounted Price Margin", marker='o')
        plt.xlabel(f"{var_name} (%)")
    else:
        plt.plot(var_values, price_margins, label="Price Margin", marker='o')
        plt.plot(var_values, discounted_margins, label="Discounted Price Margin", marker='o')
        plt.xlabel(var_name)
    plt.axhline(y=0.5, color='r', linestyle='--', label="50% Threshold")
    plt.ylabel("Margin")
    plt.title(f"Effect of {var_name} on Price and Discounted Price Margins")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.2)  # Set y-axis from 0 to 120% for clarity
    plt.show()

# Compute and plot for each variable
variables = [
    ("Eval Price", eval_price_vars),
    ("Discount %", discount_pct_vars),
    ("Eval Pass Rate", eval_pass_rate_vars),
    ("Sim Funded Rate", sim_funded_rate_vars),
    ("Avg Payout", avg_payout_vars)
]

for var_name, var_values in variables:
    price_margins, discounted_margins = compute_margins_for_variable(var_name, var_values)
    print(f"\n{var_name} Variations:")
    for i, (val, pm, dpm) in enumerate(zip(var_values, price_margins, discounted_margins)):
        if var_name in ["Discount %", "Eval Pass Rate", "Sim Funded Rate"]:
            print(f"{var_name}: {val*100:.2f}%, Price Margin: {pm:.4f} ({pm*100:.2f}%), Discounted Margin: {dpm:.4f} ({dpm*100:.2f}%)")
        else:
            print(f"{var_name}: {val:.4f}, Price Margin: {pm:.4f} ({pm*100:.2f}%), Discounted Margin: {dpm:.4f} ({dpm*100:.2f}%)")
    plot_margins(var_name, var_values, price_margins, discounted_margins)

# Extreme case test with user inputs
print("\nExtreme Case (High Cost Scenario):")
extreme_pm, extreme_dpm = calculate_margins(
    BASE_EVAL_PRICE,
    min(1.0, BASE_DISCOUNT_PCT + 0.2),  # Max discount at 100%
    min(1.0, BASE_EVAL_PASS_RATE * 1.5),
    min(1.0, BASE_SIM_FUNDED_RATE * 1.5),
    BASE_AVG_PAYOUT * 1.4
)
print(f"Price Margin: {extreme_pm:.4f} ({extreme_pm*100:.2f}%), Discounted Price Margin: {extreme_dpm:.4f} ({extreme_dpm*100:.2f}%)")
