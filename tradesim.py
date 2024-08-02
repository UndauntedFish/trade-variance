import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.stats import norm
from scipy.optimize import curve_fit

def upload_file():
    """
    Opens a file dialog to select a CSV file and generates the trading chart based on the selected file.
    """
    file_path = filedialog.askopenfilename()
    if file_path:
        # Use custom_trade_var instead of custom_trade_amount_var
        custom_trade_amount = custom_trade_var.get()
        generate_chart(file_path, custom_trade_amount)

def generate_chart(file_path, custom_trade_amount):
    """
    Generates a trading simulation chart based on data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file containing trade data.
        custom_trade_amount (float): Custom trade amount for the simulation.
    """
    confidence_interval = confidence_var.get()
    if confidence_interval:
        confidence_interval = float(confidence_interval)
        if confidence_interval <= 0 or confidence_interval >= 100:
            print("Confidence interval must be between 1% and 99%.")
            return
    else:
        confidence_interval = 95  # Default confidence interval

    # Calculate standard deviation multiplier
    if confidence_interval == 95:
        std_dev_multiplier = 2
    else:
        z_score = norm.ppf((1 + (confidence_interval / 100)) / 2)
        std_dev_multiplier = z_score

    # Read trade data from the CSV file
    trades_data = read_csv(file_path)

    num_trades = len(trades_data)
    num_simulations = 2000000

    # Simulate trading sequences
    simulations, cumulative_pl, avg_pl, std_dev, worst_pl, best_pl = simulate_trades(trades_data, num_trades, num_simulations)

    # Calculate extended range for simulations and breakeven trades
    extended_num_trades = num_trades * 2
    extended_simulations, extended_breakeven_simulations = extend_simulations(simulations, trades_data, num_simulations, extended_num_trades)
    
    extended_avg_pl, extended_std_dev, extended_worst_pl, extended_best_pl, extended_cumulative_pl = analyze_extended_simulations(extended_simulations, extended_num_trades)
    breakeven_avg_pl, breakeven_std_dev, breakeven_worst_pl, breakeven_best_pl = analyze_breakeven_trades(extended_breakeven_simulations, extended_num_trades, trades_data)

    # Calculate Max Drawdowns
    drawdown_stats = calculate_max_drawdowns(cumulative_pl)
    
    # Find the crossover point
    crossover_index = find_crossover_index(extended_avg_pl, extended_std_dev, breakeven_avg_pl, breakeven_std_dev, num_trades, extended_num_trades)

    # Plot the results
    plot_results(trades_data, extended_avg_pl, extended_std_dev, extended_worst_pl, extended_best_pl,
                 breakeven_avg_pl, breakeven_std_dev, breakeven_worst_pl, breakeven_best_pl,
                 drawdown_stats, crossover_index, std_dev_multiplier)

def read_csv(file_path):
    """
    Reads trade data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        np.ndarray: Array of trade data.
    """
    trades_data = []
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            trades_data.append(float(row[0]))
    return np.array(trades_data)

def simulate_trades(trades_data, num_trades, num_simulations):
    """
    Simulates trading sequences based on historical trade data.

    Args:
        trades_data (np.ndarray): Array of trade data.
        num_trades (int): Number of trades in the simulation.
        num_simulations (int): Number of simulations to run.

    Returns:
        tuple: Contains simulations, cumulative P/L, average P/L, standard deviation, worst P/L, and best P/L.
    """
    simulations = np.random.choice(trades_data, (num_simulations, num_trades))
    cumulative_pl = np.cumsum(simulations, axis=1)
    avg_pl = np.mean(cumulative_pl, axis=0)
    std_dev = np.std(cumulative_pl, axis=0)
    worst_pl = np.min(cumulative_pl, axis=0)
    best_pl = np.max(cumulative_pl, axis=0)
    return simulations, cumulative_pl, avg_pl, std_dev, worst_pl, best_pl

def extend_simulations(simulations, trades_data, num_simulations, extended_num_trades):
    """
    Extends simulations to include additional trades.

    Args:
        simulations (np.ndarray): Array of simulated trading sequences.
        trades_data (np.ndarray): Array of trade data.
        num_simulations (int): Number of simulations.
        extended_num_trades (int): Number of trades in the extended simulation.

    Returns:
        tuple: Contains extended simulations and extended breakeven simulations.
    """
    extended_simulations = np.zeros((num_simulations, extended_num_trades))
    extended_breakeven_simulations = np.zeros((num_simulations, extended_num_trades))

    for i in range(num_simulations):
        extended_simulations[i] = np.concatenate([simulations[i], np.random.choice(trades_data, extended_num_trades - len(simulations[i]))])
        extended_breakeven_simulations[i] = np.where(np.random.rand(extended_num_trades) < breakeven_winrate(trades_data), 
                                                    np.mean(trades_data[trades_data > 0]), 
                                                    np.mean(trades_data[trades_data < 0]))
    return extended_simulations, extended_breakeven_simulations

def breakeven_winrate(trades_data):
    """
    Calculates the win rate for breakeven trades.

    Args:
        trades_data (np.ndarray): Array of trade data.

    Returns:
        float: Win rate for breakeven trades.
    """
    avg_win = np.mean(trades_data[trades_data > 0])
    avg_loss = np.mean(trades_data[trades_data < 0])
    return -avg_loss / (avg_win - avg_loss)

def analyze_extended_simulations(extended_simulations, extended_num_trades):
    """
    Analyzes extended simulations to calculate average P/L, standard deviation, worst P/L, and best P/L.

    Args:
        extended_simulations (np.ndarray): Array of extended simulations.
        extended_num_trades (int): Number of trades in the extended simulations.

    Returns:
        tuple: Contains average P/L, standard deviation, worst P/L, and best P/L for extended simulations.
    """
    extended_cumulative_pl = np.cumsum(extended_simulations, axis=1)
    extended_avg_pl = np.mean(extended_cumulative_pl, axis=0)
    extended_std_dev = np.std(extended_cumulative_pl, axis=0)
    extended_worst_pl = np.min(extended_cumulative_pl, axis=0)
    extended_best_pl = np.max(extended_cumulative_pl, axis=0)
    return extended_avg_pl, extended_std_dev, extended_worst_pl, extended_best_pl, extended_cumulative_pl

def analyze_breakeven_trades(extended_breakeven_simulations, extended_num_trades, trades_data):
    """
    Analyzes breakeven trades to calculate average P/L, standard deviation, worst P/L, and best P/L.

    Args:
        extended_breakeven_simulations (np.ndarray): Array of extended breakeven simulations.
        extended_num_trades (int): Number of trades in the extended simulations.
        trades_data (np.ndarray): Array of trade data.

    Returns:
        tuple: Contains average P/L, standard deviation, worst P/L, and best P/L for breakeven trades.
    """
    extended_breakeven_cumulative_pl = np.cumsum(extended_breakeven_simulations, axis=1)
    breakeven_avg_pl = np.mean(extended_breakeven_cumulative_pl, axis=0)
    breakeven_std_dev = np.std(extended_breakeven_cumulative_pl, axis=0)
    breakeven_worst_pl = np.min(extended_breakeven_cumulative_pl, axis=0)
    breakeven_best_pl = np.max(extended_breakeven_cumulative_pl, axis=0)
    return breakeven_avg_pl, breakeven_std_dev, breakeven_worst_pl, breakeven_best_pl

def calculate_max_drawdowns(cumulative_pl):
    """
    Calculates maximum drawdowns and their durations.

    Args:
        cumulative_pl (np.ndarray): Array of cumulative P/L for simulations.

    Returns:
        tuple: Contains average and worst max drawdowns, and average and worst max drawdown durations.
    """
    drawdowns = []
    for pl in cumulative_pl:
        peak = pl[0]
        max_drawdown = 0
        max_drawdown_duration = 0
        current_drawdown_duration = 0
        for val in pl[1:]:
            if val > peak:
                peak = val
                max_drawdown_duration = max(max_drawdown_duration, current_drawdown_duration)
                current_drawdown_duration = 0
            else:
                current_drawdown_duration += 1
                drawdown = peak - val
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        drawdowns.append((max_drawdown, max_drawdown_duration))
    avg_max_drawdown = np.mean([d[0] for d in drawdowns])
    worst_max_drawdown = np.max([d[0] for d in drawdowns])
    avg_max_drawdown_duration = np.mean([d[1] for d in drawdowns])
    worst_max_drawdown_duration = np.max([d[1] for d in drawdowns])
    return avg_max_drawdown, worst_max_drawdown, avg_max_drawdown_duration, worst_max_drawdown_duration

def find_crossover_index(extended_avg_pl, extended_std_dev, breakeven_avg_pl, breakeven_std_dev, num_trades, extended_num_trades):
    """
    Finds the index where the performance of the trading strategy crosses the performance of the breakeven strategy.
    If no crossover is found within the given number of trades, fits parabolic equations to the standard deviation bands
    and finds their intersection.

    Args:
        extended_avg_pl (np.ndarray): Extended average P/L for actual trades.
        extended_std_dev (np.ndarray): Extended standard deviation for actual trades.
        breakeven_avg_pl (np.ndarray): Extended average P/L for breakeven trades.
        breakeven_std_dev (np.ndarray): Extended standard deviation for breakeven trades.
        num_trades (int): Number of trades.
        extended_num_trades (int): Number of extended trades.

    Returns:
        int: Index where the trading strategy performance crosses the breakeven strategy performance.
    """
    crossover_index = extended_num_trades
    
    # Check for crossover within the given number of trades
    for i in range(min(len(extended_avg_pl), len(breakeven_avg_pl))):
        if extended_avg_pl[i] - 2 * extended_std_dev[i] > breakeven_avg_pl[i] + 2 * breakeven_std_dev[i]:
            crossover_index = i
            break
    
    if crossover_index == extended_num_trades:
        # Define the parabolic function
        def parabolic(x, a, b, c):
            return a * x**2 + b * x + c

        # Fit parabolas to the lower green and upper grey standard deviation bands
        x = np.arange(len(extended_avg_pl))
        y_lower = extended_avg_pl - 2 * extended_std_dev
        y_upper = breakeven_avg_pl + 2 * breakeven_std_dev

        try:
            params_lower, _ = curve_fit(parabolic, x, y_lower)
            params_upper, _ = curve_fit(parabolic, x, y_upper)

            # Find intersection of the parabolas
            def intersection(x):
                return parabolic(x, *params_lower) - parabolic(x, *params_upper)
            
            # Find the x where the intersection function changes sign
            from scipy.optimize import fsolve
            intersection_x = fsolve(intersection, len(extended_avg_pl) / 2)

            # Ensure the intersection_x is within the range
            if 0 <= intersection_x < extended_num_trades:
                crossover_index = int(np.round(intersection_x))
            else:
                crossover_index = extended_num_trades

        except Exception as e:
            print(f"Error fitting parabolas: {e}")
            crossover_index = extended_num_trades

    return crossover_index

def plot_results(trades_data, extended_avg_pl, extended_std_dev, extended_worst_pl, extended_best_pl,
                 breakeven_avg_pl, breakeven_std_dev, breakeven_worst_pl, breakeven_best_pl,
                 drawdown_stats, crossover_index, std_dev_multiplier):
    """
    Plots the trading simulation results.

    Args:
        trades_data (np.ndarray): Array of trade data.
        extended_avg_pl (np.ndarray): Extended average P/L for actual trades.
        extended_std_dev (np.ndarray): Extended standard deviation for actual trades.
        extended_worst_pl (np.ndarray): Extended worst P/L for actual trades.
        extended_best_pl (np.ndarray): Extended best P/L for actual trades.
        breakeven_avg_pl (np.ndarray): Extended average P/L for breakeven trades.
        breakeven_std_dev (np.ndarray): Extended standard deviation for breakeven trades.
        breakeven_worst_pl (np.ndarray): Extended worst P/L for breakeven trades.
        breakeven_best_pl (np.ndarray): Extended best P/L for breakeven trades.
        drawdown_stats (tuple): Contains statistics for max drawdowns.
        crossover_index (int): Index of the crossover point.
        std_dev_multiplier (float): Multiplier for the standard deviation bands.
    """
    avg_max_drawdown, worst_max_drawdown, avg_max_drawdown_duration, worst_max_drawdown_duration = drawdown_stats

    num_trades = len(trades_data)
    extended_num_trades = len(extended_avg_pl)

    # Determine the limit for the x-axis
    x_limit = max(num_trades, crossover_index + 1)

    plt.figure(figsize=(12, 8))

    # Actual trading results
    plt.plot(np.arange(num_trades), np.cumsum(trades_data), linestyle='-', color='blue', label='Actual Trading Results')

    # Average P/L curve
    plt.plot(np.arange(min(x_limit, len(extended_avg_pl))), extended_avg_pl[:x_limit], linestyle='-', color='black', label="Expected Value (EV) of Actual Trading Results")

    # Standard deviation bands for actual trades
    plt.fill_between(np.arange(min(x_limit, len(extended_avg_pl))), 
                     extended_avg_pl[:x_limit] - std_dev_multiplier * extended_std_dev[:x_limit], 
                     extended_avg_pl[:x_limit] + std_dev_multiplier * extended_std_dev[:x_limit],
                     color='green', alpha=0.1, label=f'±{std_dev_multiplier} Std Dev (Actual Trading Results)')
    
    # Worst and best P/L curves for actual trades
    plt.plot(np.arange(min(x_limit, len(extended_worst_pl))), extended_worst_pl[:x_limit], linestyle='-', color='red', label='Worst P/L (Actual Trading Results)')
    plt.plot(np.arange(min(x_limit, len(extended_best_pl))), extended_best_pl[:x_limit], linestyle='-', color='green', label='Best P/L (Actual Trading Results)')

    # Standard deviation bands for breakeven trades
    plt.fill_between(np.arange(min(x_limit, len(breakeven_avg_pl))), 
                     breakeven_avg_pl[:x_limit] - std_dev_multiplier * breakeven_std_dev[:x_limit], 
                     breakeven_avg_pl[:x_limit] + std_dev_multiplier * breakeven_std_dev[:x_limit],
                     color='gray', alpha=0.1, label=f'±{std_dev_multiplier} Std Dev (Random Walk)')
    
    # Worst and best P/L curves for breakeven trades
    plt.plot(np.arange(min(x_limit, len(breakeven_worst_pl))), breakeven_worst_pl[:x_limit], linestyle=':', color='red', label='Worst P/L (Random Walk)')
    plt.plot(np.arange(min(x_limit, len(breakeven_best_pl))), breakeven_best_pl[:x_limit], linestyle=':', color='green', label='Best P/L (Random Walk)')

    # Annotation: Statistics
    plt.text(0.02, 0.95, f'Average Max Drawdown: {avg_max_drawdown:.2f}R', color='black', ha='left', va='top', transform=plt.gca().transAxes)
    plt.text(0.02, 0.90, f'Worst Max Drawdown: {worst_max_drawdown:.2f}R', color='black', ha='left', va='top', transform=plt.gca().transAxes)
    plt.text(0.02, 0.85, f'Average Max Drawdown Duration: {avg_max_drawdown_duration} Trades', color='black', ha='left', va='top', transform=plt.gca().transAxes)
    plt.text(0.02, 0.80, f'Worst Max Drawdown Duration: {worst_max_drawdown_duration} Trades', color='black', ha='left', va='top', transform=plt.gca().transAxes)
    plt.text(0.02, 0.75, f'Trades for 95% Power: {crossover_index}', color='black', ha='left', va='top', transform=plt.gca().transAxes)

    # Legend
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

    plt.xlabel('Trades')
    plt.ylabel('P/L')
    plt.title('Trading Simulation Results')
    plt.xlim(0, x_limit)  # Set x-axis limit
    plt.grid(True)
    plt.show()

# GUI setup
root = tk.Tk()
root.title("Trading Simulation")

# Create and place the custom trade amount input field
custom_trade_label = tk.Label(root, text="Use custom trade amount (leave blank for auto):")
custom_trade_label.pack(pady=5)
custom_trade_var = tk.StringVar()  # Ensure this matches the variable used in upload_file
custom_trade_entry = tk.Entry(root, textvariable=custom_trade_var)
custom_trade_entry.pack(pady=5)

# Create and place the confidence interval input field
confidence_label = tk.Label(root, text="Confidence interval for power analysis: 95%")
confidence_label.pack(pady=5)
confidence_var = tk.StringVar(value="95")
confidence_entry = tk.Entry(root, textvariable=confidence_var)
confidence_entry.pack(pady=5)

tk.Button(root, text="Upload CSV", command=upload_file).pack()

root.mainloop()