import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import csv

def upload_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        generate_chart(file_path)

def generate_chart(file_path):
    # Read data from CSV
    trades_data = []
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            trades_data.append(float(row[0]))

    # Number of simulations
    num_simulations = 50000

    # Number of trades per simulation
    num_trades = len(trades_data)

    # Simulate trading sequences
    simulations = np.zeros((num_simulations, num_trades))
    for i in range(num_simulations):
        simulations[i] = np.random.choice(trades_data, num_trades)

    # Calculate cumulative P/L for each simulation
    cumulative_pl = np.cumsum(simulations, axis=1)

    # Calculate average P/L and standard deviation
    avg_pl = np.mean(cumulative_pl, axis=0)
    std_dev = np.std(cumulative_pl, axis=0)

    # Calculate worst and best P/L at each trade point
    worst_pl = np.min(cumulative_pl, axis=0)
    best_pl = np.max(cumulative_pl, axis=0)

    # Calculate Max Drawdowns
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

    # Calculate statistics
    avg_max_drawdown = np.mean([drawdown[0] for drawdown in drawdowns])
    worst_max_drawdown = np.max([drawdown[0] for drawdown in drawdowns])
    avg_max_drawdown_duration = np.mean([drawdown[1] for drawdown in drawdowns])
    worst_max_drawdown_duration = np.max([drawdown[1] for drawdown in drawdowns])

    # Round Max Drawdown Duration statistics to the nearest trade
    avg_max_drawdown_duration = round(avg_max_drawdown_duration)
    worst_max_drawdown_duration = round(worst_max_drawdown_duration)

    # Plotting
    plt.figure(figsize=(12, 8))

    # Cumulative sum of trades_data
    plt.plot(np.cumsum(trades_data), linestyle='-', color='blue', label='Actual Trading Results')

    # Average P/L curve
    plt.plot(avg_pl, linestyle='--', color='black', label="Expected Value (EV)")

    # Two standard deviation bands
    plt.fill_between(range(num_trades), avg_pl - 2 * std_dev, avg_pl + 2 * std_dev,
                    color='gray', alpha=0.1, label='±2 Std Dev')

    # Worst P/L curve
    plt.plot(worst_pl, linestyle='--', color='red', label='Worst P/L')

    # Best P/L curve
    plt.plot(best_pl, linestyle='--', color='green', label='Best P/L')

    # Annotation: Statistics
    plt.text(0.02, 0.95, f'Average Max Drawdown: {avg_max_drawdown:.2f}R', color='black', ha='left', va='top', transform=plt.gca().transAxes)
    plt.text(0.02, 0.90, f'Worst Max Drawdown: {worst_max_drawdown:.2f}R', color='black', ha='left', va='top', transform=plt.gca().transAxes)
    plt.text(0.02, 0.85, f'Average Max Drawdown Duration: {avg_max_drawdown_duration} Trades', color='black', ha='left', va='top', transform=plt.gca().transAxes)
    plt.text(0.02, 0.80, f'Worst Max Drawdown Duration: {worst_max_drawdown_duration} Trades', color='black', ha='left', va='top', transform=plt.gca().transAxes)

    # Legend
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5)

    plt.xlabel('Trades')
    plt.ylabel('P/L')
    plt.title('Trading Simulation Results')
    plt.grid(True)
    plt.show()

# Create the main application window
root = tk.Tk()
root.title("Trading Simulation Visualizer")

# Create and place the Upload button
upload_button = tk.Button(root, text="Upload CSV", command=upload_file)
upload_button.pack(pady=20)

# Run the application
root.mainloop()