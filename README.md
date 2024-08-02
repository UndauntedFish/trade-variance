# trade-variance
Takes in a trader's trade history, and generates a chart that visualizes the potential distribution of that trader's returns.

This helps the trader comprehend the probabilistic variance created by executing their trading strategy. It also helps differentiate between problematic losing streaks caused by trader error, and natural losing streaks caused by variance.

## Usage
1. Download trade-sim.exe: TBA

2. Upload a CSV file containing the results of each trade in your system, expressed as an R multiple. The CSV file should follow the same format as [trades_data.csv](trades_data.csv).

3. More on how to calculate/use R multiples here: https://stonkjournal.com/a-guide-to-r-multiple-and-risk-management/

## Example of Usage 
If a trader wanted to see the kind of P/L swings executing their trades will put them through over a large sample size, they can upload their current trade log into trade-sim.exe and generate a visualization to show that.

![PAMR_Results](https://github.com/UndauntedFish/trade-variance/assets/58181651/cbb20b43-3537-402c-91ec-13ad8c6e6369)
Now the trader can see the potential spread of outcomes of their trades over millions of trades, and get an idea of what their worst-case losing periods look like, how long they will last, and how to adjust their risk exposure to survive them.
