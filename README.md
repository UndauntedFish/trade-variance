# trade-variance
Takes in a trader's trade history, and generates a chart that visualizes the potential distribution of that trader's returns.

This helps the trader comprehend the probabilistic variance created by executing their trading strategy. It also helps differentiate between problematic losing streaks caused by trader error, and natural losing streaks caused by variance.

## Usage
1. Download trade-sim.exe: https://www.dropbox.com/scl/fi/xfddyxp0xdtanz5i4uzk6/tradesim.exe?rlkey=cworj7o3x97ruw5sbi76e1bbr&dl=0

2. Upload a CSV file containing the results of each trade in your system, expressed as an R multiple. The CSV file should follow the same format as trades_data.csv.

3. More on how to calculate/use R multiples here: https://stonkjournal.com/a-guide-to-r-multiple-and-risk-management/

## Example of Usage 
Say that I'm a trader and I want to see the kind of P/L swings my strategy can put me through. I compile my trade history into a CSV file, upload the CSV file into trade-sim.exe, and then let the simulation run.

![PAMR_Results](https://github.com/UndauntedFish/trade-variance/assets/58181651/cbb20b43-3537-402c-91ec-13ad8c6e6369)
Now I can see the potential spread of outcomes my strategy can take me through and get an idea of what my worst-case losing periods look like, how long they will last, and how to adjust my risk exposure to survive them.
