import pandas as pd
import sys
import os


'''
Averaging the score of multiple alpaca-eval runs
'''

assert len(sys.argv) > 1, "Must provide a directory path"

df = pd.read_csv(os.path.join(sys.argv[1], "leaderboard.csv"))
average_value = df['length_controlled_winrate'].mean()

print(f"Averaged win rate: {average_value}")
with open(os.path.join(sys.argv[1], "lc_win_rate.txt"), "w") as file:
    file.write(str(round(average_value, 3)) + '\n')


