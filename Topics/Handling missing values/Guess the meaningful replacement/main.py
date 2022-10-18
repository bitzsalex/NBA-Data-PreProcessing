#  write your code here 
import pandas as pd


df = pd.read_csv("./data/dataset/input.txt")
df["totsp"] = df["livingsp"] + df["nonlivingsp"]
print(df.totsp.sum())
