#  write your code here 
import pandas as pd


df = pd.read_csv("./data/dataset/input.txt")

df["height"] = df.groupby("location")["height"].apply(lambda col: col.fillna(round(col.mean(), 1)))

print(df.height.sum())
