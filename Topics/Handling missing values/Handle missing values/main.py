#  write your code here 
import pandas as pd


df = pd.read_csv("./data/dataset/input.txt")
# # print(df.head())
# print(df.isna().sum())
# df = df.dropna(axis=1, thresh=7)
# # print(df.shape)
# df.price.fillna(df.price.median(), inplace=True)
# print(df.isna().sum())
print(df.head())
