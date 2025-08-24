import pandas as pd

load_csv = "data/house.csv"
df = pd.read_csv(load_csv)

df.drop(labels="No.", axis=1, inplace=True)
df.drop(labels="Info", axis=1, inplace=True)

print(df.head())

df.to_csv("data/house_cleaned.csv", index=False)