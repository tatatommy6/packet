#이런 ㅈ밥 코드는 주석 안달아요.
import pandas as pd

load_csv = "data/house.csv"
df = pd.read_csv(load_csv)

df.drop(labels="No.", axis=1, inplace=True)
df.drop(labels="Info", axis=1, inplace=True)

print(df.head())

df.to_csv("data/house_cleaned.csv", index=False)