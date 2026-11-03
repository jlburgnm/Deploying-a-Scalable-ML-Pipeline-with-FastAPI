import pandas as pd

path = "data/census.csv"
df = pd.read_csv(path)

df.columns = df.columns.str.strip()
for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].str.strip()

df.to_csv(path, index=False)
