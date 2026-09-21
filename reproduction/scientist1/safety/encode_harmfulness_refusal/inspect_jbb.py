import pandas as pd
df = pd.read_csv("/data/zhenqian/data/JailbreakBench/data/judge-comparison.csv")
print(df.shape)
print(df.columns.tolist())
print(df.head(3).to_dict(orient='records'))
print()
print("uniq attack types:", df["attack_type"].value_counts().head(20) if "attack_type" in df.columns else "n/a")
