import pandas as pd
df = pd.read_parquet('/data/zhenqian/data/boolq/data/validation-00000-of-00001.parquet')
print('boolq', df.shape); print(df.columns.tolist()); print(df.iloc[0].to_dict())
print('---')
df2 = pd.read_parquet('/data/zhenqian/data/openbookqa/main/test-00000-of-00001.parquet')
print('obqa', df2.shape); print(df2.columns.tolist()); print(df2.iloc[0].to_dict())
print('---')
df3 = pd.read_parquet('/data/zhenqian/data/bbh/causal_judgement/test-00000-of-00001.parquet')
print('bbh', df3.shape); print(df3.columns.tolist()); print(df3.iloc[0].to_dict())
