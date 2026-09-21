import pandas as pd
for split in ['valid', 'test', 'train']:
    df = pd.read_parquet(f'/data/zhenqian/data/Swiss-Prot/{split}.parquet')
    print(f'{split}: shape={df.shape}, cols={df.columns.tolist()}')
df = pd.read_parquet('/data/zhenqian/data/Swiss-Prot/valid.parquet')
print(df.head(1).T)
