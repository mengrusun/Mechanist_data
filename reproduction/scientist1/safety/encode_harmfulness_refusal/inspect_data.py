import pandas as pd, os, json

print("=== AdvBench ===")
harm = pd.read_csv('/data/zhenqian/data/AdvBench/harmful_behaviors.csv')
print(harm.shape, harm.columns.tolist())
print(harm.iloc[0].to_dict())

print("\n=== Alpaca ===")
alp = pd.read_parquet('/data/zhenqian/data/Alpaca/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet')
print(alp.shape, alp.columns.tolist())
print(alp.iloc[0].to_dict())

print("\n=== XSTest ===")
for f in os.listdir('/data/zhenqian/data/XSTest/data'):
    df = pd.read_parquet(f'/data/zhenqian/data/XSTest/data/{f}')
    print(f, df.shape, df.columns.tolist())
    print(df.iloc[0].to_dict())

print("\n=== JailbreakBench ===")
for f in ['benign-behaviors.csv', 'harmful-behaviors.csv']:
    df = pd.read_csv(f'/data/zhenqian/data/JailbreakBench/data/{f}')
    print(f, df.shape, df.columns.tolist())
    print(df.iloc[0].to_dict())

print("\n=== CATQA (jsonlines) ===")
lines = []
with open('/data/zhenqian/data/CATQA/data/catqa_english.json') as f:
    for i, line in enumerate(f):
        lines.append(line.strip())
        if i >= 2: break
for l in lines:
    print(l[:300])

print("\n=== Sorry-Bench ===")
with open('/data/zhenqian/data/Sorry-Bench/question.jsonl') as f:
    for i, line in enumerate(f):
        print(json.loads(line))
        if i >= 1: break
