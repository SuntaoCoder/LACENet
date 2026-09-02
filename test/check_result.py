# %%
import pandas as pd

df = pd.read_csv("./results/c2db_Z_reference_gated.csv")
for col in df.columns[1:]:
    print(f"{col}: ", df[col].mean().round(6))

# %%
