import pandas as pd

# Import "data/mexico-real-estate-clean.csv"
df = pd.read_csv("C:/Users/STEVE/OneDrive/Desktop/PYTHON/Housing_in_Mexico/data/mexico-reai-estate-clean.csv", encoding='latin1')

# Print object type, shape, and head
print("df type:", type(df))

print("df shape:", df.shape)

df.head()

df["state"].value_counts().head(10)

import numpy as np
df["log_price_usd"] = np.log(df["price_usd"])
df["log_area_m2"] = np.log(df["area_m2"])

probability_val = pred
