import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

raw_data = "data/raw_data/raw_data.xlsx"
data = pd.read_excel(raw_data)
print(data)

# check for null value
data.info()