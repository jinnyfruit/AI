import torch
import math
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Read data
df = pd.read_csv("data/HousingData.csv")
print("---------------------- Describe data ----------------------")
print(df.describe())

# Data Preprocessing
print("---------------------- Check NAN ----------------------")
print(df.isna())
print("---------------------- Drop NAN ----------------------")
print(df.dropna(inplace=True))

# Define data class
batch_size = 3
class myDataset(Dataset):
   def __init__(self, df_data) -> None:
       super.__init__()
       # By using "self" we can use these in other methods
       self.df_data = df_data

       # Split target data
       self.data_x = df_data.drop(["MEDV"], axis=1)
       self.data_y = df_data.loc[:,["MEDV"]]

    def




