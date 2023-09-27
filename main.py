#This is a starting point
#uncomment below to install merlin library
#!pip install merlin, nvtabular, merlin.models
#I had to downgrade keras to 12.2.0 maybe issues regarding tensorflow
#run lines below if you have also have an error ab not finding a keras package
#!pip uninstall keras
#!pip install keras==2.12.0


#These are the same imports from
#https://github.com/NVIDIA-Merlin/models/blob/main/examples/02-Merlin-Models-and-NVTabular-integration.ipynb

import os
import pandas as pd
import nvtabular as nvt
from merlin.models.utils.example_utils import workflow_fit_transform
import merlin.io

import merlin.models.tf as mm

from nvtabular import ops
from merlin.core.utils import download_file
from merlin.schema.tags import Tags

#This is how to iniate a dataset using NVtabular an NVIDIA 
#optimized ETL/feature engineering(i think?) library
#data = nvt.io.dataset.Dataset('data/reviews.csv')

data = pd.read_csv("data/reviews.csv")
data_size = data.shape[0]
train_split_ratio = int(.2 * data_size)

train_data = data[train_split_ratio:]
valid_ratio = data[:train_split_ratio]

#Just checking to make sure it all runs
print(data.head())