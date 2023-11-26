import os
import pandas as pd
import numpy as np
import nvtabular as nvt
import matplotlib.pyplot as plt
import seaborn as sns
import keras
import polars as pl

from sklearn.model_selection import train_test_split

from merlin.models.utils.example_utils import workflow_fit_transform
import merlin.io
import tensorflow as tf

import merlin.models.tf as mm
from merlin.dag.ops.subgraph import Subgraph
from merlin.io.dataset import Dataset
from nvtabular.ops import *
from merlin.core.utils import download_file
from merlin.schema.tags import Tags
from datetime import datetime, timezone, date


#READING IN DATA FILTERING OUT WHATS NOT USEFUL, COUNTING NUM OF VIEWS AND MAKING COLUMN, CREATING BINARY ORDERED & PRODUCT ADDED COLUMN, SAMPLING, CREATING 50:50 SPLIT BETWEEN 'CONVERTED' AND 'NOT CONVERTED' AKA 1/0

#THIS CAN BE IMPROVED WITH USING POLARS OR A MORE DYNAMIC BIG DATA PROCESSING LIBRARY IF IT MATTERS
def read_target_sample(file_path = '/Users/andrew/Desktop/projects/recsys_data/2023-10-05 9_23pm (2).csv'): #this is just my own filepath
    df = pd.read_csv(file_path).dropna()
    view_counts = df[df['EVENT_NAME'] == 'product_viewed'].groupby(['USER_ID', 'ITEM_ID'])['EVENT_NAME'].count().reset_index()
    view_counts.rename(columns={'EVENT_NAME': 'VIEW_COUNT'}, inplace=True)

    conversion_counts = df[(df['EVENT_NAME'] == 'order') | (df['EVENT_NAME'] == 'product_added')].groupby(['USER_ID', 'ITEM_ID'])['EVENT_NAME'].count().reset_index()
    conversion_counts.rename(columns={'EVENT_NAME': 'CONVERSION_COUNT'}, inplace=True)

    result_df = pd.merge(df, view_counts, on=['USER_ID', 'ITEM_ID'], how='left').fillna(0)
    df = pd.merge(result_df, conversion_counts, on=['USER_ID', 'ITEM_ID'], how='left').fillna(0)
    
    df['EVENT_TIMESTAMP'] = pd.to_datetime(df['EVENT_TIMESTAMP'])

    df = df.sort_values(['USER_ID', 'ITEM_ID', 'EVENT_TIMESTAMP'], ascending=[True, True, False])

    df = df.drop_duplicates(['USER_ID', 'ITEM_ID'], keep='first')

    df = df.reset_index(drop=True)
    df['TIMESTAMP'] = df['EVENT_TIMESTAMP']
    
    df['CONVERSION_COUNT'] = np.where(df['CONVERSION_COUNT']<0.5,0,1)
    
    conversion_1_group = df[df['CONVERSION_COUNT'] == 1]
    conversion_0_group = df[df['CONVERSION_COUNT'] == 0]

    sample_size = min(len(conversion_1_group), len(conversion_0_group))

    sampled_conversion_1 = conversion_1_group.sample(n=sample_size, random_state=42)

    sampled_conversion_0 = conversion_0_group.sample(n=sample_size, random_state=42)

    balanced_df = pd.concat([sampled_conversion_1, sampled_conversion_0])
    
    df = df.sample(frac = 1)

    df = balanced_df.reset_index(drop=True)
    
    return df

#THIS FUNC CREATES A COLUMN THAT WILL ESTIMATE INTEREST BASED ON # OF VIEWS AND TIME SINCE LAST VIEW
def add_interact_decay(df, halflife = 90):
    #Maybe I want to just use the latest time per item customer combo
    now = datetime.now(timezone.utc)
    df = df.groupby(['USER_ID', 'ITEM_ID']).size().reset_index(name='VIEWCOUNT').merge(df)
    #df.groupby(['USER_ID'])['EVENT_TIMESTAMP'].max().reset_index(name='LATESTTIMESTAMP')
    df['DAYSSINCE'] = (now - pd.to_datetime(df['TIMESTAMP'])).dt.days
    df['DAYSSINCEDECAY'] = np.exp((-1/halflife) * df['DAYSSINCE'])
    df['INTERACTIONDECAY'] = df['VIEW_COUNT'] * df['DAYSSINCEDECAY']
    df = df.drop(columns = {"TIMESTAMP"})
    return df

#NOT USED
def negative_sample(df, multiplier = 1):
    negative_df = df[['USER_ID', 'ITEM_ID', 'TARGET', 'TIMESTAMP']]
    sections = df.shape[0]//10000
    negative_data = []
    item_map = df['ITEM_ID'].unique()
    rng = np.random.default_rng()
    zero_multiplier = multiplier #https://datascience.stackexchange.com/questions/6939/ratio-of-positive-to-negative-sample-in-data-set-for-best-classification
    #The above datascience stackexchange says 1:1 is a good ratio, i didn't look into it that much tho

    for chunk in np.array_split(df, sections):
        user_id_counts = pd.value_counts(chunk.USER_ID)
        item_ids = []
        for user_id, count in user_id_counts.items():
            item_ids.append(np.random.randint(low = 0, high = item_map.shape[0], size = count*zero_multiplier))
        item_ids = np.concatenate(item_ids)
        negative_data.append(pd.DataFrame({'USER_ID': np.repeat(user_id_counts.index, repeats = user_id_counts.values*zero_multiplier),
                                           'ITEM_ID': item_map[item_ids],
                                           'TIMESTAMP': date(1970, 1, 1),
                                           'TARGET': 0}))
    negative_data.append(negative_df)
    sample_data = pd.concat(negative_data)
    sample_data = sample_data.reset_index(drop = True)
    return sample_data


#ENDED UP NOT BEING USED, BUGGY IF BUT USED IF MANUALLY CREATING NEGATIVE SAMPLES
#WHEN USING THE MERLIN MODEL FRAMEWORK SHOULD NOT BE NECESSARY AS THERE ARE WAYS
#MM DOES THIS ALREADY
def merge(negative_df, df):
    item_df = df[['STYLE', 'USER_ID', 'PRICE_INFORMATION', 'AVG_REVIEW_SCORE',
              'ITEM_ID', 'TAXONOMY_STYLE', 'COLOR_NAME', 'PRODUCT_CLASS',
              'PRODUCT_SUBCLASS', 'TEAM', 'FRANCHISE', 'PRODUCT_GROUP', 'EVENT_NAME', 'EVENT_TIMESTAMP']]
    
    user_df = df[['USER_ID', 'ITEM_ID', 'COUNTRY', 'DERIVED_GENDER_BY_NAME', 'CLICKSTREAM_EVENTS_TOTAL', 'FIRST_PURCHASE_AT', 'FIRST_VISIT_AT',
              'LATEST_VISIT_AT', 'LATEST_PURCHASE_AT']]
    
    df_subset = negative_df.merge(user_df, on = ['USER_ID', 'ITEM_ID'], how = 'left').sort_values(['USER_ID', 'TARGET'], ascending=False).drop_duplicates(subset = ['USER_ID', 'ITEM_ID']).fillna(method='ffill')
    df_subset = df_subset.merge(item_df, on = ['USER_ID', 'ITEM_ID'], how = 'left').sort_values(['ITEM_ID', 'TARGET'], ascending=False).drop_duplicates(subset = ['USER_ID', 'ITEM_ID']).fillna(method='ffill')
    return df_subset

#SPLIT TRAIN AND VALIDATION AND CREATE PARQUET FILE
def train_valid_split_to_parquet(df, ratio = .2):
    data_size = df.shape[0]

    train_split_ratio = int(ratio * data_size)
    
    df = df.sample(frac = 1, random_state = 42)

    train = df[:-train_split_ratio]
    valid = df[-train_split_ratio:]
    train.to_parquet("train.parquet")
    valid.to_parquet("valid.parquet")
    
    return train, valid

#DATA ENGINEERING PIPELINE
def nvtabular_pipeline():
    
    user_id = ["USER_ID"] >> Categorify(dtype = "int32", out_path='categories') >> TagAsUserID()
    item_id = ["ITEM_ID"] >> Categorify(dtype = "int32", out_path='categories') >> TagAsItemID()
    
    target = ["CONVERSION_COUNT"] >> AddTags(["binary_classification", "target"]) >> Rename(name="INTERACTION_BINARY")
    
    user_features = (["CLICKSTREAM_EVENTS_TOTAL", "COUNTRY", "DERIVED_GENDER_BY_NAME"] >> Categorify() >> Normalize() >> TagAsUserFeatures())
    
    item_features = ([
        "STYLE",
        "TAXONOMY_STYLE",
        "COLOR_NAME",
        "PRODUCT_SUBCLASS",
        "TEAM",
        "FRANCHISE",
        "PRODUCT_GROUP"
    ] >> Categorify() >> Normalize() >> TagAsItemFeatures())
    
    decay = ['INTERACTIONDECAY'] >> Normalize() >> AddTags(['regression', 'target', 'continuous'])
    
    output = user_id + item_id + user_features + item_features + target + decay

    workflow_fit_transform(output, 'train.parquet', 'valid.parquet', 'integration') 
    
    train = merlin.io.Dataset(
        os.path.join("integration", "train"), engine="parquet"
    )
    valid = merlin.io.Dataset(
        os.path.join("integration", "valid"), engine="parquet"
    )

    return train, valid, output

#PUT IT ALL TOGETHER NOW
def preprocessing(order_weight = 1, added_weight = .5):
    df = read_target_sample()
    df1 = add_interact_decay(df)
    train_valid_split_to_parquet(df1)
    train, valid, output = nvtabular_pipeline()
    return train, valid, output

#DLRM MODEL IS PREFERRED MODEL BUT CAN BE BUGGY
#FINISHED WITH ~98% RECALL AND NDCG @ 10-20
#AUC IS SUPERIOR TO NCF
def dlrm(train, valid, epoch = 1, lr = .0175):
    model = mm.DLRMModel(
        train.schema,
        embedding_dim=64,
        bottom_block=mm.MLPBlock([128, 64]),
        top_block=mm.MLPBlock([128, 64, 32]),
        prediction_tasks=mm.BinaryOutput('INTERACTION_BINARY'),
    )

    model.compile(optimizer=keras.optimizers.Adam(learning_rate = lr), run_eagerly=False, metrics=[tf.keras.metrics.AUC(), mm.RecallAt(10), mm.NDCGAt(10)]);
    model.fit(train, validation_data=valid, batch_size=1024, epochs = epoch); 
    print(model.evaluate(valid, batch_size=1024, return_dict=True))
    return model.predict(valid, batch_size = 1024)

#ALSO A VALID MODEL SIMILAR RECALL AND NDCG TO DLRM (~98%) BUT AUC IS WORSE BY AROUND 7%
def ncf_model(train, valid, lr = .0175, epoch = 1):
    model = mm.benchmark.NCFModel(
        train.schema,
        embedding_dim=64,
        mlp_block=mm.MLPBlock([128, 64]),
        prediction_tasks=mm.BinaryOutput(train.schema.select_by_tag(Tags.TARGET).column_names[0]),
    )

    model.compile(optimizer=keras.optimizers.Adam(learning_rate = lr), run_eagerly=False, metrics=[tf.keras.metrics.AUC(), mm.RecallAt(10), mm.NDCGAt(10)]);
    model.fit(train, validation_data=valid, batch_size=1024, epochs = epoch); #Less epochs, more accurate valid... less accurate train
    print(model.evaluate(valid, batch_size = 1024, return_dict = True))
    return model.predict(valid, batch_size = 1024)