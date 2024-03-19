import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np

if __name__ == "__main__":
    # Assuming your input shape is (num_samples, num_timesteps, num_features)
    players_data = pd.read_csv('fangraphs_stats.csv')
    players_data.columns = players_data.columns.str.lower()

    # FIGURE OUT FEATURES AND LABELS
    feats = players_data.loc[:, ["name", "season", "age", "bb%", "k%", "barrel%"]]
    input_data = feats.groupby("name")
    # Sort each group by 'season' in descending order
    input_data = input_data.apply(lambda x: x.sort_values(by='season', ascending=True)).reset_index(drop=True).groupby("name")
    feats = feats.columns.drop(["name"])
    output_dim = len(feats.drop(["season", "age"]))
    input_shape = (8, len(feats) + 2)  # Update the input shape to include year and age

    # Convert data to sequences
    input_data = input_data.filter(lambda x: len(x) >= 3).groupby("name")
    seqs = input_data.apply(lambda x: x[feats].values.tolist()).reset_index(name='features')
    seqs.set_index("name", inplace=True)

    # Include year and age of the player from the last season
    seqs['last_season_info'] = input_data.apply(lambda x: x[['season', 'age']].iloc[-1].tolist()).tolist()
    x_train = seqs.apply(lambda row: row['features'][:-1], axis=1).tolist()

    # Pad sequences using pad_sequences
    pad_x = pad_sequences(x_train, padding='post', dtype='float32', value=0)

    # Handle padding for additional information (season and age)
    sea_age = pad_sequences(seqs['last_season_info'], padding='post', dtype='float32', value=0)

    pad_x = np.array(pad_x)
    sea_age = np.array(sea_age).reshape(-1, 2)
    
    # Create a 3D array by concatenating info to each element in pad_x for each player
    x = [np.concatenate([seq, np.tile(info, (seq.shape[0], 1))], axis=1) for seq, info in zip(pad_x, sea_age)]
    x = np.array(x)
    print(x.shape)  # Check the shape
    
    y = np.array([latest for latest in seqs['features'].apply(lambda x: x[-1][-3:]).tolist()], dtype='float32')
    
    # Create a sequential model
    model = Sequential()
    model.add(Input(shape=(8, len(feats) + 2)))  # Update the input shape

    # Add a GRU layer with 50 units
    model.add(GRU(50, activation='relu'))

    # Add a Dense layer for output
    model.add(Dense(units=output_dim, activation='linear'))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    hist = model.fit(x, y, epochs=10, batch_size=32, validation_split=0.2)
    print(hist.history)

    pred_inputs = input_data.apply(lambda x: x[feats].values.tolist()).reset_index(name="features")
    
    x_preds = [np.concatenate([seq, np.tile(info, (seq.shape[0], 1))], axis=1) for seq, info in zip(pad_x, sea_age)]
    x_preds = np.array(x_preds)
    print(x.shape)
    preds = model.predict(input_data)
