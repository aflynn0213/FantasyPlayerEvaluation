import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Input, Masking
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
    
    # Convert data to sequences
    input_data = input_data.filter(lambda x: len(x) >= 3).groupby("name")
    seqs = input_data.apply(lambda x: x[feats].values.tolist()).reset_index(name='features')
    seqs.set_index("name", inplace=True)

    # Include year and age of the player from the last season
    seqs['last_season_info'] = input_data.apply(lambda x: x[['season', 'age']].iloc[-1].tolist()).tolist()
    x_train = seqs.apply(lambda row: row['features'][:-1], axis=1).tolist()

    # Pad sequences using pad_sequences
    pad_x = pad_sequences(x_train, padding='pre', dtype='float32', value=0)
    ply = 0
    for seq in pad_x:
        index = 0
        for season in seq:
            if season[0] == 2015:
                tmp = seq[0]
                seq[0] = season
                seq[index] = tmp
                index+=1
                continue
            elif season[0] == 2016:
                tmp = seq[1]
                seq[1] = season
                seq[index] = tmp
                index+=1
                continue
            elif season[0] == 2017:
                tmp = seq[2]
                seq[2] = season
                seq[index] = tmp
                index+=1
                continue    
            elif season[0] == 2018:
                tmp = seq[3]
                seq[3] = season
                seq[index] = tmp
                index+=1
                continue
            elif season[0] == 2019:
                tmp = seq[4]
                seq[4] = season
                seq[index] = tmp
                index+=1
                continue
            elif season[0] == 2020:
                tmp = seq[5]
                seq[5] = season
                seq[index] = tmp
                index+=1
                continue
            elif season[0] == 2021:
                tmp = seq[6]
                seq[6] = season
                seq[index] = tmp
                index+=1
                continue
            elif season[0] == 2022:
                tmp = seq[7]
                seq[7] = season
                seq[index] = tmp
                index+=1
                continue
            else:
                index+=1
                continue
        pad_x[ply] = seq
            
            
    print(pad_x)       

    # Handle padding for additional information (season and age)
    sea_age = pad_sequences(seqs['last_season_info'], padding='post', dtype='float32', value=-1)

    pad_x = np.array(sorted_pad_x)
    sea_age = np.array(sea_age).reshape(-1, 2)
    
    # Create a 3D array by concatenating info to each element in pad_x for each player
    x = [np.concatenate([seq, np.tile(info, (seq.shape[0], 1))], axis=1) for seq, info in zip(pad_x, sea_age)]
    x = np.array(x)
    print(x.shape)  # Check the shape
    
    y = np.array([latest for latest in seqs['features'].apply(lambda x: x[-1][-3:]).tolist()], dtype='float32')
    
    # Create a sequential model
    model = Sequential()
    model.add(Input(shape=(8, len(feats) + 2)))  # Update the input shape
    model.add(Masking(mask_value=-1))
    # Add a GRU layer with 50 units
    model.add(GRU(50, activation='relu'))

    # Add a Dense layer for output
    model.add(Dense(units=output_dim, activation='linear'))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    hist = model.fit(x, y, epochs=10, batch_size=32, validation_split=0.2)
    print(hist.history)

    pred_inputs = input_data.apply(lambda x: x[feats].values.tolist()).reset_index(name="features")
    players_2023 = seqs[seqs['last_season_info'].apply(lambda x: x[0] == 2023)]
    
    i = 0
    for ply in players_2023['last_season_info']:  
        players_2023['last_season_info'][i][0] = ply[0]+1
        players_2023['last_season_info'][i][1] = ply[1]+1
        i+=1
    print(players_2023["last_season_info"].shape)
    
    ny_age = np.array(players_2023["last_season_info"])
    feats = players_2023.apply(lambda row: row['features'], axis=1).tolist()
    feats = pad_sequences(feats, padding='post', dtype='float32', value=0)
    feats = np.array(feats)
    x = [np.concatenate([seq, np.tile(info, (seq.shape[0], 1))], axis=1) for seq, info in zip(feats, ny_age)]
    x = np.array(x)
    print(x.shape)
    preds = model.predict(x)
    print(preds)



