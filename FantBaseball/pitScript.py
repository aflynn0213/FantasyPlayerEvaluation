# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 09:36:37 2024

@author: lu516e
"""
import pandas as pd
import math as ma
#import statistics as st

excel_file = "ERA_VAR.xlsx"

df = pd.read_excel(excel_file)

pitcher_counts = df['Name'].value_counts()

# Filter out pitchers that appear less than 3 times
filtered_pitchers = pitcher_counts[pitcher_counts >= 5].index

# Filter the original DataFrame based on the selected pitchers
df = df[df['Name'].isin(filtered_pitchers)]
print(df)

df['IP'] = (df['IP']-df['IP']%1)+(df['IP']%1)*3.33333333
print(df['IP'])
eras = df.groupby('Name')['ERA']
wera_mean = df.groupby('Name').apply(lambda group: (group['ERA']*group['IP']).sum()/group['IP'].sum())
wfip_mean = df.groupby('Name').apply(lambda group: (group['FIP']*group['IP']).sum()/group['IP'].sum())

print("WEIGHTED MEAN ERA BY PLAYER: ",wera_mean)
print("WEIGHTED MEAN FIP BY PLAYER: ",wfip_mean)

wera_var = df.groupby('Name').apply(lambda group: (group['IP']*(group['ERA']-wera_mean.loc[group.name])**2).sum() / group['IP'].sum())
wfip_var = df.groupby('Name').apply(lambda group: (group['IP']*(group['FIP']-wera_mean.loc[group.name])**2).sum() / group['IP'].sum())

print("WEIGHTED VARIANCE ERA BY PLAYER: ", wera_var)
print("WEIGHTED VARIANCE FIP BY PLAYER: ", wfip_var)

print("AVERAGE PLAYER ERA VARIANCE: ",wera_var.mean())
print("AVERAGE PLAYER FIP VARIANCE: ",wfip_var.mean())

wera_mean_lg = (wera_mean*df.groupby('Name')['IP'].sum()).sum() / df['IP'].sum()
wfip_mean_lg = (wfip_mean*df.groupby('Name')['IP'].sum()).sum() / df['IP'].sum()

print("LEAGUE ERA MEAN: ",wera_mean_lg)
print("LEAGUE FIP MEAN: ",wfip_mean_lg)

wera_var_lg = (df.groupby('Name')['IP'].sum()*(wera_mean-wera_mean_lg)**2).sum() / df['IP'].sum()
wfip_var_lg = (df.groupby('Name')['IP'].sum()*(wfip_mean-wfip_mean_lg)**2).sum() / df['IP'].sum()

print("VARIANCE AGAINST LEAGUE ERA: ",wera_var_lg)
print("VARIANCE AGAINST LEAGUE ERA: ",wfip_var_lg)

# print("Overall ERA Variance:", wera_var)

# print("Overall FIP Variance:", wfip_var)