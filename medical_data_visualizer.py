import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv("medical_examination.csv")


# 2
df['overweight'] = ((df['weight']/((df['height'])/100)**2))
df['overweight'] = df['overweight'].apply(lambda x: 1 if x > 25 else 0)


# 3
df[['cholesterol', 'gluc']] = df[['cholesterol', 'gluc']].map(lambda y: 0 if y == 1 else 1)


# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])

    # 8
    fig = sns.catplot(data=df_cat, x='variable', hue='value', col='cardio', kind='count')
    fig.set_ylabels('total')
    fig = fig.fig


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[(df['height'] >= df['height'].quantile(0.025)) & 
             (df['height'] <= df['height'].quantile(0.975)) & 
             (df['weight'] >= df['weight'].quantile(0.025)) & 
             (df['weight'] <= df['weight'].quantile(0.975))]

    # 12
    corr = df_heat.corr()

        

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))



    # 14
    fig, ax = plt.subplots(figsize=(11, 9))

    # 15
    sns.heatmap(corr, mask=mask, annot=True,fmt=".1f", cmap="magma",
        square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)


    # 16
    fig.savefig('heatmap.png')
    return fig
