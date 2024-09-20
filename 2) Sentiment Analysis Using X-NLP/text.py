import pandas as pd

df=pd.read_csv('twitter_training.csv', encoding="utf-8")
df.dropna()

df.columns=['year', 'name', 'label', 'text']
print(df.head())
df_new=df.iloc[:10000,:]
df_new.to_csv('twitter.csv')