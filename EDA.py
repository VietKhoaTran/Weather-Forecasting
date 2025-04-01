import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("Data\Cleaned_data.csv")
#Use pearson method to find top 20 pairs of variables that have highest correlation

Pearson = df.select_dtypes(include=['float64', 'int64']).corr(method='pearson')
Pearson_unstacked = Pearson.abs().unstack()
Pearson_unstacked = Pearson_unstacked[Pearson_unstacked < 1]

top_20 = Pearson_unstacked.sort_values(ascending=False).head(40)
top_20_df = top_20.reset_index()
top_20_df = top_20_df.iloc[::2]
top_20_df = top_20_df.reset_index(drop=True)

fig, axes = plt.subplots(4, 5, figsize = (15, 12))
for i, ax in enumerate(axes.flatten()):
    df.plot(kind = 'scatter', x = top_20_df.loc[i, 'level_0'], y = top_20_df.loc[i, 'level_1'], ax = ax)

plt.tight_layout()
plt.show()

#Visualization for temperatures and precipitation
#Histograms
plt.subplot(1, 2, 1)
plt.hist(df['temperature_celsius'], bins = 30, color ='blue')
plt.title('Temperature Distribution')
plt.xlabel('Temperature')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(df['precip_mm'], bins = 30, color ='blue')
plt.title('Precipitation Distribution')
plt.xlabel('Precipitation')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

#Scatter plot
df.plot(kind = 'scatter', x = 'temperature_celsius', y = 'precip_mm')
plt.tight_layout()
plt.show()