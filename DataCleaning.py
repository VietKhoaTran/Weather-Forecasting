import pandas as pd
df = pd.read_csv('Data\GlobalWeatherRepository.csv')

# HANDLE MISSING VALUES
missing_cols = df.columns[df.isnull().any()].tolist()

# Fill in with mean values
# for i in missing_cols:
#     df[i].fillna(df[i].mean(), inplace = True) 

# Fill in using KNNImputer
from sklearn.impute import KNNImputer
imputer = KNNImputer()
for i in missing_cols:
    df[[i]] = imputer.fit_transform(df[[i]])

print(df.isnull().sum())

# ANOMALY DETECTION using Isolation forest

from sklearn.ensemble import IsolationForest

num_cols = df.select_dtypes(include='number').columns.tolist()
num_cols = [col for col in num_cols if col not in ['last_updated_epoch', 'visibility_km', 'visibility_miles', 'latitude', 'longitude']]
# num_cols = ['temperature_celsius', 'precip_mm']
forest = IsolationForest(contamination = 0.01)
predictions = forest.fit_predict(df[num_cols])

import matplotlib.pyplot as plt
plt.scatter(df[predictions == 1]['temperature_celsius'], 
            df[predictions == 1]['precip_mm'], 
            color = 'blue', label="Normal", alpha=0.5)

plt.scatter(df[predictions == -1]['temperature_celsius'], 
            df[predictions == -1]['precip_mm'], 
            color='red', label="Anomaly", alpha=0.5)
plt.show()

df['Anomalies'] = predictions
df =  df[df['Anomalies'] == 1].drop(columns = ['Anomalies']) #Removing outliers
print(df.info())

plt.scatter(df['temperature_celsius'], df['precip_mm'])
plt.show()

print(num_cols)
# NORMALIZE DATA
# I will use min_max scale for bounded variables and standard scale 
# for unbounded ones

bounded_cols = [
    'wind_mph', 'wind_kph', 'wind_degree', 'humidity', 'cloud','uv_index', 
    'gust_mph', 'gust_kph', 'air_quality_us-epa-index', 'air_quality_gb-defra-index', 'moon_illumination'
]
unbounded_cols = ['temperature_celsius', 'temperature_fahrenheit', 'feels_like_celsius', 'feels_like_fahrenheit', 
'pressure_mb', 'pressure_in', 'precip_mm', 'precip_in', 'air_quality_Carbon_Monoxide', 'air_quality_Ozone', 
'air_quality_Nitrogen_dioxide', 'air_quality_Sulphur_dioxide', 'air_quality_PM2.5', 
'air_quality_PM10'
]

from sklearn.preprocessing import MinMaxScaler, StandardScaler

min_max = MinMaxScaler()
df[bounded_cols] = min_max.fit_transform(df[bounded_cols])
standard = StandardScaler()
df[unbounded_cols] = standard.fit_transform(df[unbounded_cols])
print(df.head())

df.to_csv('Data/Cleaned_data.csv', index = False)