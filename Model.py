import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv("Data/Cleaned_data.csv")

predictor_variables = ['temperature_celsius', 'humidity', 'pressure_mb', 'wind_kph', 'cloud']
target_variable = 'temperature_celsius'

x = df[predictor_variables].values  
y = df[target_variable].values  
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# MODEL BUILDING

# LSTM Model
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(len(predictor_variables),)))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='linear'))  

model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001), metrics=['mae'])

model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

loss, mae = model.evaluate(x_test, y_test)
from sklearn.metrics import r2_score

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.2f}")
print(f"Mean Absolute Error for Neural Network model: {mae:.2f}")
model.save('Model/LSTM_model.h5')

# Time series analysis

df_sorted = df.sort_values(by='last_updated')
df_sorted.to_csv('Data/Sorted_dataset.csv', index=False)

df = pd.read_csv('Data/Sorted_dataset.csv', parse_dates= ['last_updated'], index_col = 'last_updated')
numerical_cols = df.select_dtypes(include = ['float64', 'int64']).columns.to_list()


fig, axes = plt.subplots(5, 6, figsize=(20, 15))
axes = axes.flatten() 
for i in range(-1, len(numerical_cols)-2):
    col_name = numerical_cols[i]
    mean_values = df[col_name].resample('ME').mean()

    ax = axes[i+1]
    ax.plot(mean_values, marker='o', label=col_name)
    ax.set_xticks(mean_values.index) 
    ax.set_xticklabels(mean_values.index.strftime('%b'), rotation=45, fontsize = 8) 
    ax.set_ylabel(numerical_cols[i], fontsize = 8)

plt.tight_layout()
plt.show()
