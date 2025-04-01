# Weather-Forecasting

This project involves building a model to predict weather trends and analyze correlations. The process follows three key steps: Data Preprocessing, Exploratory Data Analysis (EDA), and Model Building.

Data Preprocessing

At this stage, I performed several tasks to clean and prepare the dataset:
Handling Missing Values: Used KNNImputer to fill in missing values.


Handling Anomalies: Applied Isolation Forest to detect anomalies, which were then removed. Below is an example of anomalies detected in precipitation and temperature using Isolation Forest.
![image](https://github.com/user-attachments/assets/5567a598-a6cf-4ce6-81e6-54b6a9da9759)

![image](https://github.com/user-attachments/assets/f82cca95-acf5-40bb-ae23-7cb32f35a6d5)

Normalization: Applied different scaling techniques based on data characteristics:
MinMaxScaler for bounded columns (e.g., wind degree).
StandardScaler for unbounded columns (e.g., temperature in Celsius)
This approach ensures better normalization and consistency across variables.

EDA

I Used Pearson correlation to identify the top 20 column pairs with the highest correlations.

![image](https://github.com/user-attachments/assets/55ae5782-84f2-4229-9263-210265c088c1)

Visualized temperature and precipitation trends by plotting:
Distributions of precipitation and temperature.

![image](https://github.com/user-attachments/assets/239fc0fd-a6a7-4ae5-9e49-5667f96b1e11)

Correlation between these two parameters.

![image](https://github.com/user-attachments/assets/ef44e900-e2fb-4ef3-8e65-b7f566a1f5e8)

Model building and time series analysis

I chose LSTM (Long Short-Term Memory) for model building.
Regression models were initially tested but performed poorly, leading to a switch to LSTM, which provided significantly better results.

![image](https://github.com/user-attachments/assets/a2ffc98d-1e1d-45f8-a0c0-95863b437d35)


For time series analysis, since the dataset was already in a suitable format, minimal preprocessing was required.


Below is a visualization showing how different parameters change over several months.

![image](https://github.com/user-attachments/assets/b00b94a8-ae75-414e-b838-3ec7668b179f)


