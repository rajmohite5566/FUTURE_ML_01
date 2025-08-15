import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

#dataset
df = dataset.copy()

#format
df.rename(columns={'Date': 'ds', 'Sales': 'y'}, inplace=True)

#format
df['ds'] = pd.to_datetime(df['ds'])

#model
model = Prophet()
model.fit(df)

#next 24 months forcating
future = model.make_future_dataframe(periods=24, freq='M')
forecast = model.predict(future)

forecast = forecast[['ds', 'yhat']]

#actual and forcasting model
merged = pd.merge(forecast, df, on='ds', how='left')
merged['Actual'] = merged['y']
merged['Forecast'] = merged['yhat']

merged['Year'] = merged['ds'].dt.year
yearly = merged.groupby('Year', as_index=False).agg({
    'Actual': 'sum',
    'Forecast': 'sum'
})

#chart
plt.figure(figsize=(10, 6))
bar_width = 0.35
x = range(len(yearly))

plt.bar(x, yearly['Actual'], width=bar_width, label='Actual', color='royalblue')
plt.bar([i + bar_width for i in x], yearly['Forecast'], width=bar_width, label='Forecast', color='orange')

plt.xticks([i + bar_width/2 for i in x], yearly['Year'])
plt.xlabel('Year')
plt.ylabel('Sales of Products')
plt.title('Yearly Actual sales vs Forecasting sales')
plt.legend()
plt.tight_layout()
plt.show()

#output
result = yearly
