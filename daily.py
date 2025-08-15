import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

#dataset for BI
df = dataset.copy()

#format
df['Date'] = pd.to_datetime(df['Date'])

#prophet 
prophet_df = df.rename(columns={'Date': 'ds', 'Sales': 'y'})

model = Prophet()
model.fit(prophet_df)

#forcast model 
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

#multi line chart 
plt.figure(figsize=(10,6))
plt.plot(df['Date'], df['Sales'], label='Actual Sales', color='blue')
plt.plot(forecast['ds'], forecast['yhat'], label='Forecasting Trends', color='orange')
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                 color='lightgreen', alpha=0.4, label='Confidence Interval(Future Value)')

#ploting x and y labels
plt.xlabel('Dates')
plt.ylabel('Sales Of Products')
plt.title('Sales Forecasting(Daily Trends) - Actual vs Predicted')
plt.legend()
plt.grid(True)
plt.show()
