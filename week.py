import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet


df = dataset.copy()
df['Date'] = pd.to_datetime(df['Date'])

prophet_df = df.rename(columns={'Date': 'ds', 'Sales': 'y'})

#model
model = Prophet()
model.fit(prophet_df)

#forecast 12 weeks
future = model.make_future_dataframe(periods=12, freq='W')
forecast = model.predict(future)

#trends
plt.figure(figsize=(10,6))
plt.plot(df['Date'], df['Sales'], label='Actual Weekly Sales', color='#2ca02c')  
plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='Magenta')    
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                 color='Pink', alpha=0.4, label='Interval')        

#ploting x and y labels
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Weekly Sales Forecasting - Actual vs Predicted')
plt.legend()
plt.grid(True)
plt.show()
