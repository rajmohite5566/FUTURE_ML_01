import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

df = dataset.copy()
df['Date'] = pd.to_datetime(df['Date'])

#Prophet
prophet_df = df.rename(columns={'Date': 'ds', 'Sales': 'y'})

model = Prophet()
model.fit(prophet_df)

#forcasting monthly for 12 months
future = model.make_future_dataframe(periods=12, freq='M')
forecast = model.predict(future)

#trends and charts
plt.figure(figsize=(10,6))
plt.plot(df['Date'], df['Sales'], label='Actual Monthly Sales', color='Blue')  
plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='Red')    
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                 color='#98df8a', alpha=0.4, label='Confidence Interval')      

#ploting x and y labels
plt.xlabel('Date')
plt.ylabel('Sales Of Products')
plt.title('Monthly Sales Forecast - Actual vs Predicted')
plt.legend()
plt.grid(True)
plt.show()
