from prophet import Prophet
from prophet.diagnostics import cross_validation

m = Prophet()
m.make_future_dataframe(periods=365)