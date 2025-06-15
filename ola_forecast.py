# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error as mae
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import holidays
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('data/ola.csv')

# Feature Engineering
# Extract date and time features
df['datetime'] = pd.to_datetime(df['datetime'])
df['date'] = df['datetime'].dt.date
df['time'] = df['datetime'].dt.hour
df['day'] = df['datetime'].dt.day
df['month'] = df['datetime'].dt.month
df['year'] = df['datetime'].dt.year

# Add weekday feature (0 for weekend, 1 for weekday)
df['weekday'] = df['datetime'].apply(lambda x: 0 if x.weekday() > 4 else 1)

# Add AM/PM feature (0 for AM, 1 for PM)
df['am_or_pm'] = df['time'].apply(lambda x: 1 if x > 11 else 0)

# Add holiday feature (1 if holiday, 0 otherwise)
def is_holiday(date_obj):
    india_holidays = holidays.country_holidays('IN')
    return 1 if india_holidays.get(date_obj) else 0

df['holidays'] = df['date'].apply(is_holiday)

# Drop unnecessary columns
df.drop(['datetime', 'date', 'registered', 'time'], axis=1, inplace=True)

# Remove outliers
df = df[(df['windspeed'] < 32) & (df['humidity'] > 0)]

# Split data into features and target
features = df.drop(['count'], axis=1)
target = df['count'].values

# Split into train and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(
    features, target, test_size=0.1, random_state=22
)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Model Training and Evaluation
models = [LinearRegression(), Lasso(), RandomForestRegressor(), Ridge()]

for model in models:
    model.fit(X_train, Y_train)
    print(f'{model.__class__.__name__}:')
    
    train_preds = model.predict(X_train)
    print('Training MAE:', mae(Y_train, train_preds))
    
    val_preds = model.predict(X_val)
    print('Validation MAE:', mae(Y_val, val_preds))
    print()
