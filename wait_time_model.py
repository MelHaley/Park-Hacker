import numpy as np
import pandas as pd
import dill
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit, cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklego.preprocessing import RepeatingBasisFunction

# load parks_df to get ride duration info
parks_df = pd.read_csv('data/parks_df.csv')

# times series model
# polynomial transform to model drift
class PolyTransform(BaseEstimator, TransformerMixin):
    """Make features x and x^2."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.c_[X, X**2]
    

model_drift = Pipeline([('drift', PolyTransform()),
                        ('regressor', LinearRegression())])

# fourier transform for seasonal components
class SeasonComponents(BaseEstimator, TransformerMixin):
    '''Returns wave'''
    def __init__(self, freq):
        self.freq = freq
    
    def fit(self, X, y=None):
        self.X0 = X[0]
        return self
    
    def transform(self, X):
        dt = (X - self.X0) * 2 * np.pi * self.freq
        return np.c_[np.sin(dt), np.cos(dt)]
    
    
# data resolution is 19hours/day (19*365 = 6935)
season_union = FeatureUnion([('year', SeasonComponents(1/6935)),
                             ('month', SeasonComponents(11/6935)),
                             ('week', SeasonComponents(6/6935))])
    
model_season = Pipeline([('season', season_union),
                         ('regressor', LinearRegression())])

# times series model
ts_union = FeatureUnion([('drift', PolyTransform()), 
                         ('season', season_union)])

ts_columns = ColumnTransformer([('index', 'passthrough', ['index'])])

model_drift_seasonal = Pipeline([('columns', ts_columns),
                                 ('union', ts_union),
                                 ('regressor', LinearRegression())])



# weather model
weather_v = ['weather_wdwprecip','wdwmeantemp','wdwmaxtemp', 'wdwmintemp']
weather_transform = ColumnTransformer([('weather', StandardScaler(), weather_v)])
weather_param = {'n_neighbors': np.arange(10,15,1)}
weather_gs = GridSearchCV(KNeighborsRegressor(), weather_param)

weather_model = Pipeline([('transform', weather_transform),('est', weather_gs)])


# cyclic model
cyclic_v = ['dayofweek', 'hourofday', 'monthofyear', 'dayofyear', 'weekofyear']
cyclic_transform = ColumnTransformer([('cyclic', OneHotEncoder(handle_unknown='ignore'), cyclic_v)])
cyclic_param = {'n_estimators' : np.arange(125,150,5), 'max_depth': np.arange(9,15,1)}
cyclic_gs = GridSearchCV(RandomForestRegressor(), cyclic_param)

cyclic_model = Pipeline([('transform', cyclic_transform),
                         ('est', RandomForestRegressor(max_depth=10, n_estimators=150))])


# holiday model
holiday_v = ['holidayn', 'holiday']
holiday_transform = ColumnTransformer([('category', OneHotEncoder(handle_unknown='ignore'), holiday_v)])
holiday_param = {'alpha': np.arange(20,30,1)}
holiday_gs = GridSearchCV(Ridge(), holiday_param)

holiday_model = Pipeline([('transform', holiday_transform), 
                          ('est', LinearRegression())])


# ensemble predictor
class ModelTransformer(BaseEstimator, TransformerMixin):
    '''takes an indivdual estimator and turns it into a transformer to 
    be used in an ensemble model'''
    def __init__(self, model):
        self.model = model
        self.y_pred = np.array
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.model.fit(self.X, self.y)
        return self
    
    def transform(self, X):
        self.X = X
        self.y_pred = np.array(self.model.predict(self.X))
        
        self.y_pred = self.y_pred.reshape(self.y_pred.shape[0], -1)
        return self.y_pred

    
weather_trans = ModelTransformer(weather_model)
cyclic_trans = ModelTransformer(cyclic_model)
holiday_trans = ModelTransformer(holiday_model)
ts_trans = ModelTransformer(ts_model)

union = FeatureUnion([('weather', weather_trans),
                      ('cyclic', cyclic_trans),
                      ('holiday', holiday_trans),
                      ('ts', ts_trans)])

ride_wt_model = Pipeline([('features', union), 
                          ('est', LinearRegression())])
                          #('est', RandomForestRegressor(max_depth=5, n_estimators=150))])


# residual model
class ResidualFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, column, window=400):
        """Generate features based on window statistics of past noise/residuals."""
        self.window = window
        self.column = column
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        roll = X[self.column].rolling(self.window).mean().fillna(0)
        return np.array(roll).reshape(-1,1)
    
    
residual_model = Pipeline([('residual', ResidualFeatures('index', window=400)),
                           ('regressor', KNeighborsRegressor(n_neighbors=3))])    
    
    
# full model
class FullModel(BaseEstimator, RegressorMixin):
    def __init__(self, baseline, residual_model):
        """Combine a baseline and residual model to predict any number of steps in the future."""
        
        self.baseline = baseline
        self.residual_model = residual_model
    
        
    def fit(self, X, y):
        self.baseline.fit(X, y)
        resd = y - self.baseline.predict(X)
        self.residual_model.fit(X, resd)
                
        return self
    
    def predict(self, X):
        y_b = self.baseline.predict(X)
        resd_pred = pd.Series(self.residual_model.predict(X))
        y_pred = y_b + resd_pred
        
        return y_pred

    
    
    

def get_predictions(ride_list, df_X):
    ride_fits = {}
    for ride in ride_list:
        fit_name = ride+'_fit'
        with open('data/ride_fits/'+fit_name, 'rb') as f:
            ride_fits[ride] = dill.load(f)
    pred_df = pd.DataFrame()
    for ride in ride_list:
        pred_df[ride] = ride_fits[ride].predict(df_X)
        ride_duration = parks_df['duration'].loc[(parks_df['ride'] == ride)].item()
        pred_df[ride] = pred_df[ride].add(ride_duration)
    pred_df['hourofday'] = sorted(hours)
    pred_df = pred_df.set_index('hourofday')
    return pred_df

hours = [11,12,13,14,15,16,17,18,19,20,21,22,23,0,9,10,1,8,7,6]

