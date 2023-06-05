# import libraries
import numpy as np
import pandas as pd
import datetime
import dill
import holidays
import bisect
from itertools import permutations
from collections import defaultdict

# load necessary data

# parks_df contains park, ride names, ride durations, ride file name w/o extension
parks_df = pd.read_csv('data/parks_df.csv')

# 'weather' contains average mean/max/min temps for each day of the year (at park location)
weather = pd.read_csv('data/weather-by-day.csv')

# 'us-holidays' contains holidays.US(subdiv='NY')
with open('data/us-holidays', 'rb') as f:
    us_holidays = dill.load(f)

# 'travel_df' contains the travel duration/distance between each ride in park    
travel_df = pd.read_csv('data/travel_df.csv')

# list of hours when park is open
hours = [11,12,13,14,15,16,17,18,19,20,21,22,23,0,9,10,1,8,7,6]

# list of features used in prediction
input_columns = ['date', 'hourofday', 'epoch', 'dayofweek', 'dayofyear', 'weekofyear',
       'monthofyear', 'year', 'season', 'holidayn', 'holiday',
       'weather_wdwprecip', 'wdwmintemp', 'wdwmeantemp', 'wdwmaxtemp']



def get_holidays(user_input):
    '''Checks to see if date is holiday'''
    if user_input['date'] in us_holidays:
        name = us_holidays.get(user_input['date'])
        if name in holiday_dict:
            return holiday_dict[name]
    else: 
        return 'no_holiday'

    
def input_df(user_input):
    '''Returns feature dataframe from user input'''
    df_X = pd.DataFrame(columns=input_columns, index=sorted(hours))
    day_of_year = user_input['date'].timetuple().tm_yday
    year_week_day = user_input['date'].isocalendar()
    for n in hours:
        df_X.loc[n, 'wdwmintemp'] = weather.iloc[day_of_year]['wdwmintemp'].astype('float')
        df_X.loc[n,'wdwmaxtemp'] = weather.iloc[day_of_year]['wdwmaxtemp'].astype('float')
        df_X.loc[n,'wdwmeantemp'] = weather.iloc[day_of_year]['wdwmeantemp'].astype('float')
        df_X.loc[n,'weather_wdwprecip'] = weather.iloc[day_of_year]['weather_wdwprecip'].astype('float')
        df_X.loc[n,'holiday'] = int(user_input['date'] in us_holidays)
        df_X.loc[n, 'dayofyear'] = day_of_year
        df_X.loc[n, 'dayofweek'] = year_week_day[2]
        df_X.loc[n, 'weekofyear'] = year_week_day[1]
        df_X.loc[n, 'monthofyear'] = user_input['date'].month
        df_X.loc[n, 'date'] = user_input['date']
        df_X.loc[n, 'holidayn'] = get_holidays(user_input)
        df_X.loc[n, 'hourofday'] = n
        df_X.loc[n, 'epoch'] = int(user_input['date'].strftime('%s'))
    return df_X.reset_index()


def get_predictions(ride_list, df_X):
    '''Returns wait time predictions from user input'''
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



def get_travel_time(origin, destination):
    '''Returns the travel duration (in mins) between pairs of rides in itinerary'''
    try:
        travel = travel_df[((travel_df['origin']==origin) & (travel_df['destination']==destination))].dropna()
        return travel['duration'].item()/60
    except:
        travel = travel_df[(travel_df['origin']==destination) & (travel_df['destination']==origin)].dropna()           
        return travel['duration'].item()/60



def get_itinerary(ride_choices, ride_pred, user_input):
    '''Generates all combinations of rides and returns the shortest itinerary'''
    start_time = (60* user_input['time'].hour) + user_input['time'].minute
    rides = user_input['rides']
    combos = permutations(rides, len(rides))
    options = defaultdict(list)
    day_max = []
    minutes = [x*60 for x in ride_pred.index]
    # get all ride combinations
    for i, combo in enumerate(combos):
        current_total = start_time
    # get wait + ride duration + travel for each ride in itinerary
        for j in range(len(combo)):
            ride = combo[j]
            column = parks_df['ride'].loc[(parks_df['short_name'] == ride)].item()
            if j == 0:
                travel = get_travel_time('entrance',ride)
            elif j > 0:
                travel = get_travel_time(combo[j-1],combo[j])
            current_total += travel
            current_time = bisect.bisect(minutes, current_total)
            current_wait = ride_pred[column].iloc[current_time]
            options[i].append((combo[j], current_wait, current_total))
            current_total += current_wait 
    # list of total time for each itinerary
        day_max.append(options[i][-1][1]+options[i][-1][2])
        total_time = (min(day_max) - start_time)
    return options[day_max.index((min(day_max)))], total_time 
        

    
def get_alternate(ride_choices, ride_pred, user_input):
    '''Generates all combinations of rides and returns the shortest itinerary'''
    start_time = (60* user_input['time'].hour) + user_input['time'].minute
    rides = user_input['rides']
    combo=sorted(rides, reverse=True)
    options = defaultdict(list)
    current_total = start_time
    minutes = [x*60 for x in ride_pred.index]
   #get wait + ride duration + travel for each ride in alternate itinerary
    for j in range(len(combo)):
        ride = combo[j]
        column = parks_df['ride'].loc[(parks_df['short_name'] == ride)].item()
        if j == 0:
            travel = get_travel_time('entrance',ride)
        elif j > 0:
            travel = get_travel_time(combo[j-1],combo[j])
        current_total += travel
        current_time = bisect.bisect(minutes, current_total)
        current_wait = ride_pred[column].iloc[current_time]
        options['alt'].append((combo[j], current_wait, current_total))
        current_total += current_wait 
    # list of total time for each itinerary
        day_max = options['alt'][-1][1]+options['alt'][-1][2]
        total_time = (day_max - start_time)
    return options['alt'], total_time    
    
    
    
    
        
def get_comparison(ride_choices, ride_pred, user_input):
    start_time = (60* user_input['time'].hour) + user_input['time'].minute
    rides = ride_choices.copy()
    combo=sorted(rides, reverse=True)
    options = defaultdict(list)
    options['combo'] = [[x, x] for x in list(combo)]
    numbers = [x*60 for x in ride_pred.index]
    current_time = start_time
    current_wait = 0
    for j in range(len(ride_choices)):
        ride = options['combo'][j][0] 
        column = parks_df['ride'].loc[(parks_df['short_name'] == ride)].item()
        x = bisect.bisect(numbers, current_time)
        current_wait = ride_pred[column].iloc[x] 
        options['combo'][j][1] = (current_wait, current_time)
        current_time += current_wait 
   
    if len(ride_choices) >= 2:
        
        return options
    

