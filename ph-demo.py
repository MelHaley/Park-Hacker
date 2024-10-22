# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import user_predictor as up


# load parks dataframe
parks_df = pd.read_csv('data/parks_df.csv')

# streamlit app

st.image('background.png', caption='Image by macrovector on Freepik') 
st.title('**Park Hacker**')
st.subheader('_navigate the most efficient route through a theme park_')
st.write("Every year, many families and groups plan vacations that are centered around visiting theme parks. While various theme parks have different assortments of things to do, one thing that they unfortunately all have in common is long, boring lines.")
st.write("Current ride wait time data is available via park-specific apps, but they are of limited use in planning your trip or if you'd rather not spend your day at the park looking up wait times on your phone.")
st.write("With ParkHacker, you simply enter the park you wish to visit, the day of your visit, the time you plan to arrive at the park, and the rides you want to go on. In return, Park Hacker will output a customized itinerary to minimize wait times!")

# park choice
park_choice = st.radio(
    'Which park are you visiting?',
    parks_df['code'].unique())

# date choice
date_choice = st.date_input(
    "What day are you planning to visit?",
    datetime.date(2023, 5, 15))

# time choice
time_choice = st.time_input('When will you start your day?', datetime.time(9, 0))

# ride choices
ride_choices = []
if park_choice == 'Animal Kingdom':
    ride_choices = st.multiselect(
        'What rides do you want to go on? (pick 2 minimum)',
        parks_df['short_name'].where(parks_df['code'] == 'Animal Kingdom').dropna().sort_values())
elif park_choice == 'Epcot Center':
    ride_choices = st.multiselect(
        'What rides do you want to go on? (pick 2 minimum)',
        parks_df['short_name'].where(parks_df['code'] == 'Epcot Center').dropna().sort_values())
elif park_choice == 'Hollywood Studios':
    ride_choices = st.multiselect(
        'What rides do you want to go on? (pick 2 minimum)',
        parks_df['short_name'].where(parks_df['code'] == 'Hollywood Studios').dropna().sort_values())
elif park_choice == 'Magic Kingdom':
    ride_choices = st.multiselect(
        'What rides do you want to go on? (pick 2 minimum)',
        parks_df['short_name'].where(parks_df['code'] == 'Magic Kingdom').dropna().sort_values())
       

def hour_min(minutes):
    """function to convert minutes to hour/minutes"""
    return "%02d:%02d " % (divmod(minutes, 60))
 
    
# make dict of user input
user_input = {'park': park_choice, 
              'date': date_choice, 
              'time': time_choice,
              'rides': ride_choices}

     
# create features from user input
if len(ride_choices) >= 2:
    df_x = up.input_df(user_input)

# get predictions
    ride_pred = pd.DataFrame()
    ride_list = []
    for ride in ride_choices:
        ride_list.append(parks_df['ride'].loc[(parks_df['short_name'] == ride)].item())
    ride_pred = up.get_predictions(ride_list, df_x) 

# get itinerary and comparison
    itinerary = up.get_itinerary(ride_pred, user_input)
    alternate = up.get_alternate(ride_pred, user_input)
  
    if itinerary:
        st.write(f"Your Itinerary for {park_choice} on {date_choice} starting at {time_choice}:")
        # total_time = 0
        # compared_to = 0
        for i in range(len(ride_choices)):
            st.write(f"{i+1}. {hour_min(itinerary[0][i][2])} | {  itinerary[0][i][0]}   (approx. wait = {int(itinerary[0][i][1])}mins)")
        total_time = int(itinerary[1])
        compared_to = int(alternate[1])
        saved = compared_to - total_time
        st.write(f"Total time: {hour_min(total_time)}")
    
        if st.button("compare"):
            st.write(f"compared to {hour_min(compared_to)}=>  {hour_min(saved)} mins saved!")
        
        
        
     
