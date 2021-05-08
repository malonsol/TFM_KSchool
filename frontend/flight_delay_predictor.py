
# Import libraries to be used

# Directories/Files management
import os

# Timing
import time
import datetime

# Objects storage:
import joblib
import pickle

# Online data retrieval:
import requests

# Data analysis and wrangling
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None) # Show all columns in DataFrames
## pd.set_option('display.max_rows', None) # It greatly slows down the output display and may freeze the kernel
import missingno as msno
from collections import Counter

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot') # choose a style: 'plt.style.available'
sns.set_theme(context='notebook',
              style="darkgrid") # {darkgrid, whitegrid, dark, white, ticks}
palette = sns.color_palette("flare", as_cmap=True);
import altair as alt

# Machine Learning:
# - Model selection:
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, KFold, cross_val_score, StratifiedKFold, \
                                    GridSearchCV, RandomizedSearchCV
from sklearn.inspection import permutation_importance

# - Basic classes for custom-made transformers:
from sklearn.base import BaseEstimator, TransformerMixin

# - Transformers:
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders.target_encoder import TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# - Pipeline:
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# - Models:
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, plot_importance, plot_tree

# - Metrics:
from sklearn.metrics import fbeta_score, f1_score, recall_score, precision_score, accuracy_score, \
                            confusion_matrix, classification_report, roc_curve, precision_recall_curve, \
                            roc_auc_score, average_precision_score, plot_roc_curve, plot_precision_recall_curve

# Model interpretability:
import shap

# Frontend:
import streamlit as st
import streamlit.components.v1 as components


# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------

# Define the dataset columns:
cols = [
    'MONTH',
    'DAY_OF_WEEK',
    'OP_UNIQUE_CARRIER',
    'ORIGIN',
    'DEST',
    'DEP_TIME_hour',
    'TAXI_OUT_median',
    'TAXI_IN_median',
    'ARR_DEL15', # → Target !!
    'ARR_TIME_hour',
    'DISTANCE',
    'LATITUDE_Origin',
    'LONGITUDE_Origin',
    'HourlyAltimeterSetting_Origin',
    'HourlyDryBulbTemperature_Origin',
    'HourlyPrecipitation_Origin',
    'HourlyRelativeHumidity_Origin',
    'HourlySkyConditions_Origin',
    'HourlyVisibility_Origin',
    'HourlyWindGustSpeed_Origin',
    'HourlyWindSpeed_Origin',
    'LATITUDE_Dest',
    'LONGITUDE_Dest',
    'HourlyAltimeterSetting_Dest',
    'HourlyDryBulbTemperature_Dest',
    'HourlyPrecipitation_Dest',
    'HourlyRelativeHumidity_Dest',
    'HourlySkyConditions_Dest',
    'HourlyVisibility_Dest',
    'HourlyWindGustSpeed_Dest',
    'HourlyWindSpeed_Dest',
]

cols_dtypes = {
    'MONTH' : 'string',
    'DAY_OF_WEEK' : 'string',
    'OP_UNIQUE_CARRIER' : 'string',
    'ORIGIN' : 'string',
    'DEST' : 'string',
    'DEP_TIME_hour' : 'string',
    'TAXI_OUT_median' : 'int32',
    'TAXI_IN_median' : 'int32',
    'ARR_DEL15' : 'int32', # → Target !!
    'ARR_TIME_hour' : 'string',
    'DISTANCE' : 'int32',
    'LATITUDE_Origin' : 'float64',
    'LONGITUDE_Origin' : 'float64',
    'HourlyAltimeterSetting_Origin' : 'float64',
    'HourlyDryBulbTemperature_Origin' : 'int32',
    'HourlyPrecipitation_Origin' : 'float64',
    'HourlyRelativeHumidity_Origin' : 'int32',
    'HourlySkyConditions_Origin' : 'string',
    'HourlyVisibility_Origin' : 'int32',
    'HourlyWindGustSpeed_Origin' : 'int32',
    'HourlyWindSpeed_Origin' : 'int32',
    'LATITUDE_Dest' : 'float64',
    'LONGITUDE_Dest' : 'float64',
    'HourlyAltimeterSetting_Dest' : 'float64',
    'HourlyDryBulbTemperature_Dest' : 'int32',
    'HourlyPrecipitation_Dest' : 'float64',
    'HourlyRelativeHumidity_Dest' : 'int32',
    'HourlySkyConditions_Dest' : 'string',
    'HourlyVisibility_Dest' : 'int32',
    'HourlyWindGustSpeed_Dest' : 'int32',
    'HourlyWindSpeed_Dest' : 'int32',
}

# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------

# # Load the data:
# @st.cache
# def load_data():
#     input_folder = '../../data/output/us_dot-noaa/'
#     file_name = "3_otp_lcd_2019.csv"

#     df = pd.read_csv(input_folder + file_name,
#                      encoding='latin1',
#     #                      nrows=1e5,
#                      usecols=cols,
#                      dtype=cols_dtypes
#                     )
#     X = df.drop(['ARR_DEL15'], axis=1)
#     y = df['ARR_DEL15']
#     return df, X, y

# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------

# Load the model:
# @st.cache
def load_model(path=""):
    model = joblib.load(path)
    return model
    
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------

def frontend_appearance():
    """
    Design frontend appearance
    """

    # frontend elements of the web page 
    html_temp = """ 
    <div style ="background-color:SteelBlue;padding:13px"> 
    <h1 style ="color:white;text-align:center;">Flight Delay Forecaster</h1> 
    </div> 
    """   

    # display the frontend aspect
    st.markdown(html_temp, unsafe_allow_html = True)

# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------  

@st.cache
def weather_forecast(lat, lon):
    
    # Get/Build the url:
    coord_API_endpoint = "http://api.openweathermap.org/data/2.5/onecall?"
    join_key = "&appid="
    API_key = "b51fbb9d87131aabef6d7c2cd42b128e"
    units = "&units=imperial"
    exclude = "&exclude=current,minutely,daily,alerts"

    lat_lon = "lat=" + str(round(lat, 2))+ "&lon=" + str(round(lon, 2))

    url = coord_API_endpoint + lat_lon + exclude + join_key + API_key + units

    forecast_json_data = requests.get(url).json()
    df_predictions = pd.DataFrame()

    # Creating empty lists
    days = []
    hours = []
    pressures = []
    temperatures = []
    precipitations = []
    relHumidities = []
    skyConditions = []
    visibilities = []
    windGusts = []
    winds = []

    # Loop Through the JSON
    for num_forecasts in forecast_json_data['hourly']: # Hourly forecast for next 48 hours
        days.append(datetime.datetime.fromtimestamp(num_forecasts['dt']).strftime('%Y-%m-%d'))
        hours.append(int(datetime.datetime.fromtimestamp(num_forecasts['dt']).strftime('%H')))
        pressures.append(round(num_forecasts['pressure'] * 0.029529983071445, 2)) # hPa to inHg
        temperatures.append(int(round(num_forecasts['temp'], 0))) # imperial: Fahrenheit
        try: # "Where available"
            precipitations.append(round(num_forecasts['rain']['1h'] * 0.03937007874015748, 2)) # mm to in
        except KeyError:
            precipitations.append(0)
        relHumidities.append(num_forecasts['humidity']) # Humidity, %
        skyConditions.append(num_forecasts['clouds']) # Cloudiness, %
        visibilities.append(int(round(num_forecasts['visibility'] * 0.0006213712, 0))) # m to mi
        try: # "Where available"
            windGusts.append(int(round(num_forecasts['wind_gust'] * 0.8689762, 0))) # mi/h to kt
        except KeyError:
            windGusts.append(0)
        winds.append(int(round(num_forecasts['wind_speed'] * 0.8689762, 0))) # mi/h to kt

    # Put data into a dataframe

    def skycond(x):
        if x == 0:
            return 'CLR'
        elif x < 2/8 * 100:
            return 'FEW'
        elif x < 4/8 * 100:
            return 'SCT'
        elif x < 7/8 * 100:
            return 'BKN'
        elif x <= 8/8 * 100:
            return 'OVC'

    skyConditions = list(map(skycond, skyConditions))

    df_predictions['day'] = days
    df_predictions['hour'] = [str(hour) for hour in hours]
    df_predictions['pressure_inHg'] = pressures
    df_predictions['temperature_F'] = temperatures
    df_predictions['precipitation_in'] = precipitations
    df_predictions['relHumidity_%'] = relHumidities
    df_predictions['skyCondition'] = skyConditions
    df_predictions['visibility_mi'] = visibilities
    df_predictions['windGust_kt'] = windGusts
    df_predictions['wind_kt'] = winds
    
    return df_predictions, url    
       
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------

# def user_inputs(df):
def user_inputs(root):
    """
    Define user input fields
    """
    
    # Create user input fields:

    
# FLIGHT DATA

    st.markdown('---')
    st.title('Flight data')
#     st.markdown('<p style="text-align: center;">Flight data</p>')
#     st.markdown(""" 
#         <div style ="background-color:#DFE9FB;padding:3px"> 
#         <h4 style ="color:black;text-align:center;">FLIGHT DATA</h4> 
#         </div> 
#         """  , unsafe_allow_html = True)
    
    
# 1) CARRIER:
    st.subheader('Carrier')
    
    # Carrier:
    with open(root + "dict_mappers/carriers_dict.pkl", "rb") as f:
        carriers_dict = pickle.load(f)
#     carrier = st.selectbox('Carrier', df['OP_UNIQUE_CARRIER'].value_counts().index, format_func = carriers_dict.get)
    with open(root + "dict_mappers/carriers_sorted_list.pkl", "rb") as f:
        carriers = pickle.load(f)
    carrier = st.selectbox('Carrier', carriers, format_func = carriers_dict.get)
    
# 2) ORIGIN:
    st.subheader('Origin')
    
    # Origin:
    originType = st.radio('Departure airport', options=['Currently operated by the A/L', 'All airports'], index=0, key=1)
    if originType == 'Currently operated by the A/L':
        with open(root + "dict_mappers/carrierOrigins_dict.pkl", "rb") as f:
            carrierOrigins_dict = pickle.load(f)
            origins = carrierOrigins_dict[carrier]
        origin = st.selectbox('Origin', origins)
    elif originType == 'All airports':
        with open(root + "dict_mappers/origins_sorted_list.pkl", "rb") as f:
            origins = pickle.load(f)
        origin = st.selectbox('Origin', origins)

    # Latitude / Longitude - ORIGIN:
    with open(root + "dict_mappers/latitude_dict.pkl", "rb") as f:
        latitude_dict = pickle.load(f)
    with open(root + "dict_mappers/longitude_dict.pkl", "rb") as f:
        longitude_dict = pickle.load(f)
    col1, col2, = st.beta_columns(2)
    with col1:
        st.text('ORIGIN - Latitude')
        latitudeOrigin = st.markdown(('{:8.5f}').format((latitude_dict[origin])))
        latitudeOrigin = latitude_dict[origin] # So that the model can use it in the proper format
    with col2:
        st.text('ORIGIN - Longitude')
        longitudeOrigin = st.markdown(('{:8.5f}').format((longitude_dict[origin])))
        longitudeOrigin = longitude_dict[origin] # So that the model can use it in the proper format
    # Taxi-out:
    with open(root + "dict_mappers/taxi_out_dict.pkl", "rb") as f:
        taxi_out_dict = pickle.load(f)          
    try:
        taxiout = st.slider('Taxi-out time [min] (*)',
                            min_value=0, max_value=40, value=taxi_out_dict[origin + '_' + carrier], step=1)
        st.markdown("(\*) *Default value is set to the median for the combination of: Origin and Carrier*")
    except KeyError:
        taxiout = st.slider('Taxi-out time [min] (*)', min_value=0, max_value=40, value=15, step=1)
        st.markdown("(\*) *The selected Carrier has not flown from Origin before. Please select a value*")
    
    
# 3) DESTINATION:
    st.subheader('Destination')
    
    # Destination:
    destType = st.radio('Departure airport', options=['Currently operated by the A/L', 'All airports'], index=0, key=2)
    if destType == 'Currently operated by the A/L':
        with open(root + "dict_mappers/carrierDests_dict.pkl", "rb") as f:
            carrierDests_dict = pickle.load(f)
            dests = carrierDests_dict[carrier]
        dest = st.selectbox('Dest', dests)
    elif destType == 'All airports':
        with open(root + "dict_mappers/dests_sorted_list.pkl", "rb") as f:
            dests = pickle.load(f)
        dest = st.selectbox('Destination', dests)

    # Latitude / Longitude - DESTINATION:
    col3, col4, = st.beta_columns(2)
    with col3:
        st.text('DESTINATION - Latitude')
        latitudeDest = st.markdown(('{:8.5f}').format((latitude_dict[dest])))
        latitudeDest = latitude_dict[dest] # So that the model can use it in the proper format
    with col4:
        st.text('DESTINATION - Longitude')
        longitudeDest = st.markdown(('{:8.5f}').format((longitude_dict[dest])))
        longitudeDest = longitude_dict[dest] # So that the model can use it in the proper format
    # Taxi-in:
    with open(root + "dict_mappers/taxi_in_dict.pkl", "rb") as f:
        taxi_in_dict = pickle.load(f)          
    try:
        taxiin = st.slider('Taxi-in time [min] (*)',
                            min_value=0, max_value=20, value=taxi_in_dict[dest + '_' + carrier], step=1)
        st.markdown("(\*) *Default value is set to the median for the combination of: Destination and Carrier*")
    except KeyError:
        taxiin = st.slider('Taxi-in time [min] (*)', min_value=0, max_value=20, value=6, step=1)
        st.markdown("(\*) *The selected Carrier has not flown to Destination before. Please select a value*")

        
# 4) TIME:
    st.subheader('Time')
    st.write("Current time:", datetime.datetime.now().strftime("%Y-%m-%d | %H:%M:%S"),
             "({})".format(datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo))
    
    col5, col6, col7 = st.beta_columns(3)    
    with col5:
        # Date:
        fdate = st.date_input("Flight date", value=datetime.date.today(),
                              min_value=datetime.date(2019, 1, 1), max_value=datetime.date(2021, 12, 31))
        fmonth = str(fdate.month)
        fweekday = str(fdate.isoweekday())    
    with col6:
        # Departure time:
#         deptime = st.selectbox('Departure time hour', list(map(str, sorted([int(hour) for hour in df['DEP_TIME_hour'].unique()]))))    
        with open(root + "dict_mappers/depTimeHours_list.pkl", "rb") as f:
            depTimeHours = pickle.load(f)
        deptime = st.selectbox('Departure time hour', list(map(str, sorted([int(hour) for hour in depTimeHours]))))    
    with col7:
        # Arrival time:
        with open(root + "dict_mappers/arrTimeHour_dict.pkl", "rb") as f:
            arrTimeHour_dict = pickle.load(f)
        with open(root + "dict_mappers/arrTimeHour_dict_2.pkl", "rb") as f:
            arrTimeHour_dict_2 = pickle.load(f)
        try:
            arrtime = st.number_input('Arrival  time hour (*)', min_value=0,
                                      value=int(arrTimeHour_dict[origin + '_' + dest + '_' + carrier + '_' + deptime]),
                                      max_value=23, step=1)
            arrtime = str(arrtime) # So that the model can use it in the proper format
            st.markdown("(\*) *Default value is set by the combination of: Origin, Destination, Carrier and Departure time hour*")
        except KeyError:
            try:
                arrtime = st.number_input('Arrival  time hour (*)', min_value=0,
                                          value=int(deptime) + int(arrTimeHour_dict_2[origin + '_' + dest]),
                                          max_value=23, step=1)
                arrtime = str(arrtime) # So that the model can use it in the proper format
                st.markdown("(\*) *The selected combination of Origin, Destination, Carrier and Departure time hour has not been flown before*")
                st.markdown("*Therefore, a default value has been set based on the combination of: Origin and Destination*")
            except KeyError:
                arrtime = st.number_input('Arrival  time hour (*)', min_value=0, value=0, max_value=23, step=1)
                arrtime = str(arrtime) # So that the model can use it in the proper format
                st.markdown("(\*) *The selected combination of Origin and Destination has not been flown before*")        

    # Distance:    
    with open(root + "dict_mappers/distance_dict.pkl", "rb") as f:
        distance_dict = pickle.load(f) 
    try:
        if distance_dict[origin + '_' + dest]:
            pass
    except KeyError:
        distance = st.slider('Distance covered [mi] (*)', min_value=0, max_value=6000, value=600, step=50)
        st.markdown("(\*) *The selected route (Origin-Destination) has not been flown before. Please select a value*")
    else: 
        st.markdown('Distance covered [mi]')
        distance = st.markdown(distance_dict[origin + '_' + dest])
        distance = distance_dict[origin + '_' + dest] # So that the model can use it in the proper format

# METEOROLOGICAL DATA

    st.markdown('---')
    st.title('Meteorological data')

    col8, col9, col10 = st.beta_columns([3, 1, 3])
    
    with col8:
    # 1) ORIGIN:
        st.subheader('Origin')     
        
       
        df_predictions, url = weather_forecast(latitudeOrigin, longitudeOrigin)
        flight_forecast = df_predictions[(df_predictions['day'] == str(fdate)) & (df_predictions['hour'] == deptime)]

        if len(flight_forecast) > 0:
            st.success("""*Weather forecast is available for the
                           departure. Therefore, meteorological
                           inputs have been defaulted accordingly.  
                           Powered by [OpenWeather]({})*""".format(url))
            altset_def = flight_forecast['pressure_inHg'].iloc[0]
            temp_def = int(flight_forecast['temperature_F'].iloc[0])
            precip_def = float(flight_forecast['precipitation_in'].iloc[0])
            relHumid_def = flight_forecast['relHumidity_%'].iloc[0]
            skyCond_def_dict = {'CLR': 0, 'FEW': 1, 'SCT': 2, 'BKN': 3, 'OVC': 4}
            skyCond_def = skyCond_def_dict[flight_forecast['skyCondition'].iloc[0]]
            visibility_def = int(flight_forecast['visibility_mi'].iloc[0])
            gust_def = int(flight_forecast['windGust_kt'].iloc[0])
            wind_def = int(flight_forecast['wind_kt'].iloc[0])
                
        else:
            st.warning("""*Unfortunately, no weather prediction is
                           available for the selected flight. Predictions
                           are only available for the next 48h.*""")
            altset_def = 29.92
            temp_def = 59
            precip_def = 0.
            relHumid_def = 60
            skyCond_def = 0
            visibility_def = 10
            gust_def = 0
            wind_def = 8
            
        # Altimeter setting - ORIGIN: 
        altsetOrigin = st.number_input('Altimeter setting [inHg]', min_value=27., value=altset_def,
                                       max_value=32., step=0.01, key=1)

        # Temperature - ORIGIN:
        tempTypeOrigin = st.radio('Temperature unit', options=['ºF', 'ºC'], index=0, key=1)
        if tempTypeOrigin == 'ºF':
            tempOrigin = st.slider('Temperature [ºF]', min_value=-50, max_value=130, value=temp_def, step=1, key=1)
        elif tempTypeOrigin == 'ºC':
            tempOrigin = st.slider('Temperature [ºC]', min_value=-50, max_value=50,
                                   value=int((temp_def - 32) / 1.8), step=1, key=1)
            tempOrigin = int(1.8 * tempOrigin + 32) # Convert Celsius to Fahrenheit to properly feed the model
            
        # Hourly precipitation - ORIGIN:
        precipOrigin = st.number_input('Hourly precipitation [in]', min_value=0.,
                                       value=precip_def, max_value=30., step=0.01, key=1)

        # Relative humidity - ORIGIN:    
        relhumOrigin = st.number_input('Relative humidity [%]', min_value=0,
                                       value=relHumid_def, max_value=100, step=1, key=1)

        # Sky condtions - ORIGIN: 
        with open(root + "dict_mappers/sky_dict.pkl", "rb") as f:
            sky_dict = pickle.load(f)
        skyOrigin = st.selectbox('Sky conditions', options=list(sky_dict.keys()),
                                 index=skyCond_def, format_func = sky_dict.get, key=1)

        # Visibility - ORIGIN:
        visibOrigin = st.number_input('Visibility [mi]', min_value=0,
                                       value=visibility_def, max_value=100, step=1, key=1)

        # Wind gust speed - ORIGIN:
        gustOrigin = st.slider('Wind gust speed [mph]', min_value=0, max_value=40, value=gust_def, step=1, key=1)

        # Wind speed - ORIGIN:
        windOrigin = st.slider('Wind speed [mph]', min_value=0, max_value=40, value=wind_def, step=1, key=1)

    with col10:
    # 2) DESTINATION:
        st.subheader('Destination')     

        df_predictions, url = weather_forecast(latitudeDest, longitudeDest)
        if int(arrtime) < int(deptime): # Late night flight arriving on the following day
            flight_forecast = df_predictions[(df_predictions['day'] == str(fdate + datetime.timedelta(days=1))) \
                                           & (df_predictions['hour'] == arrtime)]
        else:
            flight_forecast = df_predictions[(df_predictions['day'] == str(fdate)) & (df_predictions['hour'] == arrtime)]

        if len(flight_forecast) > 0:
            st.success("""*Weather forecast is available for the
                           arrival. Therefore, meteorological
                           inputs have been defaulted accordingly.  
                           Powered by [OpenWeather]({})*""".format(url))
            altset_def = flight_forecast['pressure_inHg'].iloc[0]
            temp_def = int(flight_forecast['temperature_F'].iloc[0])
            precip_def = float(flight_forecast['precipitation_in'].iloc[0])
            relHumid_def = flight_forecast['relHumidity_%'].iloc[0]
            skyCond_def_dict = {'CLR': 0, 'FEW': 1, 'SCT': 2, 'BKN': 3, 'OVC': 4}
            skyCond_def = skyCond_def_dict[flight_forecast['skyCondition'].iloc[0]]
            visibility_def = int(flight_forecast['visibility_mi'].iloc[0])
            gust_def = int(flight_forecast['windGust_kt'].iloc[0])
            wind_def = int(flight_forecast['wind_kt'].iloc[0])
                
        else:
            st.warning("""*Unfortunately, no weather prediction is
                           available for the selected flight. Predictions
                           are only available for the next 48h.*""")
            altset_def = 29.92
            temp_def = 59
            precip_def = 0.
            relHumid_def = 60
            skyCond_def = 0
            visibility_def = 10
            gust_def = 0
            wind_def = 8
            
        # Altimeter setting - DEST: 
        altsetDest = st.number_input('Altimeter setting [inHg]', min_value=27., value=altset_def,
                                       max_value=32., step=0.01, key=2)

        # Temperature - DEST:
        tempTypeDest = st.radio('Temperature unit', options=['ºF', 'ºC'], index=0, key=2)
        if tempTypeDest == 'ºF':
            tempDest = st.slider('Temperature [ºF]', min_value=-50, max_value=130, value=temp_def, step=1, key=2)
        elif tempTypeDest == 'ºC':
            tempDest = st.slider('Temperature [ºC]', min_value=-50, max_value=50,
                                   value=int((temp_def - 32) / 1.8), step=1, key=2)
            tempDest = int(1.8 * tempDest + 32) # Convert Celsius to Fahrenheit to properly feed the model
            
        # Hourly precipitation - DEST:
        precipDest = st.number_input('Hourly precipitation [in]', min_value=0.,
                                       value=precip_def, max_value=30., step=0.01, key=2)

        # Relative humidity - DEST:    
        relhumDest = st.number_input('Relative humidity [%]', min_value=0,
                                       value=relHumid_def, max_value=100, step=1, key=2)

        # Sky condtions - DEST: 
        with open(root + "dict_mappers/sky_dict.pkl", "rb") as f:
            sky_dict = pickle.load(f)
        skyDest = st.selectbox('Sky conditions', options=list(sky_dict.keys()),
                                 index=skyCond_def, format_func = sky_dict.get, key=2)

        # Visibility - DEST:
        visibDest = st.number_input('Visibility [mi]', min_value=0,
                                       value=visibility_def, max_value=100, step=1, key=2)

        # Wind gust speed - DEST:
        gustDest = st.slider('Wind gust speed [mph]', min_value=0, max_value=40, value=gust_def, step=1, key=2)

        # Wind speed - DEST:
        windDest = st.slider('Wind speed [mph]', min_value=0, max_value=40, value=wind_def, step=1, key=2)
    

    user_inputs = [
        fmonth, fweekday, carrier, origin, dest, deptime, taxiout, taxiin, arrtime, distance,                
        altsetOrigin, tempOrigin, precipOrigin,
        relhumOrigin, skyOrigin, visibOrigin, gustOrigin, windOrigin,
        altsetDest, tempDest, precipDest,
        relhumDest, skyDest, visibDest, gustDest, windDest
    ]

    
    return user_inputs

# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------

def prediction(model, X_test):  
    # Making predictions: 
    prediction = model.predict(X_test)
    score = model.predict_proba(X_test)[0, 0]
    if prediction == 0:
        result = 'ON-TIME'
    else:
        result = 'DELAYED'
    return result, score

# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------

@st.cache(hash_funcs={shap.explainers._tree.Tree: hash})
def load_shap_explainer(filename='', X_test_transformed=pd.DataFrame()):   
    # Load the explainer file (instead of generating it so as to save time):
    explainer = joblib.load(filename)
    shap_values = explainer.shap_values(X_test_transformed)
    return explainer, shap_values

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height) 

# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------


if __name__=='__main__': 
    
#     root = "/app/tfm_kschool/frontend/" # Used for deployment on Streamlit Sharing platform
    root = "" # Used for running local tests
    
    # Let the user know the data is loading and load the data:
#     df, X, y = load_data()

    # Load the model:
#     pipe = load_model(path="XGBoost_pipeline_model.joblib.dat")
    pipe = load_model(path=root + "XGBoost_pipeline_model.joblib.dat")
    transformer = pipe[:-1]
    model = pipe.named_steps['clf']
    
    # Load the general HMI framework:
    col1, col2, col3 = st.beta_columns([0.1,1.25,0.1])
    with col1:
        st.write("")
    with col2:
        st.image(root + 'logo3.jpeg')
    with col3:
        st.write("")
        
#     frontend_appearance()
    
    # Load the input fields:
#     inputs = user_inputs(df)
    inputs = user_inputs(root)
    
    # Generate an array based on user input values thal will be fed into the model:
    dismissed_cols = [
                      'ARR_DEL15',
                      'LATITUDE_Origin',
                      'LONGITUDE_Origin',
                      'LATITUDE_Dest',
                      'LONGITUDE_Dest'
                    ]
    X_test = pd.DataFrame(
            data=np.array(inputs)[np.newaxis], # Kind of transpose the resulting array from the 'inputs' list
            columns=[col for col in cols if col not in dismissed_cols]
        )
    cols_dtypes_frontend = cols_dtypes.copy()
    for col in dismissed_cols:
        del cols_dtypes_frontend[col]
    X_test = X_test.astype(cols_dtypes_frontend)

    # Indicate numerical and categorical features:
    num_attribs = X_test.select_dtypes('number').columns
    cat_attribs = X_test.select_dtypes(['string', 'category', 'object']).columns
    # Transform categorical variables:
    X_test_categTransformed_df = pd.DataFrame(transformer.transform(X_test)[:, 0:9], columns=cat_attribs)
    # Concatenate categorical transformed features with 'as-is' numerical features:
    X_test_transformed = pd.concat([X_test_categTransformed_df, X_test[num_attribs]], axis=1)
    X_test_transformed = X_test_transformed[X_test.columns]
       
    # When 'Predict' button is clicked, make the prediction and store it: 
    st.markdown('---')
    col11, col12, col13 = st.beta_columns([3, 1, 3])
    if col12.button("Predict"):
        # Calculate prediction:
        result = prediction(pipe, X_test)[0]
        score = prediction(pipe, X_test)[1]
        if result == 'ON-TIME':
            st.success('The flight is predicted to be **{}** ({:5.2f}%)'.format(result, 100*score))
        elif result == 'DELAYED':
            st.error('The flight is predicted to be **{}** ({:5.2f}%)'.format(result, 100*(1-score)))
        
        # SHAP values and force plot:
        with st.beta_expander("See explanatory details for this flight prediction"):
            st.write("""
                Below a *SHAP force plot* explains the contribution of each variable to the model's prediction.  
                - Red features are *forcing* the prediction to **DELAY**.
                - On the contrary, blue variables drive the prediction to **ON-TIME**.
            """)
            explainer, shap_values = load_shap_explainer(root + 'shap_treeExplainer.bz2', X_test_transformed)
            st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:], link='logit'))
            
            # -----------

            shap.decision_plot(base_value=explainer.expected_value, shap_values=shap_values[0],
                              features=X_test.iloc[0,:], link='logit', feature_display_range=slice(None, -X_test.shape[1]-1, -1),
                              return_objects=True, show=False, y_demarc_color='#00172b')
            fig = plt.gcf()
            ax = plt.gca()
            fig.patch.set_facecolor('#00172b')
            ax.set_facecolor('#00172b')
            ax.set_xlabel('Probability', fontsize=16, color='white')
            ax.tick_params(axis='both', colors='white')
            ax.grid(axis='both', color='white', linestyle='-', linewidth=0.25)
            for ln in ax.lines:
                ln.set_linewidth(3)
            for text in ax.texts:
                text.set_color('white')
                text.set_alpha(0.75)
            st.pyplot(fig)
            
            # -----------
    
#     # SHAP values general overview:
#     shapSummary = st.checkbox(label="SHAP Summary Plot")
#     if shapSummary:
#         st.image(root + 'shap_summaryPlot.png')

       
