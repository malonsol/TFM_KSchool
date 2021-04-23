
import streamlit as st
import streamlit.components.v1 as components
import os
import joblib
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot') # choose a style: 'plt.style.available'
sns.set_theme(context='notebook',
              style="darkgrid") # {darkgrid, whitegrid, dark, white, ticks}
palette = sns.color_palette("flare", as_cmap=True);
import altair as alt
import shap
import datetime

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

# Load the data:
@st.cache
def load_data():
    input_folder = '../../data/output/us_dot-noaa/'
    file_name = "3_otp_lcd_2019.csv"

    df = pd.read_csv(input_folder + file_name,
                     encoding='latin1',
    #                      nrows=1e5,
                     usecols=cols,
                     dtype=cols_dtypes
                    )
    X = df.drop(['ARR_DEL15'], axis=1)
    y = df['ARR_DEL15']
    return df, X, y

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
    <div style ="background-color:powderblue;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Flight Delay Prediction ML App</h1> 
    </div> 
    """   
    # display the frontend aspect
    st.markdown(html_temp, unsafe_allow_html = True)

# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------  

# def user_inputs(df):
def user_inputs():
    """
    Define user input fields
    """
    
    # Create user input fields:

    
# FLIGHT DATA    
    st.markdown('---')
    st.title('Flight data')
    
    
# 1) CARRIER:
    st.subheader('Carrier')
    
    # Carrier:
    with open("dict_mappers/carriers_dict.pkl", "rb") as f:
        carriers_dict = pickle.load(f)
#     carrier = st.selectbox('Carrier', df['OP_UNIQUE_CARRIER'].value_counts().index, format_func = carriers_dict.get)
    with open("dict_mappers/carriers_sorted_list.pkl", "rb") as f:
        carriers = pickle.load(f)
    carrier = st.selectbox('Carrier', carriers, format_func = carriers_dict.get)
    
# 2) ORIGIN:
    st.subheader('Origin')
    
    # Origin:
#     origin = st.selectbox('Origin', sorted(df['ORIGIN'].unique()))
    with open("dict_mappers/origins_sorted_list.pkl", "rb") as f:
        origins = pickle.load(f)
    origin = st.selectbox('Origin', origins)

    # Latitude / Longitude - ORIGIN:
    with open("dict_mappers/latitude_dict.pkl", "rb") as f:
        latitude_dict = pickle.load(f)
    with open("dict_mappers/longitude_dict.pkl", "rb") as f:
        longitude_dict = pickle.load(f)
    col1, col2, = st.beta_columns(2)
    with col1:
        st.text('ORIGIN - Latitude')
        latitudeOrigin = st.code(('{:8.5f}').format((latitude_dict[origin])), language='python')
        latitudeOrigin = latitude_dict[origin] # So that the model can use it in the proper format
    with col2:
        st.text('ORIGIN - Longitude')
        longitudeOrigin = st.code(('{:8.5f}').format((longitude_dict[origin])), language='python')
        longitudeOrigin = longitude_dict[origin] # So that the model can use it in the proper format
    # Taxi-out:
    with open("dict_mappers/taxi_out_dict.pkl", "rb") as f:
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
#     dest = st.selectbox('Destination', sorted(df['DEST'].unique()))
    with open("dict_mappers/dests_sorted_list.pkl", "rb") as f:
        dests = pickle.load(f)
    dest = st.selectbox('Destination', dests)

    # Latitude / Longitude - DESTINATION:
    col3, col4, = st.beta_columns(2)
    with col3:
        st.text('DESTINATION - Latitude')
        latitudeDest = st.code(('{:8.5f}').format((latitude_dict[dest])), language='python')
        latitudeDest = latitude_dict[dest] # So that the model can use it in the proper format
    with col4:
        st.text('DESTINATION - Longitude')
        longitudeDest = st.code(('{:8.5f}').format((longitude_dict[dest])), language='python')
        longitudeDest = longitude_dict[dest] # So that the model can use it in the proper format
    # Taxi-in:
    with open("dict_mappers/taxi_in_dict.pkl", "rb") as f:
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

    col5, col6, col7 = st.beta_columns(3)    
    with col5:
        # Date:
        fdate = st.date_input("Flight date", value=datetime.date(2019, 7, 6),
                              min_value=datetime.date(2019, 1, 1), max_value=datetime.date(2019, 12, 31))
        fmonth = str(fdate.month)
        fweekday = str(fdate.isoweekday())    
    with col6:
        # Departure time:
#         deptime = st.selectbox('Departure time hour', list(map(str, sorted([int(hour) for hour in df['DEP_TIME_hour'].unique()]))))    
        with open("dict_mappers/depTimeHours_list.pkl", "rb") as f:
            depTimeHours = pickle.load(f)
        deptime = st.selectbox('Departure time hour', list(map(str, sorted([int(hour) for hour in depTimeHours]))))    
    with col7:
        # Arrival time:
        with open("dict_mappers/arrTimeHour_dict.pkl", "rb") as f:
            arrTimeHour_dict = pickle.load(f)
        with open("dict_mappers/arrTimeHour_dict_2.pkl", "rb") as f:
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
    with open("dict_mappers/distance_dict.pkl", "rb") as f:
        distance_dict = pickle.load(f) 
    try:
        st.text('Distance covered [mi] (fixed value based on Origin-Destination values)')
        distance = st.code(distance_dict[origin + '_' + dest], language='python')
        distance = distance_dict[origin + '_' + dest] # So that the model can use it in the proper format
    except KeyError:
        distance = st.slider('Distance covered [mi] (*)', min_value=0, max_value=6000, value=600, step=50)
        st.markdown("(\*) *The selected route (Origin-Destination) has not been flown before. Please select a value*")      


# METEOROLOGICAL DATA
    st.markdown('---')
    st.title('Meteorological data')

    col8, col9, col10 = st.beta_columns([3, 1, 3])
    
    with col8:
    # 1) ORIGIN:
        st.subheader('Origin')     

        # Altimeter setting - ORIGIN:
        altsetOrigin = st.slider('Altimeter setting [inHg]', min_value=27., max_value=32., value=30., step=0.25, key=1)

        # Temperature - ORIGIN:
        tempTypeOrigin = st.radio('Temperature unit', options=['ºF', 'ºC'], index=0, key=1)
        if tempTypeOrigin == 'ºF':
            tempOrigin = st.slider('Temperature [ºF]', min_value=-50, max_value=130, value=65, step=5, key=1)
        elif tempTypeOrigin == 'ºC':
            tempOrigin = st.slider('Temperature [ºC]', min_value=-50, max_value=50, value=20, step=5, key=1)
            tempOrigin = int(1.8 * tempOrigin + 32) # Convert Celsius to Fahrenheit to properly feed the model
            
        # Hourly precipitation - ORIGIN:
        precipOrigin = st.number_input('Hourly precipitation [in]', min_value=0.,
                                       value=0., max_value=30., step=0.5, key=1)

        # Relative humidity - ORIGIN:    
        relhumOrigin = st.number_input('Relative humidity [%]', min_value=0,
                                       value=60, max_value=100, step=5, key=1)

        # Sky condtions - ORIGIN: 
        with open("dict_mappers/sky_dict.pkl", "rb") as f:
            sky_dict = pickle.load(f)
        skyOrigin = st.selectbox('Sky conditions', options=list(sky_dict.keys()),
                                 index=0, format_func = sky_dict.get, key=1)

        # Visibility - ORIGIN:
        visibOrigin = st.number_input('Visibility [mi]', min_value=0,
                                       value=10, max_value=100, step=1, key=1)

        # Wind gust speed - ORIGIN:
        gustOrigin = st.slider('Wind gust speed [mph]', min_value=0, max_value=40, value=0, step=1, key=1)

        # Wind speed - ORIGIN:
        windOrigin = st.slider('Wind speed [mph]', min_value=0, max_value=40, value=8, step=1, key=1)

    with col10:
    # 2) DESTINATION:
        st.subheader('Destination')     

        #  - DESTINATION:
        altsetDest = st.slider('Altimeter setting [inHg]', min_value=27., max_value=32., value=30., step=0.25, key=2)

        # Temperature - DESTINATION:
        tempTypeDest = st.radio('Temperature unit', options=['ºF', 'ºC'], index=0, key=2)
        if tempTypeDest == 'ºF':
            tempDest = st.slider('Temperature [ºF]', min_value=-50, max_value=130, value=65, step=5, key=2)
        elif tempTypeDest == 'ºC':
            tempDest = st.slider('Temperature [ºC]', min_value=-50, max_value=50, value=20, step=5, key=2)
            tempDest = int(1.8 * tempDest + 32) # Convert Celsius to Fahrenheit to properly feed the model

        # Hourly precipitation - DESTINATION:
        precipDest = st.number_input('Hourly precipitation [in]', min_value=0.,
                                     value=0., max_value=30., step=0.5, key=2)

        # Relative humidity - DESTINATION:    
        relhumDest = st.number_input('Relative humidity [%]', min_value=0,
                                     value=60, max_value=100, step=5, key=2)

        # Sky condtions - DESTINATION: 
        skyDest = st.selectbox('Sky conditions', options=list(sky_dict.keys()),
                               index=0, format_func = sky_dict.get, key=2)

        # Visibility - DESTINATION:
        visibDest = st.number_input('Visibility [mi]', min_value=0,
                                       value=10, max_value=100, step=1, key=2)

        # Wind gust speed - DESTINATION:
        gustDest = st.slider('Wind gust speed [mph]', min_value=0, max_value=40, value=0, step=1, key=2)

        # Wind speed - DESTINATION:
        windDest = st.slider('Wind speed [mph]', min_value=0, max_value=40, value=8, step=1, key=2)
    

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

def upload_file():
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)    

# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------

def clf_metrics(classifier, X_test, y_test, y_pred):
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("Normalized confusion matrix:\n", confusion_matrix(y_test, y_pred, normalize='true'), '\n')
    print(classification_report(y_test, y_pred, target_names=['on-time', 'delayed']))
    print("F-beta (ß=2) = {:6.3f}".format(fbeta_score(y_test, y_pred, beta=2)))   
    print("F1 =           {:6.3f}".format(f1_score(y_test, y_pred)))   
    print("Recall =       {:6.3f}".format(recall_score(y_test, y_pred)))   
    print("Precision =    {:6.3f}".format(precision_score(y_test, y_pred)))   
    print("Accuracy =     {:6.3f}".format(accuracy_score(y_test, y_pred)))    

# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------

if __name__=='__main__': 
    
    # Let the user know the data is loading and load the data:
#     df, X, y = load_data()

    st.write('HOLA')
    st.write(os.getcwd())
#     # Load the model:
# #     pipe = load_model(path="XGBoost_pipeline_model.joblib.dat")
#     pipe = load_model(path="https://github.com/malonsol/TFM_KSchool/blob/ec891057d90c5a3dfd36af321855ee12f2ef3cac/frontend/XGBoost_pipeline_model.joblib.dat")
#     transformer = pipe[:-1]
#     model = pipe.named_steps['clf']
    
#     # Load the general HMI framework:
#     frontend_appearance()
    
#     # Load the input fields:
# #     inputs = user_inputs(df)
#     inputs = user_inputs()
    
#     # Generate an array based on user input values thal will be fed into the model:
#     dismissed_cols = [
#                       'ARR_DEL15',
#                       'LATITUDE_Origin',
#                       'LONGITUDE_Origin',
#                       'LATITUDE_Dest',
#                       'LONGITUDE_Dest'
#                     ]
#     X_test = pd.DataFrame(
#             data=np.array(inputs)[np.newaxis], # Kind of transpose the resulting array from the 'inputs' list
#             columns=[col for col in cols if col not in dismissed_cols]
#         )
#     cols_dtypes_frontend = cols_dtypes.copy()
#     for col in dismissed_cols:
#         del cols_dtypes_frontend[col]
#     X_test = X_test.astype(cols_dtypes_frontend)

#     # Indicate numerical and categorical features:
#     num_attribs = X_test.select_dtypes('number').columns
#     cat_attribs = X_test.select_dtypes(['string', 'category', 'object']).columns
#     # Transform categorical variables:
#     X_test_categTransformed_df = pd.DataFrame(transformer.transform(X_test)[:, 0:9], columns=cat_attribs)
#     # Concatenate categorical transformed features with 'as-is' numerical features:
#     X_test_transformed = pd.concat([X_test_categTransformed_df, X_test[num_attribs]], axis=1)
#     X_test_transformed = X_test_transformed[X_test.columns]
       
#     # When 'Predict' button is clicked, make the prediction and store it: 
#     st.markdown('---')
#     col11, col12, col13 = st.beta_columns([3, 1, 3])
#     if col12.button("Predict"):
#         # Calculate prediction:
#         result = prediction(pipe, X_test)[0]
#         score = prediction(pipe, X_test)[1]
#         if result == 'ON-TIME':
#             st.success('The flight is predicted to be **{}** ({:5.2f}%)'.format(result, 100*score))
#         elif result == 'DELAYED':
#             st.error('The flight is predicted to be **{}** ({:5.2f}%)'.format(result, 100*(1-score)))
        
#         # SHAP values and force plot:
#         with st.beta_expander("See explanatory details for this flight prediction"):
#             st.write("""
#                 Below a *SHAP force plot* explains the contribution of each variable to the model's prediction.  
#                 - Red features are *forcing* the prediction to **DELAY**.
#                 - On the contrary, blue variables drive the prediction to **ON-TIME**.
#             """)
#             explainer, shap_values = load_shap_explainer('shap_treeExplainer.bz2', X_test_transformed)
#             st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:]))
            
        
#     # SHAP values general overview:
#     shapSummary = st.checkbox(label="SHAP Summary Plot")
#     if shapSummary:
#         st.image('shap_summaryPlot.png')
                    
#    # END:
#     st.markdown('---')
    
