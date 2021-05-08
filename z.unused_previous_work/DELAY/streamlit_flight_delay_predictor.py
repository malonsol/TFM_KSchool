
import streamlit as st
from joblib import load
import pickle
import numpy as np
import pandas as pd

# Load the model:
model = load("XGBoost_30_best_dask_rscv.joblib.dat")

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
    'ARR_TIME_hour',
    'DISTANCE_GROUP',
    'HourlyAltimeterSetting_Origin',
    'HourlyDryBulbTemperature_Origin',
    'HourlyPrecipitation_Origin',
    'HourlyRelativeHumidity_Origin',
    'HourlySkyConditions_Origin',
    'HourlyVisibility_Origin',
    'HourlyWindGustSpeed_Origin',
    'HourlyWindSpeed_Origin',
    'HourlyAltimeterSetting_Dest',
    'HourlyDryBulbTemperature_Dest',
    'HourlyPrecipitation_Dest',
    'HourlyRelativeHumidity_Dest',
    'HourlySkyConditions_Dest',
    'HourlyVisibility_Dest',
    'HourlyWindGustSpeed_Dest',
    'HourlyWindSpeed_Dest'
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
    'ARR_TIME_hour' : 'string',
    'DISTANCE_GROUP' : 'string',
    'HourlyAltimeterSetting_Origin' : 'float64',
    'HourlyDryBulbTemperature_Origin' : 'float64',
    'HourlyPrecipitation_Origin' : 'float64',
    'HourlyRelativeHumidity_Origin' : 'float64',
    'HourlySkyConditions_Origin' : 'string',
    'HourlyVisibility_Origin' : 'float64',
    'HourlyWindGustSpeed_Origin' : 'float64',
    'HourlyWindSpeed_Origin' : 'float64',
    'HourlyAltimeterSetting_Dest' : 'float64',
    'HourlyDryBulbTemperature_Dest' : 'float64',
    'HourlyPrecipitation_Dest' : 'float64',
    'HourlyRelativeHumidity_Dest' : 'float64',
    'HourlySkyConditions_Dest' : 'string',
    'HourlyVisibility_Dest' : 'float64',
    'HourlyWindGustSpeed_Dest' : 'float64',
    'HourlyWindSpeed_Dest' : 'float64',
}

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
    
def user_inputs():
    """
    Define user input fields
    """

    # Load the target-encoding mapper dictionary:
    te_map_file = open("te_map_file.pkl", "rb")
    te_map_dict = pickle.load(te_map_file)
   
    # Create user input fields:
    # Categorical:
    month = st.selectbox('Month', sorted(list(te_map_dict['MONTH'])))
    weekday = st.selectbox('Weekday', sorted(list(te_map_dict['DAY_OF_WEEK'])))
    carrier = st.selectbox('A/L', sorted(list(te_map_dict['OP_UNIQUE_CARRIER'])))
    origin = st.selectbox('Origin', sorted(list(te_map_dict['ORIGIN'])))
    dest = st.selectbox('Destination', sorted(list(te_map_dict['DEST'])))
    deptime = st.selectbox('Departure time', sorted([int(hour) for hour in list((te_map_dict['DEP_TIME_hour']))]))
    arrtime = st.selectbox('Arrival time', sorted([int(hour) for hour in list((te_map_dict['ARR_TIME_hour']))]))
    distgroup = st.selectbox('Distance group', sorted([int(group) for group in list((te_map_dict['DISTANCE_GROUP']))]))
    skyorigin = st.selectbox('Sky conditions (Origin)', sorted(list(te_map_dict['HourlySkyConditions_Origin'])))
    skydest = st.selectbox('Sky conditions (Dest)', sorted(list(te_map_dict['HourlySkyConditions_Dest'])))
    
    # Numerical:
    taxiout = st.number_input('TAXI_OUT_median')
    taxiin = st.number_input('TAXI_IN_median')
    altsetorigin = st.number_input('HourlyAltimeterSetting_Origin')
    temporigin = st.number_input('HourlyDryBulbTemperature_Origin')
    preciporigin = st.number_input('HourlyPrecipitation_Origin')
    relhumorigin = st.number_input('HourlyRelativeHumidity_Origin')
    visiborigin = st.number_input('HourlyVisibility_Origin')
    gustorigin = st.number_input('HourlyWindGustSpeed_Origin')
    windorigin = st.number_input('HourlyWindSpeed_Origin')
    altsetdest = st.number_input('HourlyAltimeterSetting_Dest')
    tempdest = st.number_input('HourlyDryBulbTemperature_Dest')
    precipdest = st.number_input('HourlyPrecipitation_Dest')
    relhumdest = st.number_input('HourlyRelativeHumidity_Dest')
    visibdest = st.number_input('HourlyVisibility_Dest')
    gustdest = st.number_input('HourlyWindGustSpeed_Dest')
    winddest = st.number_input('HourlyWindSpeed_Dest')

    user_inputs = [month, weekday, carrier, origin, dest, deptime, int(float(taxiout)), int(float(taxiin)), arrtime, distgroup, 
                   altsetorigin, temporigin, preciporigin, relhumorigin, skyorigin, visiborigin, gustorigin, windorigin,
                   altsetdest, tempdest, precipdest, relhumdest, skydest, visibdest, gustdest, winddest]
    
    return user_inputs

# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------

def te_mapper(te_dict, x):
    """
    Mapper function to apply Training's target encoded values to user categorical inputs
    """
    
    try:
        te_mapper = te_dict[x] # If category appeared in Training dataset, apply the corresponding value
    except KeyError:
        te_mapper = np.median(list(te_dict.values())) # Otherwise, apply the median from the entire Training dataset
    return te_mapper

# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------

@st.cache()
def target_encoding(X_test):
    """
    Take the user input variables and apply the 'te_mapper' encoding to prepare data for model feeding
    """
    # Load the target-encoding mapper dictionary:
    te_map_file = open("te_map_file.pkl", "rb")
    te_map_dict = pickle.load(te_map_file)

    # Declare which features are going to be target-encoded:
    te_features = X_test.select_dtypes(['string', 'category']).columns.to_list()
    
    # Map the values using the 'te_mapper' function:
    for cat_col in te_features:
        X_test[cat_col + '_te'] = X_test[cat_col].apply(lambda x: te_mapper(te_map_dict[cat_col], x))
    # Drop the original features to harmonize format:
    X_test.drop(te_features, axis=1, inplace=True)
    
    return X_test        

# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------

def prediction(model, prepared_data):  
    # Making predictions: 
    prediction = model.predict(prepared_data)
    score = model.predict_proba(prepared_data)[0, 0]
    if prediction == 0:
        result = 'ON-TIME'
    else:
        result = 'DELAYED'
    return result, score

# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------

if __name__=='__main__': 
    frontend_appearance()
    inputs = user_inputs()
    X_test = pd.DataFrame(
            data=np.array(inputs)[np.newaxis], # Kind of transpose the resulting array from the 'inputs' list
            columns=cols
        )
    X_test = X_test.astype(cols_dtypes)
    prepared_data = target_encoding(X_test)
    
    # When 'Predict' is clicked, make the prediction and store it: 
    if st.button("Predict"):
        result = prediction(model, prepared_data)[0]
        score = prediction(model, prepared_data)[1]
        if result == 'ON-TIME':
            st.success('The flight is likely to be {} ({:5.2f}%)'.format(result, 100*score))
        elif result == 'DELAYED':
            st.warning('The flight is likely to be {} ({:5.2f}%)'.format(result, 100*(1-score)))
            
