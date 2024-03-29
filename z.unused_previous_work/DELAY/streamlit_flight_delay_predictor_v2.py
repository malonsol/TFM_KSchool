
import streamlit as st
import streamlit.components.v1 as components
import os
import joblib
from joblib import load
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

if os.name == 'nt': # Windows
    root = r"C:\Users\turge\CompartidoVM\0.TFM"
    print("Running on Windows.")
elif os.name == 'posix': # Ubuntu
    root = "/home/dsc/shared/0.TFM"
    print("Running on Ubuntu.")
print("root path\t", root)

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
    'ARR_DEL15', # Not in model
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
    'ARR_DEL15' : 'int32',
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

# Load the data:
@st.cache
def load_data():
    preprocessed_input_csv_path = os.path.join(root,
                                               "Output_Data",
                                               "US_DoT-NOAA",
                                               "OTP_LCD_allColumns_v2.csv")
    df = pd.read_csv(preprocessed_input_csv_path,
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
    model = load(path)
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

def user_inputs(df):
    """
    Define user input fields
    """

    # Load the target-encoding mapper dictionary:
    te_map_file = open("te_map_file.pkl", "rb")
    te_map_dict = pickle.load(te_map_file)
    
#     add_selectbox = st.sidebar.selectbox("How would you like to be contacted?", ("Email", "Home phone", "Mobile phone"))
    
    # Create user input fields:
    # Categorical:
    # Date:
    fdate = st.date_input("Flight date", value=datetime.date(2019, 7, 6),
                          min_value=datetime.date(2019, 1, 1), max_value=datetime.date(2019, 12, 31))
    fmonth = str(fdate.month)
    fweekday = str(fdate.isoweekday())
    # Carrier:
    carriers_dict = {
                        '9E' : '[9E] Endeavor Air Inc.',
                        'AA' : '[AA] American Airlines Inc.',
                        'AS' : '[AS] Alaska Airlines Inc.',
                        'B6' : '[B6] JetBlue Airways',
                        'DL' : '[DL] Delta Air Lines Inc.',
                        'EV' : '[EV] ExpressJet Airlines LLC',
                        'F9' : '[F9] Frontier Airlines Inc.',
                        'G4' : '[G4] Allegiant Air',
                        'HA' : '[HA] Hawaiian Airlines Inc.',
                        'MQ' : '[MQ] Envoy Air',
                        'NK' : '[NK] Spirit Air Lines',
                        'OH' : '[OH] PSA Airlines Inc.',
                        'OO' : '[OO] SkyWest Airlines Inc.',
                        'UA' : '[UA] United Air Lines Inc.',
                        'WN' : '[WN] Southwest Airlines Co.',
                        'YV' : '[YV] Mesa Airlines Inc.',
                        'YX' : '[YX] Republic Airline'
                   }
#     carrier = st.selectbox('Airline', sorted(list(te_map_dict['OP_UNIQUE_CARRIER'])), format_func = carriers_dict.get)
    carrier = st.selectbox('Airline', df['OP_UNIQUE_CARRIER'].value_counts().index, format_func = carriers_dict.get)
    
    # Origin:
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
    altsetorigin = st.slider('ORIGIN - Altimeter Setting [inHg]', min_value=27., max_value=32., value=30., step=0.25)
    temporigin = st.slider('ORIGIN - Temperature [ºF]', min_value=-50, max_value=130, value=65, step=5)
    preciporigin = st.number_input('HourlyPrecipitation_Origin')
    relhumorigin = st.number_input('HourlyRelativeHumidity_Origin')
    visiborigin = st.number_input('HourlyVisibility_Origin')
    gustorigin = st.number_input('HourlyWindGustSpeed_Origin')
    windorigin = st.number_input('HourlyWindSpeed_Origin')
    altsetdest = st.slider('DESTINATION - Altimeter Setting [inHg]', min_value=27., max_value=32., value=30., step=0.25)
    tempdest = st.slider('DESTINATION - Temperature [ºF]', min_value=-50, max_value=130, value=65, step=5)
    precipdest = st.number_input('HourlyPrecipitation_Dest')
    relhumdest = st.number_input('HourlyRelativeHumidity_Dest')
    visibdest = st.number_input('HourlyVisibility_Dest')
    gustdest = st.number_input('HourlyWindGustSpeed_Dest')
    winddest = st.number_input('HourlyWindSpeed_Dest')

    user_inputs = [fmonth, fweekday, carrier, origin, dest, deptime,
                   int(float(taxiout)), int(float(taxiin)), arrtime, distgroup, 
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

# @st.cache
# def SHAP_individual_graphs(prepared_data):   
#     # Load the explainer file (instead of generating it so as to save time):
#     explainer = joblib.load(filename='SHAP_explainer.bz2')
#     # Compute SHAP values for this particular flight:
#     shap_values = explainer(prepared_data)
    
#     # Visualize prediction's explanation:
#     shap.plots.waterfall(shap_values=shap_values[0], max_display=prepared_data.shape[1], show=True)
#     plt.show()

#     # load JS visualization code to notebook:
#     shap.initjs()
#     # visualize prediction's explanation with a force plot:
#     shap.plots.force(base_value=explainer.expected_value, shap_values=shap_values[0], matplotlib=True)

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------

def shap_images(path=""):
    st.image(image=path, output_format='PNG')

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

if __name__=='__main__': 
    # Let the user know the data is loading:
    data_load_state = st.text('Loading data...')
    # Load the data:
    df, X, y = load_data()
    # Notify the user that the data was successfully loaded:
    if st.checkbox(label="SHAP Summary Plot", value=False) == False:
        data_load_state.title('') 
    else:
        data_load_state.title('SHAP Summary Plot')
        st.write(':sunglasses:')
        shap_images("SHAP_values_global.png")
    model = load_model(path="XGBoost_32_best_dask_rscv.joblib.dat")
    frontend_appearance()
    inputs = user_inputs(df)
#     upload_file()
    dismissed_cols = ['ARR_DEL15']
    X_test = pd.DataFrame(
            data=np.array(inputs)[np.newaxis], # Kind of transpose the resulting array from the 'inputs' list
            columns=[col for col in cols if col not in dismissed_cols]
        )
    cols_dtypes_frontend = cols_dtypes.copy()
    for col in dismissed_cols:
        del cols_dtypes_frontend[col]
    X_test = X_test.astype(cols_dtypes_frontend)
    prepared_data = target_encoding(X_test)
    
    # When 'Predict' is clicked, make the prediction and store it: 
    if st.button("Predict"):
        result = prediction(model, prepared_data)[0]
        score = prediction(model, prepared_data)[1]
        if result == 'ON-TIME':
            st.success('The flight is likely to be {} ({:5.2f}%)'.format(result, 100*score))
        elif result == 'DELAYED':
            st.warning('The flight is likely to be {} ({:5.2f}%)'.format(result, 100*(1-score)))
#         SHAP_individual_graphs(prepared_data)
#         explainer = shap.TreeExplainer(model)
        explainer = joblib.load(filename='SHAP_explainer.bz2')
        shap_values = explainer.shap_values(prepared_data)
        st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], prepared_data.iloc[0,:]))
