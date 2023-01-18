import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from PIL import Image

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

image = Image.open('cat2.png')
new_image = image.resize((100, 100))
st.image(new_image)

st.write("""
# Slammer Prediction App
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")
# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        Number_of_announcements = st.sidebar.text_input('Number_of_announcements >>> Input an integer value. Eg: "85","200"', 0)
        Number_of_announced_NLRI_prefixes = st.sidebar.text_input('Number_of_announced_NLRI_prefixes >>> Input an integer value. Eg: "164","595"',0)
        Maximum_AS_path_length = st.sidebar.text_input('Maximum_AS_path_length >>> Input an integer value. Eg: "14","18"', 0)
        Average_edit_distance = st.sidebar.text_input('Average_edit_distance >>> Input an integer value. Eg: "14","18"', 0)
        Maximum_edit_distance = st.sidebar.text_input('Maximum_edit_distance >>> Input a value. Eg: "1.5","3.4"', 0)
        Number_of_Interior_Gateway_Protocol_IGP_packets = st.sidebar.text_input('Number_of_Interior_Gateway_Protocol_IGP_packets >>> Input an integer value. Eg: "79","188"', 0)
        Number_of_incomplete_packets = st.sidebar.text_input('Number_of_incomplete_packets >>> Input an integer value. Eg: "6","12"', 0)
       
        data = {'Number_of_announcements': Number_of_announcements,
                'Number_of_announced_NLRI_prefixes': Number_of_announced_NLRI_prefixes,
                'Maximum_AS_path_length': Maximum_AS_path_length,
                'Average_edit_distance': Average_edit_distance,
                'Maximum_edit_distance': Maximum_edit_distance,
                'Number_of_Interior_Gateway_Protocol_IGP_packets':  Number_of_Interior_Gateway_Protocol_IGP_packets,
                'Number_of_incomplete_packets': Number_of_incomplete_packets}
                #'Labels': labels_predictor}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()
# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
slammer_raw = pd.read_csv('slammer_cleaned.csv')
slammer = slammer_raw.drop(columns=['Labels'],axis=1)
df = pd.concat([input_df,slammer],axis=0)
# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering

df = df[:1] # Selects only the first row (the user input data)
#Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)
    # Reads in saved classification model
load_clf = pickle.load(open('Slammer_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)
st.subheader('Prediction')
slammer_Labels = np.array(['Normal','Slammer'])
st.write(slammer_Labels[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
