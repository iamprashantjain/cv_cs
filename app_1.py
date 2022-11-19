import streamlit as st
import pickle
import numpy as np
import pdb
import pandas as pd



df = pd.read_csv('model_data.csv',encoding='latin-1')
loaded_model=pickle.load(open('regressor_CV_CS.pkl','rb'))


st.set_page_config(layout="wide")
st.title("CV Model Prediction (CS)")
st.image('https://web.samil.in/wp-content/uploads/2020/01/Theme1/Desktop-T_1.png')

col1,col2,col3,col4 = st.columns(4)
col5,col6,col7,col8 = st.columns(4)


placeholder = st.empty()


make  =df['Make_Clean'].unique().tolist()
Make_Clean = col1.selectbox('Select Make',make)


model = df['Model_Clean'].loc[df['Make_Clean'] == Make_Clean].drop_duplicates()
Model_Clean = col2.selectbox('Select Model',model)

mk_yr = df['MAKE_YEAR'].loc[df['Make_Clean'] == Make_Clean].drop_duplicates()
MAKE_YEAR = col3.selectbox('Select Make Year',mk_yr)


variant = df['Variant_Clean'].loc[df['Make_Clean'] == Make_Clean].drop_duplicates()
Variant_Clean = col4.selectbox('Select Variant',variant)


fuel = df['Fuel_Clean'].unique().tolist()
Fuel_Clean = col5.selectbox('Select Fuel',fuel)


METER_READING_cleaned = col7.number_input('Enter Meter Reading',min_value=100)

state = df['CV_State_Clean'].unique().tolist()
CV_State_Clean = col8.selectbox('Select State',state)


cust_seg = df['Customer_Segmentation'].loc[df['Make_Clean'] == Make_Clean].drop_duplicates()
Customer_Segmentation = col6.selectbox("Select Customer Segmentation",cust_seg)


X = pd.DataFrame(columns=['MAKE_YEAR','Make_Clean','Model_Clean','Variant_Clean','Fuel_Clean','CV_State_Clean','Customer_Segmentation','METER_READING_cleaned'],data=np.array([MAKE_YEAR,Make_Clean,Model_Clean,Variant_Clean,Fuel_Clean,CV_State_Clean,Customer_Segmentation,METER_READING_cleaned]).reshape(1,8))


if st.button('Get the Best Price'):
	predicted_amount = loaded_model.predict(X)
	st.subheader(f"The Estimated CV Price is ₹ {predicted_amount[0]:.2f}")
	
