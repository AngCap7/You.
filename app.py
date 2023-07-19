from operator import index
import streamlit as st
from pandas_profiling import *
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os 
import matplotlib.pyplot as plt
from pycaret.regression import save_model, setup, compare_models, load_model, pull
from pycaret.classification import *
from AutoClean import AutoClean
from io import *
import io
import pdfkit
import statsmodels.api as sm
import numpy as np


if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)
    

with st.sidebar: 
    st.image("http://www.ilfascinodellamatematica.com/wp-content/uploads/2018/12/16-1.png")
    st.title("You.Stats")
    choice = st.radio("Navigation", ["Upload",'Preprocessing',"Profiling","Modelling", "Download Best Model", "Time Series"])
    st.info("This project application helps you build and explore your data.")


if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)


if choice=='Preprocessing':
    st.title("Getting started with Data Preprocessing")
    df=pd.DataFrame(df)
    df_clean=AutoClean(df)
    new_df=pd.DataFrame(df_clean.output)
    new_df=new_df.select_dtypes(include=['int','float'])
    new_df= new_df.apply(pd.to_numeric, errors='ignore')
    buffer = io.BytesIO()
    new_df.to_csv(buffer, index=False)
    buffer.seek(0)
    st.download_button('Download Preprocessed Dataset', data=buffer, file_name='preprocessed_dataset.csv')
    

if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    profile_df = ProfileReport(df)
    st_profile_report(profile_df)
    profile_pdf=profile_df.to_file('report.html')
    config = pdfkit.configuration(wkhtmltopdf=r"C:\Users\redbu\OneDrive\Desktop\dati vari\wkhtmltopdf\bin\wkhtmltopdf.exe")
    pdfkit.from_file('report.html', 'report.pdf', configuration=config)
    with open('report.pdf', 'rb') as f:
        st.download_button('Download PDF', f, file_name='report.pdf')
    
if choice == "Modelling":
    chosen_target = st.selectbox('Choose the Target Column',df.columns)
    if st.button('Run Modelling'): 
        setup(df, target=chosen_target)
        setup_df = pull()
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')
         
    
if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")
        

if choice == "Time Series":
    st.title("Upload Your Time Series")
    ts=st.file_uploader("Upload here")
    if ts: 
        serie = pd.read_csv(ts, index_col=None)
        serie.to_csv('time_series.csv', index=None)
        st.dataframe(serie)
    graf=serie.plot()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    st.image(buffer, use_column_width=True)
    



        
        
        
    

    
    