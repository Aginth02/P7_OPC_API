import streamlit as st
from streamlit_option_menu import option_menu
import seaborn as sns
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
# Model and performance
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
import lightgbm
import pickle
import re 

# Importation des jeux de données
dataset_train = pd.read_csv('data2/df_train_final.csv',index_col=0)
dataset_test = pd.read_csv('data2/df_test_final.csv',index_col=0)

target_train = pd.read_csv('data2/target_train_final.csv',index_col=0)
model = pickle.load(open('data2/best_model.pickle', 'rb'))

id_client = list(dataset_test.index)


st.set_page_config('Accord Crédit')
original_title = '<p style="font-family:Arial; color:Red; font-size: 60px;text-align: center;"> Application de prediction de prêt bancaire</p>'
st.markdown(original_title, unsafe_allow_html=True)





form2 = st.form("template_form1")
analyse_client = form2.selectbox("Choix Clients",id_client,index=0)
submit_client = form2.form_submit_button("Prediction")

if submit_client :        
       
    X = dataset_test.loc[analyse_client]

    probability_default_payment = model.predict_proba(pd.DataFrame(X.values,index=X.index).T)[:, 1]
    client = "<h1 style='text-align: center; color: black;'>CLIENT " + str(analyse_client)+"</h1>"
    st.markdown(client, unsafe_allow_html=True)

        

    defaut = "<h7 style='font-family:Arial; color:Black; font-size: 20px;'>Probabilité de défaut de paiement : "+str(np.round(probability_default_payment[0],2)*100)+" % </h7>"

    st.markdown(defaut,unsafe_allow_html=True)
    if probability_default_payment > 0.5 : 
        st.markdown('<h8 style="font-family:Arial; color:Red; font-size: 40px;text-align: center;">CREDIT REFUSE</h8>',unsafe_allow_html=True)
    else :
        st.markdown('<h9 style="font-family:Arial; color:Red; font-size: 40px;text-align: center;">CREDIT ACCORDE</h9>',unsafe_allow_html=True)

        
 
