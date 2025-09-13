import tensorflow as tf
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler,OneHotEncoder, LabelEncoder
import pickle


model=tf.keras.models.load_model("model.h5")

with open('label_encoder_gender.pkl','rb') as file:
    gender_encoder=pickle.load(file)

with open('onehot_encoder_georaphy.pkl','rb') as file:
    geography_encoder=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)


st.title("Chrun Prediction")

geography=st.selectbox("Geography",geography_encoder.categories_[0])
gender=st.selectbox('Gender',gender_encoder.classes_)
age=st.slider("Age",min_value=0,max_value=92)
balance=st.number_input("Balance")
credit_score=st.number_input("Credit_Score")
estimated_salery=st.number_input("Estimated_Salery")
tenure=st.slider("Tenure",0,10)
num_of_products=st.slider("Number_of_Products",1,4)
has_cr_card=st.selectbox("Has Credit Card",[0,1])
is_active_member=st.selectbox("Is Acitve Member",[0,1])


input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[gender_encoder.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salery]
})


geo_encoder=geography_encoder.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoder,columns=geography_encoder.get_feature_names_out(["Geography"]))

input_data=pd.concat([input_data,geo_encoded_df],axis=1)


data_scaled=scaler.transform(input_data)
prediction=model.predict(data_scaled)
prediction_prob=prediction[0][0]
st.text(f"the value is: {prediction_prob:.2f}")

if prediction_prob>0.5 :
    st.text("the person churns")
else:
    st.text("person not churns")    