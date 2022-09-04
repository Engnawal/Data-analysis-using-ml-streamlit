import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from PIL import Image
import streamlit as st
from sklearn import *
import os


# importing the csv module
import csv

# field names
fields = (["Age","Gender","Total_Bilirubin","Direct_Bilirubin","Alkaline_Phosphotase","Alamine_Aminotransferase","Aspartate_Aminotransferase","Total_Protiens","Albumin","Albumin_and_Globulin_Ratio", "Dataset"])
# name of csv file
filename = "liver_patient.csv"

# writing to csv file
with open(filename, 'w') as csvfile:
	# creating a csv writer object
	csvwriter = csv.writer(csvfile)
	
	# writing the fields
	csvwriter.writerow(fields)
	
	# writing the data rows
	csvwriter.writerows(fields)

# Create a title and a subtitle
st.title('Liver Disease Prediction System')
st.write(""" This program detect if the person has liver disease or not by using machine learning and python. """)
image = Image.open(os.path.join(r'C:/Users/DELL/Desktop/AI.webp'))
st.image(image, caption='ML', use_column_width=True)

# Get the Data
df = pd.read_csv('C:/Users/DELL/Desktop/liver_patient.csv')

#Set a subheader
st.subheader('Data information')

#Show the data as table
df_data = pd.read_csv('C:/Users/DELL/Desktop/liver_patient.csv')
st.write(df_data.head())
st.write(df.describe())
st.subheader('Visualization of Age distribution on Liver dataset')
Age_dist = pd.DataFrame(df_data['Age'].value_counts()).head(100)
st.bar_chart(Age_dist)

#Split the data into independent variable and dependent variable.Where MLP model is expecting 9 features as input from the user.
X = df.iloc[:, :9].values
Y = df.iloc[:, -1].values


#Split the dataset into 75% Training and 25% Testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
select = st.sidebar.selectbox('Select blood test', ['Liver blood test'], key='1')
if not st.sidebar.checkbox("Hide", True, key='1'):
    st.title('Personal information:')
    name = st.text_input("Name:")
    n_estimetor = st.selectbox("Gender:", options=['Male', 'Female'], index=0)
    n_estimetor = st.selectbox("Obesity:", options=['Yes', 'No'], index=0)
    n_estimetor = st.selectbox("Smoker:", options=['Yes', 'No'], index=0)
    
    st.title('Symptoms:')
    n_estimetor = st.selectbox("Skin and eyes that appear yellowish:", options=['Yes', 'No'], index=0)
    n_estimetor = st.selectbox("Abdominal pain and swelling:", options=['Yes', 'No'], index=0)
    n_estimetor = st.selectbox("Swelling in the legs and ankles:", options=['Yes', 'No'], index=0)
    n_estimetor = st.selectbox("Itchy skin:", options=['Yes', 'No'], index=0)
    n_estimetor = st.selectbox("Dark urine color:", options=['Yes', 'No'], index=0)
    n_estimetor = st.selectbox("Pale stool color:", options=['Yes', 'No'], index=0)
    n_estimetor = st.selectbox("Chronic fatigue:", options=['Yes', 'No'], index=0)
    n_estimetor = st.selectbox("Nausea or vomiting:", options=['Yes', 'No'], index=0)
    n_estimetor = st.selectbox("Loss of appetite:", options=['Yes', 'No'], index=0)
    n_estimetor = st.selectbox("Tendency to bruise easily:", options=['Yes', 'No'], index=0)
    
    #Set slider value 
    Age = st.slider('Age', 5, 20, 89)
    Total_Bilirubin = st.slider(' Total_Bilirubin', 0, 15, 20)
    Direct_Bilirubin = st.slider(' Direct_Bilirubin', 0, 3, 5)
    Alkaline_Phosphotase = st.slider(' Alkaline_Phosphotase', 0, 44, 147)
    Alamine_Aminotransferase = st.slider(' Alamine_Aminotransferase', 0, 29, 33)
    Aspartate_Aminotransferase = st.slider(' Aspartate_Aminotransferase', 0, 1, 45)
    Total_Protiens = st.slider('Total_Protiens', 0, 6, 9)
    Albumin = st.slider('Albumin', 0, 3, 5)
    Albumin_and_Globulin_Ratio = st.slider('Albumin_and_Globulin_Ratio ', 0, 3, 5)

submit = st.button('Predict')

# Create and train the model
classifier = MLPClassifier()
classifier.fit(X_train, Y_train)

# Show the model matrix
st.subheader('Model Test Accuracy Score:')
st.write(str(accuracy_score(Y_test, classifier.predict(X_test)) * 100) + '%')
# store the model prediction and variable
if submit:
    prediction = classifier.predict([[Age, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase,
                                      Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Protiens, Albumin,
                                      Albumin_and_Globulin_Ratio]])
    if (prediction[0]== 0):
        st.write('Congratulation', name, 'You are not affected.')
        st.balloons()
    else:
        st.write(name, " we are really sorry to say that but it seems like you need medical care.")




        
    
