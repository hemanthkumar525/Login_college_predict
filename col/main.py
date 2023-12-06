import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import streamlit_authenticator as stauth
from dependancies import sign_up,fetch_users

try:
    st.set_page_config(page_title='College Predictor',page_icon='🔒')
    users = fetch_users()
    emails = []
    usernames = []
    passwords = []

    for user in users:
        emails.append(user['key'])
        usernames.append(user['username'])
        passwords.append(user['password'])

    credentials = {'usernames': {}}
    for index in range(len(emails)):
        credentials['usernames'][usernames[index]] = {'name': emails[index], 'password': passwords[index]}

    Authenticator = stauth.Authenticate(credentials, cookie_name='Streamlit', key='abcdef', cookie_expiry_days=1)

    email, authentication_status, username = Authenticator.login(':green[Login]', 'main')
    

    info, info1 = st.columns(2)

    if not authentication_status:
        sign_up()


    if username:
        if username in usernames:
            if authentication_status:
                st.title('College Predictor Using the Scores')
                data = pd.read_csv('Admission_Predict.csv')
                df = pd.DataFrame(data)

#input features
                st.sidebar.subheader('input features')
                GRE = st.sidebar.slider('GRE_Score', 290, 340, 295)
                CGPA = st.sidebar.slider('CGPA_Score', 7.0, 9.9, 8.5)

                X = df[['GRE Score', 'CGPA']]
                y = df['university_name']

# Split the data into a training set and a testing set
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Random Forest Classifier
                model = RandomForestClassifier()
                model.fit(X_train, y_train)

# Predict university names for test data
                predictions = model.predict(X_test)

# Evaluate the model (You can use appropriate evaluation metrics)
                accuracy = (predictions == y_test).mean()

                a = GRE
                b = CGPA

# Use the model to predict university names for new applicants
                new_applicant = pd.DataFrame({'GRE Score': [a], 'CGPA': [b]})
                predicted_university = model.predict(new_applicant)
                st.write(predicted_university[0])
                Authenticator.logout('Log Out', 'sidebar')
            elif not authentication_status:
                with info:
                    st.error('Incorrect Password or username')
            else:
                with info:
                    st.warning('Please feed in your credentials')
        else:
            with info:
                st.warning('Username does not exist, Please Sign up')
    
except:
    st.success('Refresh Page')
