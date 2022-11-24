# -----------------------------------------Importing Packages--------------------------------------

import requests
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_lottie import st_lottie
import json
import requests
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler

# -----------------------------------------Importing Packages--------------------------------------

# --------------------------------------------Page Layout------------------------------------------

con = st.container()

def lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
lottie_ani = lottieurl('https://assets7.lottiefiles.com/packages/lf20_lFXAtJ.json')
ani = st_lottie(lottie_ani, key = "hello")

title = con.title('Transaction Analysis')
anim = con.ani
txt = con.subheader('Kindly upload CSV file for generating report')
uploaded_file = con.file_uploader("Choose a file")
score = con.text('Accuracy: 98.8%')

# --------------------------------------------Page Layout------------------------------------------

# ---------------------------------------Back End (PIPELINE)----------------------------------------

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # 2) Data Engineering:

    df.dropna(how='any', inplace=True)

    sc = StandardScaler()
    df['Amount'] = sc.fit_transform(pd.DataFrame(df['Amount']))

    df = df.drop(['Time'], axis=1)

    df = df.drop('Class', axis=1)

    # 3) Prediction:

    model = joblib.load('Fraud_Detection')
    result = model.predict(df)

    # 4) Result:

    df.insert(29, "Result", result, True)

    def convert_df(df):
        return df.to_csv().encode('utf-8')

    csv = convert_df(df)

    st.download_button(
        label="Download Report as CSV",
        data=csv,
        file_name='Report.csv',
        mime='text/csv',
    )

    # 5) Report:
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    x, y = df['Result'].value_counts()
    labels = 'Not a Fraud Transaction', 'Fraud Transaction'
    sizes = [x, y]
    # explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.pyplot(fig1)