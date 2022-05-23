import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, plot_confusion_matrix, classification_report, confusion_matrix

from preprocessor import preprocess


def model_compare():
    st.title("Let's Compare Some Models")
    df = pd.read_csv('./Datasets/athlete_events.csv')
    region_df = pd.read_csv('./Datasets/noc_regions.csv')

    st.sidebar.write("""
            ### Select Season
        """)

    options = (
        'Summer',
        'Winter'
    )

    season = st.sidebar.radio('', options)

    df = preprocess(df, region_df, season)

    clean_df = df.dropna()

    data = clean_df[["Height", "Weight", "Sport"]]
    sports = data['Sport'].unique().tolist()
    sports.insert(0, 'All')

    st.write("""
                ### Select Sport
            """)
    sports = st.selectbox("", sports)

    if sports == "All":
        data['Sport'] = data['Sport'].map({
            'Shooting': 1,
            'Athletics': 2,
            'Handball': 3,
            'Water Polo': 4,
            'Football': 5,
            'Basketball': 6,
            'Volleyball': 7,
            'Swimming': 8,
            'Rowing': 9,
            'Gymnastics': 10,
            'Wrestling': 11,
            'Table Tennis': 12,
            'Sailing': 13,
            'Canoeing': 14,
            'Boxing': 15,
            'Judo': 16,
            'Cycling': 17,
            'Tennis': 18

        })
    else:
        data['Sport'] = np.where(data['Sport'] == sports, 1, 0)

    fig = plt.figure(figsize=(5, 5))
    sns.scatterplot(x="Weight",
                    y="Height",
                    hue="Sport",
                    data=data)
    st.pyplot(fig)

    y = data['Sport'].copy()
    X = data.drop('Sport', axis=1).copy()

    st.write("""
            #### Choose Model
            """)
    choice = st.selectbox(' ', ['KNeighbors','Random Forest'])

    st.write("""
            #### Test Data Percentage
            """)
    test_data = st.slider(' ', 10, 90, value=30)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_data / 100)

    if choice == "Random Forest":
        col_1, col_2, col_3 = st.columns(3)

        with col_1:
            n_estimators = math.ceil(st.number_input('n_estimators', 0, 150, value=60))
        with col_2:
            n_jobs = math.ceil(st.number_input('n_jobs', 0, 10, value=5))
        with col_3:
            max_depth = math.ceil(st.number_input('max_depth', 0, 10, value=0))

        if max_depth == 0:
            max_depth = None

        col_1, col_2, col_3 = st.columns(3)

        with col_1:
            oob_score = st.checkbox('oob_score')

        with col_2:
            criterion = st.radio('criterion', ['gini', 'entropy'])

        with col_3:
            max_features = st.radio('max_features', ['auto', 'sqrt', 'log2'])

        rfc = RandomForestClassifier(n_jobs=n_jobs, oob_score=oob_score, n_estimators=n_estimators, criterion=criterion,
                                     max_depth=max_depth, max_features=max_features)
        rfc.fit(X_train, y_train)

        rfc_pred = rfc.predict(X_test)
        rfc_score = accuracy_score(y_test, rfc_pred)

        st.write(""" #### Classification Report """)
        st.text(classification_report(y_test, rfc_pred))

        st.write(""" #### Accuracy Score is {0:.2f} %""".format(rfc_score * 100))

        if sports == "All":
            button = st.button('Save Random Forest as Pickle')

            if button:
                data = {"model": rfc}
                with open('./Pickle/random_forest.pkl', 'wb') as file:
                    pickle.dump(data, file)

    elif choice == "KNeighbors":
        col_1, col_2, col_3 = st.columns(3)

        with col_1:
            n_neighbors = math.ceil(st.number_input('n_neighbors', 0, 50, value=25))
        with col_2:
            n_jobs = math.ceil(st.number_input('n_jobs', 0, 10, value=5))
        with col_3:
            weights = st.selectbox('weights', ('uniform','distance'))

        algorithm = st.radio('algorithm ', ['auto', 'ball_tree','kd_tree','brute'])

        knc = KNeighborsClassifier(n_jobs=n_jobs, n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
        knc.fit(X_train, y_train)

        knc_pred = knc.predict(X_test)
        knc_score = accuracy_score(y_test, knc_pred)

        st.write(""" #### Classification Report """)
        st.text(classification_report(y_test, knc_pred))

        st.write(""" #### Accuracy Score is {0:.2f} %""".format(knc_score * 100))

        if sports == "All":
            button = st.button('Save KNeighbors as Pickle')

            if button:
                data = {"model": knc}
                with open('./Pickle/KNeighbors.pkl', 'wb') as file:
                    pickle.dump(data, file)
