import streamlit as st
import pickle
import math


def load_model(model):
    if model == "Random Forest":
        with open('./Pickle/random_forest.pkl', 'rb') as file:
            data = pickle.load(file)
    elif model == "KNeighbors":
        with open('./Pickle/KNeighbors.pkl', 'rb') as file:
            data = pickle.load(file)
    return data


def show_predict_page():
    st.title("Let's Predict!")

    st.write("""### We need some information to predict""")

    st.write("""
        #### Enter Height
    """)
    height = math.ceil(st.number_input('', 10, 500, value=100))

    st.write("""
            #### Enter Weight
        """)
    weight = math.ceil(st.number_input('', 10, 500, value=50))

    algo = st.radio('Algo', ['Random Forest', 'KNeighbors'])

    data = load_model(algo)

    model = data["model"]

    ok = st.button("Predict")

    if ok:
        decision = model.predict([[height,weight]])
        dict_ = {
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
        }
        for sport, index in dict_.items():
            if index == decision:
                st.write("Predicted Sport is ", sport)
