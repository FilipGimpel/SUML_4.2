import streamlit as st
import pickle
import pandas as pd

base_data = pd.read_csv("DSP_8.csv")

base_data = base_data[base_data.RestingBP != 0]
base_data = base_data[base_data.Cholesterol != 0]
base_data = base_data[base_data.RestingBP != 0]

filename = "model.sv"
model = pickle.load(open(filename, 'rb'))

sex_d = {0: "Male", 1: "Female"}
chest_pain_type = {0: "ATA", 1: "NAP", 2: "ASY", 3: "TA"}
resting_ecg = {0: "Normal", 1: "ST", 2: "LVH"}
exercise_angina = {0: "No", 1: "Yes"}
st_slope = {0: "Up", 1: "Flat", 2: "Down"}


def main():
    st.set_page_config(page_title="Heart disease prediction")
    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()

    st.image(
        "https://www.hsph.harvard.edu/nutritionsource/wp-content/uploads/sites/30/2016/04/CVD_NSHomepageWidget.jpg")

    with overview:
        st.title("Heart disease prediction")

    with left:
        sex_radio = st.radio("Gender", list(sex_d.keys()), format_func=lambda x: sex_d[x])
        chest_pain_type_radio = st.radio("Chest pain type", list(chest_pain_type.keys()),
                                         format_func=lambda x: chest_pain_type[x])
        resting_ecg_radio = st.radio("Resting ECG", list(resting_ecg.keys()), format_func=lambda x: resting_ecg[x])
        exercise_angina_radio = st.radio("Exercise angina", list(exercise_angina.keys()),
                                         format_func=lambda x: exercise_angina[x])
        st_slope_radio = st.radio("ST Slope", list(st_slope.keys()), format_func=lambda x: st_slope[x])

    with right:
        min_age = int(base_data['Age'].min())
        max_age = int(base_data['Age'].max())

        min_rest_bp = int(base_data['RestingBP'].min())
        max_rest_bp = int(base_data['RestingBP'].max())

        min_chol = int(base_data['Cholesterol'].min())
        max_chol = int(base_data['Cholesterol'].max())

        # min_bs = int(base_data['FastingBS'].min())
        # max_bs = int(base_data['FastingBS'].max())

        min_hr = int(base_data['MaxHR'].min())
        max_hr = int(base_data['MaxHR'].max())

        min_oldpeak = int(base_data['Oldpeak'].min())
        max_oldpeak = int(base_data['Oldpeak'].max())

        age_slider = st.slider("Age", min_value=min_age, max_value=max_age, step=1)
        rest_bp_slider = st.slider("Resting BP", min_value=min_rest_bp, max_value=max_rest_bp, step=1)
        chol_slider = st.slider("Cholesterol", min_value=min_chol, max_value=max_chol, step=1)
        hr_slider = st.slider("Max HR", min_value=min_hr, max_value=max_hr, step=1)
        oldpeak_slider = st.slider("Oldpeak", min_value=min_oldpeak, max_value=max_oldpeak, step=1)

    data = [[sex_radio, chest_pain_type_radio, resting_ecg_radio, exercise_angina_radio, st_slope_radio, age_slider,
             rest_bp_slider, chol_slider, hr_slider, oldpeak_slider]]

    column_names = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
            'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

    data = pd.DataFrame(data, columns=column_names)
    survival = model.predict(data)
    s_confidence = model.predict_proba(data)

    with prediction:
        st.subheader("Does the model predicts chance of heart disease?")
        st.subheader(("No" if survival[0] == 1 else "Yes"))
        st.write("Certanity {0:.2f} %".format(s_confidence[0][survival][0] * 100))


if __name__ == "__main__":
    main()
