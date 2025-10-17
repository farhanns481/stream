import streamlit as st
import pickle
import numpy as np

# Load your trained model
model = pickle.load(open("model/model_final.pkl", "rb"))

# App Title
st.title("🌸 IRIS FLOWER PREDICTION APP")
st.markdown("This app predicts the **species of Iris flower** based on the given features.")

# Sidebar input section
st.sidebar.header("Input Parameters")
st.sidebar.markdown("Enter the details below:")

# Inputs from user
sepal_length = st.sidebar.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.sidebar.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.sidebar.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.sidebar.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)

# Prediction button
if st.sidebar.button("🔍 Predict"):
    # Prepare input data
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display result
    st.subheader("🌼 Prediction Result")
    st.success(f"The predicted Iris species is: **{prediction}**")

    # Optional image display (based on result)
    if prediction == "Iris-setosa":
        st.image("https://upload.wikimedia.org/wikipedia/commons/a/a7/Irissetosa1.jpg", caption="Iris Setosa")
    elif prediction == "Iris-versicolor":
        st.image("https://upload.wikimedia.org/wikipedia/commons/3/38/Iris_versicolor_3.jpg", caption="Iris Versicolor")
    else:
        st.image("https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg", caption="Iris Virginica")

# Info section
st.markdown("---")
st.info("💡 Tip: Adjust the feature values from the sidebar to see how the prediction changes.")
