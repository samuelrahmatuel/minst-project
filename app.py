import streamlit as st
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np

# Load the trained model from disk
model = tf.keras.models.load_model("my_model.h5")

# Define a function to preprocess the input and make a prediction
def predict_output(input_data):
    # Preprocess the input data
    preprocessed_data = preprocess_input(input_data)

    # Use the model to make a prediction
    prediction = model.predict(preprocessed_data)

    return prediction

# Function to preprocess the drawn image
def preprocess_input(img):
    img = cv2.resize(img, (28, 28))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Define the Streamlit app
def app():
    st.set_page_config(page_title="Handwritten Digit Recognition", page_icon=":pencil2:")

    st.title("Handwritten Digit Recognition")
    st.markdown("## :pencil2: Draw a digit from 0 to 9")

    st.write("This app uses a trained machine learning model to recognize handwritten digits. Draw a digit (0-9) in the canvas below and click 'Submit' to see the predicted result. *WIP Project*")

    # Add a drawing canvas
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 1)",
        stroke_width=20,
        stroke_color="white",
        background_color="black",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    # Add input widgets
    submit_button = st.button("Submit")

    # When the user clicks the submit button, make a prediction
    if submit_button:
        if canvas_result.image_data is not None:
            input_data = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)
            prediction = predict_output(input_data)
            predicted_digit = np.argmax(prediction)
            st.success(f"Prediction: {predicted_digit}")
        else:
            st.warning("Please draw a digit in the canvas.")
            
    st.write("Next Progress : Train Model more, Implementing Deep Learning for recommendation, Add more features")
    st.markdown("<p style='text-align: right; font-style: italic;'>Created by: <a href='https://rahmatuelsamuel.com'>Rahmatuel Samuel</a></p>", unsafe_allow_html=True)

if __name__ == "__main__":
    app()
