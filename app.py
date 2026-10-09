#python -m pip install gradio
#python -m pip install scikit-learn
#python -m pip install numpy

import gradio as gr # type: ignore
import pickle
import numpy as np # type: ignore

model = pickle.load(open(r'regressor.pkl','rb'))
scaler = pickle.load(open(r'scaler.pkl','rb'))

def calculate_goldrate(usd_inr):
    scaled_input = scaler.transform(np.array(usd_inr).reshape(1,-1))#[1.054501]
    prediction = model.predict(scaled_input)[0]
    # Convert numpy array to Python float and then round
    return round(float(prediction.item()), 2)

demo = gr.Interface(
    fn=calculate_goldrate,
    inputs=["number"],
    outputs=["number"],
    title="How much is 1g of gold in India NOW?"
)

demo.launch(share=True)