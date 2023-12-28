import math
import pickle
import numpy as np
import gradio as gr


def visible_component():
    return gr.update(visible=True)
def calculate(room):
    yangi_data_point = np.array(int(room)).reshape(-1, 1)
    yangi_narx = model.predict(yangi_data_point)
    return int(math.ceil(yangi_narx[0]))


# load model
with open('models/linearReg-Samariddin.pkl', 'rb') as f:
    model = pickle.load(f)

with gr.Blocks() as demo:
    room = gr.Textbox(label="Count of room")
    calculate_btn = gr.Button("Calculate")
    price = gr.Textbox(label="Price", visible=False)
    calculate_btn.click(fn=calculate, inputs=room, outputs=price, api_name="calculate").then(fn=visible_component, outputs=price)


if __name__ == "__main__":
    demo.launch()