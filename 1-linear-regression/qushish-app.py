# Qo'shish

import numpy as np
from sklearn.linear_model import LinearRegression

import gradio as gr


# data
data_qushish_jadvali = [ [a, b, a+b] for a in range(1, 11) for b in range(1, 11)]

# numpy
data = np.array(data_qushish_jadvali)

X = data[:, :2]
y = data[:, 2]

model = LinearRegression()
model.fit(X, y)

def qushish(a, b):
    """
    masalan: qushish(5, 6) ==> 11
    """
    data_np = np.array([[int(a), int(b)]])
    natija = model.predict(data_np)
    return int(natija[0])

with gr.Blocks() as demo:
    gr.Markdown("# Mashina qo'shish amalini bajaradi")
    a = gr.Textbox(label="Birinchi son")
    b = gr.Textbox(label="Ikkinchi son")
    yigindi = gr.Textbox(label="Yig'indi ")
    qush = gr.Button("Qo'shish")
    qush.click(fn=qushish, inputs=[a, b], outputs=yigindi)

demo.launch(share=True)
