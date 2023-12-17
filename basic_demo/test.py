import gradio as gr

with gr.Blocks() as demo:
    a = gr.File(value=['data2/中国石化.pptx'])

demo.launch()