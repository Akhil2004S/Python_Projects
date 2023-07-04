import gradio as gr
x = gr.Interface(fn=crypto_prediction,
                 inputs=['text','number','number','number'],
                 outputs=['number'])
x.launch(debug = True)