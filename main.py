import gradio as gr
from transformers import pipeline

def greet(sentence):
    result = pipeline('sentiment-analysis')(sentence)
    label = result[0]['label']
    score = result[0]['score']
    res = { label: score, "POSITIVE" if label=='NEGATIVE' else "NEGATIVE": 1-score}
    return res


demo = gr.Interface(fn=greet, inputs=gr.Textbox(placeholder="Enter a positive or negative sentence here..."),
                    outputs="label", interpretation="default")

if __name__ == "__main__":
    demo.launch()