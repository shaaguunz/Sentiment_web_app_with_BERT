import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load the model and tokenizer
model_path = "./saved_model"
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Function to classify text
def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    return predictions.item()

# Streamlit app
st.set_page_config(page_title="Text Sentiment Classification", page_icon="🔍", layout="centered")

# Custom styles
st.markdown("""
    <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            font-size: 16px;
            padding: 10px 24px;
        }
        .stTextArea textarea {
            border-radius: 10px;
            font-size: 16px;
            padding: 10px;
        }
        .stMarkdown h2 {
            color: #2c3e50;
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

st.title("🔍 Text Sentiment Classification")
st.subheader("Analyze the sentiment of your text using DistilBERT")

st.markdown("Enter the text you want to classify, and the app will predict whether it's **positive**, **neutral**, or **negative**.")

text_input = st.text_area("Enter text to classify", placeholder="Type your text here...")

if st.button("Classify Text"):
    if text_input:
        prediction = classify_text(text_input)
        if prediction == 2:
            st.success("💬 **Prediction**: This review is **positive** 🎉")
        elif prediction == 1:
            st.info("💬 **Prediction**: This review is **neutral** 😐")
        else:
            st.error("💬 **Prediction**: This review is **negative** 😞")
    else:
        st.warning("⚠️ Please enter some text to classify.")

# Footer
st.markdown("""
    <hr>
    <div style='text-align: center;'>
        <p>Built with ❤️ using <a href='https://streamlit.io/' target='_blank'>Streamlit</a> and <a href='https://huggingface.co/' target='_blank'>Hugging Face Transformers</a>.</p>
    </div>
    """, unsafe_allow_html=True)
