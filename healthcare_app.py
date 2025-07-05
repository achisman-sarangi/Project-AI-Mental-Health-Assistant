import os
import streamlit as st
from chat_llm import get_response
from sentiment import get_sentiment
from PIL import Image

st.set_page_config(page_title="Healthcare App", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
        body {
            background: linear-gradient(160deg, #0f1117 0%, #1c1f26 100%) !important;
            color: #e0e0e0;
        }

        .appview-container .main {
            background: linear-gradient(160deg, #0f1117 0%, #1c1f26 100%) !important;
        }

        .stApp {
            background: linear-gradient(160deg, #0f1117 0%, #1c1f26 100%) !important;
            color: #e0e0e0;
        }

        .stTextInput > div > div > input {
            background-color: #20232a !important;
            color: #ffffff !important;
            border: 1px solid #444 !important;
            border-radius: 10px;
            padding: 10px;
        }

        .stTextInput label, .stTextArea label {
            font-weight: bold;
            color: #cccccc;
        }

        .custom-container {
            background: linear-gradient(145deg, #1f2128, #15171c);
            border-radius: 12px;
            padding: 20px;
            margin-top: 20px;
            box-shadow: 2px 2px 6px #0d0f13;
            border-left: 6px solid #00c6ff;
        }

        .custom-sentiment {
            background: linear-gradient(145deg, #1c1c1c, #141414);
            border-left: 6px solid #ffcc00;
            padding: 20px;
            border-radius: 12px;
            margin-top: 20px;
            box-shadow: 2px 2px 6px #0d0f13;
        }

        .title-style {
            font-size: 40px;
            font-weight: 800;
            background: linear-gradient(90deg, #ff5c83, #00c6ff, #8aff80);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: inline-block;
            padding-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.4);
        }

        .subtitle-style {
            font-size: 17px;
            color: #bbbbbb;
            margin-bottom: 25px;
        }

        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #f2f2f2;
        }

        .footer {
            margin-top: 40px;
            text-align: center;
            color: #888888;
            font-size: 13px;
        }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("🩺 About")
    st.markdown("""
        - Built using OpenAI + LangChain  
        - Mental health guidance  
        - Emotional sentiment analyzer  
    """)
    st.markdown("Made with ❤️ by **ACHISMAN SARANGI**")

logo_path = "data/Logo.png"
if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    st.image(logo, width=180)

st.markdown('<div class="title-style">🧠 AI Mental Health Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-style">Understand your mental health and emotions through AI-powered insights</div>', unsafe_allow_html=True)

user_input = st.text_input("Enter your mental health concern or query")
if user_input:
    with st.spinner("Processing your query..."):
        response = get_response(user_input)
        st.markdown(f'<div class="custom-container"><strong>Response from AI Assistant:</strong><br>{response}</div>', unsafe_allow_html=True)

sentiment_input = st.text_input("Enter text for sentiment analysis")
if sentiment_input:
    with st.spinner("Analyzing sentiment..."):
        sentiment, score = get_sentiment(sentiment_input)
        st.markdown(f"""
            <div class="custom-sentiment">
                <strong>Sentiment Analysis Result:</strong><br>
                Sentiment: <b>{sentiment}</b><br>
                Confidence Score: <b>{score:.2f}</b>
            </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="footer">© 2025 Mental Health AI Assistant | Empowering Wellness with Technology</div>', unsafe_allow_html=True)
