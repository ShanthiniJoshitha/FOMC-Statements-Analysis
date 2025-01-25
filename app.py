from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import streamlit as st

# Define the model name
model_name = "Respair/deberta-v3-large-finetuned-style"  # Replace with a verified model name

# Load the tokenizer and model with error handling
try:
    # Try loading the fast tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
except Exception as e:
    # If fast tokenizer fails, fall back to the slow tokenizer
    st.warning("Error loading fast tokenizer. Falling back to slow tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# Load the model
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Define the sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Function to process the input statement
def process_statement(statement):
    lines = statement.split('.')
    sentiments = []

    for line in lines:
        line = line.strip()
        if line:  # Avoid empty lines
            result = sentiment_analyzer(line)
            sentiments.append(result[0]['label'].lower())

    positive_count = sentiments.count('positive')
    negative_count = sentiments.count('negative')
    neutral_count = sentiments.count('neutral')
    total = len(sentiments)

    positive_percentage = (positive_count / total) * 100 if total > 0 else 0
    negative_percentage = (negative_count / total) * 100 if total > 0 else 0
    neutral_percentage = (neutral_count / total) * 100 if total > 0 else 0

    overall_sentiment = max(sentiments, key=sentiments.count) if sentiments else "neutral"

    return {
        "Positive Percentage": positive_percentage,
        "Negative Percentage": negative_percentage,
        "Neutral Percentage": neutral_percentage,
        "Overall Sentiment": overall_sentiment
    }

# Streamlit App
st.title("Sentiment Analysis App")
st.write("Enter a statement below to analyze its sentiment:")

# Input text area for user input
user_input = st.text_area("Your statement:", placeholder="Type here...")

# Button to analyze sentiment
if st.button("Analyze Sentiment"):
    if user_input.strip():
        try:
            result = process_statement(user_input)
            st.subheader("Sentiment Analysis Results:")
            st.write(f"*Positive Percentage:* {result['Positive Percentage']:.2f}%")
            st.write(f"*Negative Percentage:* {result['Negative Percentage']:.2f}%")
            st.write(f"*Neutral Percentage:* {result['Neutral Percentage']:.2f}%")
            st.write(f"*Overall Sentiment:* {result['Overall Sentiment'].capitalize()}")
        except Exception as e:
            st.error(f"An error occurred during sentiment analysis: {e}")
    else:
        st.warning("Please enter a statement to analyze.")
