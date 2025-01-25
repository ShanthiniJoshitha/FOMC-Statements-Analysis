from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import streamlit as st

# Define the model name
model_name = "yiyanghkust/finbert"  # FinBERT model for financial sentiment analysis

# Load the tokenizer with error handling
try:
    # Attempt to load the fast tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
except Exception as e:
    # If fast tokenizer fails, use the slow tokenizer
    st.warning(f"Fast tokenizer failed to load: {e}. Falling back to the slow tokenizer.")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    except Exception as e:
        st.error(f"Error loading slow tokenizer: {e}")
        st.stop()

# Load the model with error handling
try:
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
except ImportError as e:
    st.error(f"Backend dependency issue: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Define the sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Function to process the input statement
def process_statement(statement):
    lines = statement.split('.')  # Split statement into sentences
    sentiments = []

    for line in lines:
        line = line.strip()
        if line:  # Ignore empty lines
            try:
                # Analyze sentiment for each line
                result = sentiment_analyzer(line)
                sentiments.append(result[0]['label'].lower())
            except Exception as e:
                st.warning(f"Error analyzing sentiment for the line: {line}. Error: {e}")
                sentiments.append("neutral")  # Fallback to neutral if analysis fails
    
    positive_count = sentiments.count('positive')
    negative_count = sentiments.count('negative')
    neutral_count = sentiments.count('neutral')
    total = len(sentiments)

    # Calculate percentages
    positive_percentage = (positive_count / total) * 100 if total > 0 else 0
    negative_percentage = (negative_count / total) * 100 if total > 0 else 0
    neutral_percentage = (neutral_count / total) * 100 if total > 0 else 0

    # Determine the overall sentiment
    overall_sentiment = max(sentiments, key=sentiments.count) if sentiments else "neutral"

    return {
        "Positive Percentage": positive_percentage,
        "Negative Percentage": negative_percentage,
        "Neutral Percentage": neutral_percentage,
        "Overall Sentiment": overall_sentiment
    }

# Streamlit App UI
st.title("Sentiment Analysis App")
st.write("Analyze the sentiment of your statements using a fine-tuned model.")

# Input text area for user input
user_input = st.text_area("Enter your statement below:", placeholder="Type here...")

# Button to analyze sentiment
if st.button("Analyze Sentiment"):
    if user_input.strip():
        try:
            # Process the input statement and display results
            result = process_statement(user_input)
            st.subheader("Sentiment Analysis Results:")
            st.write(f"**Positive Percentage:** {result['Positive Percentage']:.2f}%")
            st.write(f"**Negative Percentage:** {result['Negative Percentage']:.2f}%")
            st.write(f"**Neutral Percentage:** {result['Neutral Percentage']:.2f}%")
            st.write(f"**Overall Sentiment:** {result['Overall Sentiment'].capitalize()}")
        except Exception as e:
            st.error(f"An error occurred during sentiment analysis: {e}")
    else:
        st.warning("Please enter a statement to analyze.")
