import streamlit as st
import pandas as pd
import requests
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import subprocess

# --- Helper Functions ---

@st.cache_data
def load_data():
    """
    Downloads and loads a subset of the MedQuAD dataset from GitHub.
    This function caches the data to avoid re-downloading on every run.
    """
    status = st.empty()
    status.info("Downloading and processing medical data...")
    data = []
    # Using a subset of the data for faster deployment
    json_urls = [
        'https://raw.githubusercontent.com/abachaa/MedQuAD/master/4_QA_json/dofus_disease_qa_pairs.json',
        'https://raw.githubusercontent.com/abachaa/MedQuAD/master/4_QA_json/drugbank_drug_qa_pairs.json',
        'https://raw.githubusercontent.com/abachaa/MedQuAD/master/4_QA_json/webmd_qa_pairs.json',
    ]

    for url in json_urls:
        try:
            response = requests.get(url)
            response.raise_for_status()  # Check for HTTP errors
            qa_pairs = response.json()
            for item in qa_pairs:
                data.append({
                    "question": item.get("question"),
                    "answer": item.get("answer")
                })
        except requests.exceptions.RequestException as e:
            status.error(f"Error downloading data from {url}: {e}")
            return pd.DataFrame() # Return an empty DataFrame on error

    df = pd.DataFrame(data)
    df.dropna(subset=['question', 'answer'], inplace=True)
    status.success("Data loaded successfully!")
    return df

@st.cache_resource
def load_spacy_model():
    """
    Loads the scispaCy model for medical entity recognition.
    This function caches the model to avoid reloading it on every run.
    """
    status = st.empty()
    status.info("Loading scispaCy model for entity recognition...")
    try:
        # The model is now installed via install.sh so we can simply load it
        nlp = spacy.load("en_core_sci_sm")
        status.success("scispaCy model loaded!")
        return nlp
    except OSError:
        status.error(
            """
            The required spaCy model could not be found. This often happens because the model
            was not correctly installed during deployment.

            Please ensure your Streamlit Cloud app is configured to run `install.sh`
            and that the `install.sh` file contains the correct installation commands.
            """
        )
        return None
    except Exception as e:
        status.error(
            f"An unexpected error occurred while loading the spacy model. Please check the logs for more details. Error: {e}"
        )
        return None

def find_best_answer(question, df, vectorizer, tfidf_matrix):
    """
    Uses TF-IDF and cosine similarity to find the most relevant answer.
    """
    # Transform the user's question into a TF-IDF vector
    try:
        query_vec = vectorizer.transform([question])
    except ValueError:
        return "Sorry, I can't find an answer for that. Please try a different query."

    # Compute cosine similarity between the query and all answers
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Get the index of the most similar answer
    most_similar_index = cosine_similarities.argmax()

    # Check if the similarity score is high enough
    if cosine_similarities[most_similar_index] > 0.1: # A simple threshold
        return df.iloc[most_similar_index]['answer']
    else:
        return "I'm sorry, I couldn't find a good answer for that question. Can you please rephrase?"

def recognize_medical_entities(text, nlp):
    """
    Identifies medical entities in a given text using the loaded scispaCy model.
    """
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    return entities

# --- Main Streamlit Application ---

def main():
    st.set_page_config(page_title="Medical Chatbot", page_icon="💊")
    st.title("👨‍⚕️ Medical Q&A Chatbot")
    st.markdown(
        """
        <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 12px;
            padding: 10px 24px;
            font-size: 16px;
        }
        .stTextInput>div>div>input {
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("Ask a medical question about diseases or drugs, and I'll do my best to answer it based on the MedQuAD dataset.")

    # Load data and the scispaCy model
    df = load_data()
    nlp = load_spacy_model()

    if df.empty or nlp is None:
        st.stop()

    # Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['answer'])

    # Get user input
    user_question = st.text_input("Enter your question here:")

    if st.button("Get Answer"):
        if user_question:
            with st.spinner("Finding the best answer..."):
                # Find the most relevant answer
                best_answer = find_best_answer(user_question, df, vectorizer, tfidf_matrix)

                st.subheader("Answer:")
                st.write(best_answer)

                # Recognize and display medical entities
                if nlp:
                    entities = recognize_medical_entities(user_question, nlp)
                    if entities:
                        st.subheader("Medical Entities Found:")
                        st.write(", ".join(entities))
        else:
            st.warning("Please enter a question to get an answer.")

# Run the app
if __name__ == "__main__":
    main()
