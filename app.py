import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import xml.etree.ElementTree as ET

# --- Helper Functions ---

@st.cache_data
def load_data():
    """
    Downloads and loads the MedQuAD dataset from GitHub, correctly parsing XML files.
    """
    status = st.empty()
    status.info("Downloading and processing medical data from MedQuAD...")
    data = []
    # These are the correct URLs to the MedQuAD XML files, as you pointed out.
    xml_urls = [
        'https://raw.githubusercontent.com/abachaa/MedQuAD/master/10_MPlus_ADAM_QA/10_MPlus_ADAM_QA.xml',
        'https://raw.githubusercontent.com/abachaa/MedQuAD/master/11_MPlusDrugs_QA/11_MPlusDrugs_QA.xml',
        'https://raw.githubusercontent.com/abachaa/MedQuAD/master/4_MPlus_Health_Topics_QA/4_MPlus_Health_Topics_QA.xml',
    ]

    for url in xml_urls:
        try:
            response = requests.get(url)
            response.raise_for_status()  # Check for HTTP errors
            
            # Use ElementTree to parse the XML content
            root = ET.fromstring(response.content)
            for qa_pair in root.findall('.//QAPair'):
                question = qa_pair.find('Question').text
                answer = qa_pair.find('Answer').text
                if question and answer:
                    data.append({
                        "question": question,
                        "answer": answer
                    })

        except requests.exceptions.RequestException as e:
            status.error(f"Error downloading data from {url}: {e}")
            return pd.DataFrame() 
        except ET.ParseError as e:
            status.error(f"Error parsing XML from {url}: {e}")
            return pd.DataFrame()

    df = pd.DataFrame(data)
    df.dropna(subset=['question', 'answer'], inplace=True)
    status.success("Data loaded successfully!")
    return df

def find_best_answer(question, df, vectorizer, tfidf_matrix):
    """
    Uses TF-IDF and cosine similarity to find the most relevant answer.
    """
    try:
        query_vec = vectorizer.transform([question])
    except ValueError:
        return "Sorry, I can't find an answer for that. Please try a different query."

    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    most_similar_index = cosine_similarities.argmax()

    if cosine_similarities[most_similar_index] > 0.1:
        return df.iloc[most_similar_index]['answer']
    else:
        return "I'm sorry, I couldn't find a good answer for that question. Can you please rephrase?"

def recognize_medical_entities(text):
    """
    Identifies basic medical entities in a given text using a simple dictionary lookup.
    """
    entities = []
    medical_terms = {
        'symptoms': ['fever', 'headache', 'cough', 'nausea', 'fatigue', 'dizziness'],
        'diseases': ['diabetes', 'hypertension', 'influenza', 'asthma', 'cancer'],
        'drugs': ['ibuprofen', 'acetaminophen', 'aspirin', 'penicillin', 'lipitor']
    }

    for term_type, terms in medical_terms.items():
        for term in terms:
            if re.search(r'\b' + re.escape(term) + r'\b', text.lower()):
                entities.append(term)
    
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

    # Load data
    df = load_data()

    if df.empty:
        st.error("Failed to load data. The chatbot cannot function.")
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
                entities = recognize_medical_entities(user_question)
                if entities:
                    st.subheader("Medical Entities Found:")
                    st.write(", ".join(entities))
        else:
            st.warning("Please enter a question to get an answer.")

# Run the app
if __name__ == "__main__":
    main()
