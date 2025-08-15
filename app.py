import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- Helper Functions ---

@st.cache_data
def load_data():
    """
    Creates a small, hard-coded subset of the MedQuAD dataset.
    This bypasses all network requests and ensures the app can always run.
    """
    st.info("Using embedded medical data...")
    data = [
        {"question": "What is diabetes?", "answer": "Diabetes is a chronic, metabolic disease characterized by elevated levels of blood glucose (or blood sugar), which leads over time to serious damage to the heart, blood vessels, eyes, kidneys and nerves."},
        {"question": "Symptoms of influenza?", "answer": "The most common symptoms of influenza are fever, cough, sore throat, and muscle aches. It is a viral infection that attacks your respiratory system."},
        {"question": "How to treat a headache?", "answer": "Headaches can often be treated with over-the-counter pain relievers like ibuprofen or acetaminophen. Rest and staying hydrated can also help."},
        {"question": "What is hypertension?", "answer": "Hypertension, also known as high blood pressure, is a serious medical condition. It can be caused by various factors and is a major risk factor for cardiovascular disease."},
        {"question": "What is the function of penicillin?", "answer": "Penicillin is a group of antibiotics used to treat a wide range of bacterial infections. It works by interfering with the formation of the bacteria's cell wall."},
        {"question": "What is the cause of asthma?", "answer": "The exact cause of asthma is not known. It is believed to be caused by a combination of genetic and environmental factors. Common triggers include pollen, dust mites, mold, and smoke."},
        {"question": "What are the side effects of Ibuprofen?", "answer": "The most common side effects of ibuprofen are nausea, vomiting, stomach pain, and heartburn. It can also increase the risk of heart attack or stroke."},
        {"question": "What is cancer?", "answer": "Cancer is a disease caused by an uncontrolled division of abnormal cells in a part of the body. It can spread to other parts of the body."}
    ]
    df = pd.DataFrame(data)
    st.success("Data loaded successfully!")
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

    st.markdown("Ask a medical question about diseases or drugs, and I'll do my best to answer it.")

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
