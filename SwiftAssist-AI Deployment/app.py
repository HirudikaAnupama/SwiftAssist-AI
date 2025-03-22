import os
import torch
import joblib
import numpy as np
import pandas as pd
import re
import spacy
from spellchecker import SpellChecker
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    AutoModelForSequenceClassification,
    AutoTokenizer
)
from flask import Flask, request, jsonify, render_template

# LangChain imports
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI

app = Flask(__name__)

################################################################################
# Environment Setup
################################################################################

# Replace with your actual Google Generative AI key
os.environ["GOOGLE_API_KEY"] = "AIzaSyD7f8_bcVqHA_W6y8V_uvNWcwP_YSmoocQ"

# Initialize the Google Generative AI LLM
llm = GoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    temperature=0.7
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################################################################
# Load Models & Data
################################################################################

# 1. Load LabelEncoder for Intent Recognition
label_encoder = joblib.load(r"C:\Users\LENOVO\Desktop\Projects\SwiftAssist-AI\SwiftAssist-AI\models\Model training\Intent recognition\label_encoder.pkl")

# 2. Load Fine-Tuned BERT Model & Tokenizer for Intent Recognition
intent_model = BertForSequenceClassification.from_pretrained(r"C:\Users\LENOVO\Desktop\Projects\SwiftAssist-AI\SwiftAssist-AI\intent_recognition_bert")
intent_tokenizer = BertTokenizer.from_pretrained(r"C:\Users\LENOVO\Desktop\Projects\SwiftAssist-AI\SwiftAssist-AI\intent_recognition_bert")

# 3. Load Sentiment Analysis Model & Tokenizer
sentiment_model_path = r"C:\Users\LENOVO\Desktop\Projects\SwiftAssist-AI\SwiftAssist-AI\contentsentiment_model"
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_path)
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_path)
sentiment_model.to(device)

# 4. Load SentenceTransformer for FAQ embeddings
faq_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 5. Load Preprocessed FAQ Data (with embeddings)
D_emb = pd.read_csv(r"C:\Users\LENOVO\Desktop\Projects\SwiftAssist-AI\SwiftAssist-AI\data\FAQ Answering\Preprocessed embedding\D_emb.csv")
D_emb["embedding"] = D_emb["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))

################################################################################
# Supporting Dictionaries & Lists
################################################################################

intent_label_mapping = {
    'playmusic': 'Play Music',
    'addtoplaylist': 'Add to Playlist',
    'ratebook': 'Rate Book',
    'searchscreeningevent': 'Search Screening Event',
    'bookrestaurant': 'Book Restaurant',
    'getweather': 'Get Weather',
    'searchcreativework': 'Search Creative Work',
    'greeting': 'Greeting',
    'greetingresponse': 'Greeting Response',
    'courtesygreeting': 'Courtesy Greeting',
    'courtesygreetingresponse': 'Courtesy Greeting Response',
    'currenthumanquery': 'Current Human Query',
    'namequery': 'Name Query',
    'realnamequery': 'Real Name Query',
    'timequery': 'Time Query',
    'thanks': 'Thanks',
    'nottalking2u': 'Not Talking to You',
    'understandquery': 'Understand Query',
    'shutup': 'Shut Up',
    'swearing': 'Swearing',
    'goodbye': 'Goodbye',
    'courtesygoodbye': 'Courtesy Goodbye',
    'whoami': 'Who Am I',
    'clever': 'Clever',
    'gossip': 'Gossip',
    'jokes': 'Jokes',
    'podbaydoor': 'Pod Bay Door',
    'podbaydoorresponse': 'Pod Bay Door Response',
    'selfaware': 'Self Aware',
    'cancelorder': 'Cancel Order',
    'changeorder': 'Change Order',
    'changeshippingaddress': 'Change Shipping Address',
    'checkcancellationfee': 'Check Cancellation Fee',
    'checkinvoice': 'Check Invoice',
    'checkpaymentmethods': 'Check Payment Methods',
    'checkrefundpolicy': 'Check Refund Policy',
    'complaint': 'Complaint',
    'contactcustomerservice': 'Contact Customer Service',
    'contacthumanagent': 'Contact Human Agent',
    'createaccount': 'Create Account',
    'deleteaccount': 'Delete Account',
    'deliveryoptions': 'Delivery Options',
    'deliveryperiod': 'Delivery Period',
    'editaccount': 'Edit Account',
    'getinvoice': 'Get Invoice',
    'getrefund': 'Get Refund',
    'newslettersubscription': 'Newsletter Subscription',
    'paymentissue': 'Payment Issue',
    'placeorder': 'Place Order',
    'recoverpassword': 'Recover Password',
    'registrationproblems': 'Registration Problems',
    'review': 'Review',
    'setupshippingaddress': 'Setup Shipping Address',
    'switchaccount': 'Switch Account',
    'trackorder': 'Track Order',
    'trackrefund': 'Track Refund'
}

faq_intents = [
    'getweather', 'searchcreativework', 'greeting', 'greetingresponse',
    'courtesygreeting', 'courtesygreetingresponse', 'currenthumanquery',
    'namequery', 'realnamequery', 'timequery', 'thanks', 'nottalking2u',
    'understandquery', 'shutup', 'swearing', 'goodbye', 'courtesygoodbye',
    'whoami', 'clever', 'gossip', 'jokes', 'podbaydoor', 'podbaydoorresponse',
    'selfaware',
]

non_faq_intents = [
    'playmusic', 'addtoplaylist', 'ratebook', 'searchscreeningevent',
    'bookrestaurant', 'cancelorder', 'changeorder', 'changeshippingaddress',
    'checkcancellationfee', 'checkinvoice', 'checkpaymentmethods',
    'checkrefundpolicy', 'complaint', 'contactcustomerservice',
    'contacthumanagent', 'createaccount', 'deleteaccount', 'deliveryoptions',
    'deliveryperiod', 'editaccount', 'getinvoice', 'getrefund',
    'newslettersubscription', 'paymentissue', 'placeorder', 'recoverpassword',
    'registrationproblems', 'review', 'setupshippingaddress', 'switchaccount',
    'trackorder', 'trackrefund'
]

################################################################################
# Core Functions
################################################################################

# 4.1 Intent Recognition
def predict_intent(text):
    inputs = intent_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=64)
    with torch.no_grad():
        outputs = intent_model(**inputs)
    predicted_label_idx = torch.argmax(outputs.logits, dim=1).item()
    intent_name = label_encoder.inverse_transform([predicted_label_idx])[0]
    readable_intent = intent_label_mapping.get(intent_name, intent_name)
    intent_type = "FAQ" if intent_name in faq_intents else "Non-FAQ"
    return readable_intent, intent_type

# 4.2 Sentiment Analysis
def predict_sentiment(text):
    encoding = sentiment_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=96
    ).to(device)
    with torch.no_grad():
        output = sentiment_model(**encoding)
        probs = torch.softmax(output.logits, dim=1)
        pred = torch.argmax(probs).item()
    return "Positive" if pred == 1 else "Negative"

# 4.3 FAQ Answering
nlp = spacy.load("en_core_web_sm")
spell = SpellChecker()

def clean_question(text):
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    words = text.split()
    corrected_words = [spell.correction(word) if spell.correction(word) is not None else word for word in words]
    text = ' '.join(corrected_words)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    return ' '.join(tokens)

def retrieve_answer(user_query, df, threshold=0.7):
    cleaned_user_query = clean_question(user_query)
    query_embedding = faq_encoder.encode([cleaned_user_query])
    similarities = cosine_similarity(query_embedding, np.stack(df["embedding"]))
    best_match_idx = np.argmax(similarities)
    best_score = similarities[0][best_match_idx]
    if best_score >= threshold:
        return df.iloc[best_match_idx]["cleaned_answer"]
    else:
        return "Sorry, I couldn't find a relevant answer."

# 4.4 AI Response Generation (for Non-FAQ)
ai_template = """
You are a helpful customer support chatbot. The user says: "{query}"

Please provide a concise and helpful response:
"""
ai_prompt = PromptTemplate(template=ai_template, input_variables=["query"])
ai_chain = LLMChain(llm=llm, prompt=ai_prompt)

def generate_ai_response(user_query):
    return ai_chain.run(query=user_query)

# 4.5 Tone Adjustment
tone_template = """
You are a helpful customer support chatbot.
The user asked: {query}
We have a raw response: {raw_answer}
The user's sentiment is: {sentiment}

Rewrite the raw response to match the user's sentiment:
- If sentiment is "Positive", use a cheerful and friendly tone.
- If sentiment is "Negative", use an empathetic and supportive tone.
- If sentiment is "Neutral", use a neutral and professional tone.

Final response:
"""
tone_prompt = PromptTemplate(
    template=tone_template,
    input_variables=["query", "raw_answer", "sentiment"]
)
tone_chain = LLMChain(llm=llm, prompt=tone_prompt)

def tone_adjust_response(user_query, raw_answer, sentiment):
    return tone_chain.run(query=user_query, raw_answer=raw_answer, sentiment=sentiment)

################################################################################
# Main End-to-End Function
################################################################################

def swiftassist_chatbot(user_query):
    predicted_intent, intent_type = predict_intent(user_query)
    sentiment = predict_sentiment(user_query)
    if intent_type == "FAQ":
        raw_answer = retrieve_answer(user_query, D_emb)
    else:
        raw_answer = generate_ai_response(user_query)
    if sentiment not in ["Positive", "Negative"]:
        sentiment = "Neutral"
    final_response = tone_adjust_response(user_query, raw_answer, sentiment)
    return {
        "Predicted Intent": predicted_intent,
        "Intent Type": intent_type,
        "Sentiment": sentiment,
        "Final Response": final_response.strip()
    }

################################################################################
# Flask Endpoints
################################################################################

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_query = data.get("query", "")
    if not user_query:
        return jsonify({"error": "Query is empty"}), 400
    result = swiftassist_chatbot(user_query)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
