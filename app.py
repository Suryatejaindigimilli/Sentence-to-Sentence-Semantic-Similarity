from flask import Flask, request, render_template, flash
from transformers import pipeline, AutoModel, AutoTokenizer
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import torch
import logging
logging.basicConfig(level=logging.DEBUG)


# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flashing messages

# Load Hugging Face Question-Answering pipeline
qa_model = pipeline("question-answering")

# Load Sentence-BERT model and tokenizer for similarity
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
model = AutoModel.from_pretrained(model_name, local_files_only=True)


# Function to encode sentence into embeddings
def encode_sentence(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

# Function to calculate similarity between all pairs of sentences
def calculate_similarities(sentences):
    embeddings = [encode_sentence(sentence).detach().numpy() for sentence in sentences]
    pairs = list(itertools.combinations(range(len(sentences)), 2))
    similarities = []
    
    for i, j in pairs:
        similarity_score = cosine_similarity(embeddings[i].reshape(1, -1), embeddings[j].reshape(1, -1))[0][0]
        similarities.append({
            'pair': (sentences[i], sentences[j]),
            'score': similarity_score
        })
    
    return similarities

# Function for semantic search between a question and multiple contexts
def semantic_search(question, contexts):
    answers = []
    
    for context in contexts:
        # Get the answer from the Hugging Face model for the current context
        result = qa_model(question=question, context=context)
        answers.append({
            'context': context,
            'answer': result['answer'],
            'score': round(result['score'], 2)
        })
    
    return answers

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/similarity', methods=['GET', 'POST'])
def similarity():
    scores = []

    if request.method == 'POST':
        sentences = request.form.getlist('sentences')  # Retrieve list of sentences from the form
        if len(sentences) > 1:
            scores = calculate_similarities(sentences)
        else:
            flash("Please enter at least two sentences to calculate similarity.", "error")

    return render_template('similarity.html', scores=scores)

@app.route('/semantic', methods=['GET', 'POST'])
def semantic():
    answers = []
    question = None  # Initialize question to avoid UnboundLocalError

    if request.method == 'POST':
        question = request.form.get('question')
        contexts = request.form.getlist('contexts')  # Get multiple contexts from the form

        if question and contexts:
            # Perform semantic search and get the answers
            answers = semantic_search(question, contexts)
        elif not question:
            flash("Please enter a question.", "error")
        elif not contexts:
            flash("Please provide at least one context.", "error")

    return render_template('semantic.html', answers=answers, question=question)

if __name__ == '__main__':
    app.run(debug=True)
