import random
import matplotlib
matplotlib.use('Agg')  # Configure non-interactive backend
import matplotlib.pyplot as plt
import networkx as nx
import math
import operator
import re
from flask import Flask, jsonify, render_template, request
from tfidf_sum import summarize_text, process_claims_v10, extract_dependencies
import os
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import json
# from markupsafe import escape
# Download necessary NLTK data sets
nltk.download('punkt')  # Example: Download the Punkt tokenizer models
nltk.download('averaged_perceptron_tagger')  # Example: POS tagger
nltk.download('wordnet')  # Example: WordNet Lemmatizer
nltk.download('stopwords') 

app = Flask(__name__)

sample_claims = []
def setup_nltk():
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('stopwords')

# Initialize Lemmatizer
wordlemmatizer = WordNetLemmatizer()
Stopwords = set(stopwords.words('english'))


def lemmatize_words(words):
    lemmatized_words = [wordlemmatizer.lemmatize(word) for word in words]
    return lemmatized_words

def remove_special_characters(text):
    regex = r'[^a-zA-Z0-9\s]'
    text = re.sub(regex, '', text)
    return text

def freq(words):
    words = [word.lower() for word in words]
    dict_freq = {}
    words_unique = set(words)
    for word in words_unique:
        dict_freq[word] = words.count(word)
    return dict_freq

def pos_tagging(text):
    pos_tag = nltk.pos_tag(text.split())
    pos_tagged_noun_verb = [word for word, tag in pos_tag if tag.startswith('NN') or tag.startswith('VB')]
    return pos_tagged_noun_verb

def tf_score(word, sentence):
    len_sentence = len(sentence.split())
    word_frequency_in_sentence = sentence.split().count(word)
    tf = word_frequency_in_sentence / len_sentence
    return tf

def idf_score(no_of_sentences, word, sentences):
    no_of_sentence_containing_word = sum(word in sentence for sentence in sentences)
    if no_of_sentence_containing_word == 0:
        return 0
    idf = math.log10(no_of_sentences / no_of_sentence_containing_word)
    return idf

def tf_idf_score(tf, idf):
    return tf * idf

def word_tfidf(dict_freq, word, sentences, sentence):
    tf = tf_score(word, sentence)
    idf = idf_score(len(sentences), word, sentences)
    tf_idf = tf_idf_score(tf, idf)
    return tf_idf

def sentence_importance(sentence, dict_freq, sentences):
    sentence_score = 0
    sentence = remove_special_characters(str(sentence))
    sentence = re.sub(r'\d+', '', sentence)
    pos_tagged_sentence = pos_tagging(sentence)
    for word in pos_tagged_sentence:
        if word.lower() not in Stopwords:
            word = word.lower()
            word = wordlemmatizer.lemmatize(word)
            sentence_score += word_tfidf(dict_freq, word, sentences, sentence)
    return sentence_score

def summarize_text(input_text, input_user=10):
    tokenized_sentence = sent_tokenize(input_text)
    text = remove_special_characters(str(input_text))
    text = re.sub(r'\d+', '', text)
    tokenized_words_with_stopwords = word_tokenize(text)
    tokenized_words = [word for word in tokenized_words_with_stopwords if word.lower() not in Stopwords]
    tokenized_words = [word for word in tokenized_words if len(word) > 1]
    tokenized_words = [word.lower() for word in tokenized_words]
    tokenized_words = lemmatize_words(tokenized_words)
    word_freq = freq(tokenized_words)

    no_of_sentences = int((input_user * len(tokenized_sentence)) / 100)
    sentence_with_importance = {}
    for index, sent in enumerate(tokenized_sentence):
        sentenceimp = sentence_importance(sent, word_freq, tokenized_sentence)
        sentence_with_importance[index + 1] = sentenceimp

    sentence_with_importance = sorted(sentence_with_importance.items(), key=operator.itemgetter(1), reverse=True)
    cnt = 0
    summary = []
    for word_prob in sentence_with_importance:
        if cnt < no_of_sentences:
            index = word_prob[0]
            summary.append(tokenized_sentence[index - 1])
            cnt += 1
        else:
            break

    readymade_summary = " ".join(summary)
    return readymade_summary

def create_dependency_graph(claims, dependencies, file_path='static/dependency_graph.png'):
    G = nx.DiGraph()
    node_colors = []

    for claim_number in dependencies.keys():
        G.add_node(claim_number)
        if "(canceled)" in claims[claim_number - 1]:
            node_colors.append('red')
        else:
            node_colors.append('lightgreen')

        for ref in dependencies[claim_number]:
            G.add_edge(claim_number, ref)

    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_color=node_colors, node_size=200, edge_color='black', linewidths=1, font_size=10)
    plt.title("Patent Claim Dependency Graph")

    # Create the 'static' directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save the graph image
    plt.savefig(file_path)
    plt.close()

def load_sample_claims():
    with open('templates/claims_samples.json', 'r') as file:
        global sample_claims
        sample_claims = json.load(file)

load_sample_claims()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_random_claim', methods=['GET'])
def get_random_claim():
    random_claim = random.choice(sample_claims)  # Randomly selects one dictionary from the list
    return jsonify(random_claim)

@app.route('/summarize', methods=['POST'])
def summarize():
    if request.method == 'POST':
        input_text = request.form['inputText']
        summary = summarize_text(input_text)

        # Process claims and extract dependencies
        combined_sentences = process_claims_v10(input_text)
        dependencies = extract_dependencies(combined_sentences)

        # Create dependency graph
        create_dependency_graph(combined_sentences, dependencies)

        # Calculate word count for each sentence and total word count
        sentences_with_count = [(sentence, len(sentence.split())) for sentence in combined_sentences]
        total_word_count = sum(word_count for _, word_count in sentences_with_count)
        summary_word_count = len(summary.split())

        return render_template('summary.html', 
                               original_text=sentences_with_count, 
                               total_word_count=total_word_count,
                               summary=summary, 
                               summary_word_count=summary_word_count,
                               graph_image='dependency_graph.png')
    return render_template('index.html')

if __name__ == "__main__":
    setup_nltk()
    app.run(debug=True)