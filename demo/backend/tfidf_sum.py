import re
import math
import operator
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Ensure necessary NLTK datasets are downloaded
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

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

def summarize_text(input_text, input_user=100):
    tokenized_sentence = sent_tokenize(input_text)
    text = remove_special_characters(str(input_text))
    text = re.sub(r'\d+', '', text)
    tokenized_words_with_stopwords = word_tokenize(text)
    tokenized_words = [word for word in tokenized_words_with_stopwords if word.lower() not in Stopwords]
    tokenized_words = [word for word in tokenized_words if len(word) > 1]
    tokenized_words = [word.lower() for word in tokenized_words]
    tokenized_words = lemmatize_words(tokenized_words)
    word_freq = freq(tokenized_words)

    no_of_sentences = int((input_user * len(tokenized_sentence)) / 40)
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
