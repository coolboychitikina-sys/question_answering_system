import nltk
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ---------------------------------------
# QUESTION ANSWERING SYSTEM
# ---------------------------------------

# NLTK tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# Question and Answer Dataset
qa_data = {
    "what is natural language processing":
        "Natural Language Processing is a field of AI that helps computers understand human language.",

    "what is nlp":
        "NLP stands for Natural Language Processing.",

    "what is artificial intelligence":
        "Artificial Intelligence is the ability of machines to perform tasks that normally require human intelligence.",

    "what is machine learning":
        "Machine Learning is a branch of AI that enables computers to learn from data.",

    "what is deep learning":
        "Deep Learning is a machine learning technique that uses neural networks with multiple layers.",

    "what is tokenization":
        "Tokenization is the process of splitting text into smaller units called tokens.",

    "what is text preprocessing":
        "Text preprocessing is the process of cleaning and preparing text for NLP analysis.",

    "what are stopwords":
        "Stopwords are common words such as the, is, a, and, and are often removed during text preprocessing.",

    "what is stemming":
        "Stemming is the process of reducing words to their basic stem or root form.",

    "what is lemmatization":
        "Lemmatization converts words into their meaningful base or dictionary form.",

    "what is pos tagging":
        "POS tagging identifies the grammatical category of each word, such as noun, verb, adjective, or adverb.",

    "what is nltk":
        "NLTK stands for Natural Language Toolkit. It is a Python library used for Natural Language Processing.",

    "what is information retrieval":
        "Information retrieval is the process of finding relevant information from a collection of data or documents.",

    "what is a question answering system":
        "A Question Answering System accepts questions in natural language and provides relevant answers.",

    "how does a question answering system work":
        "A Question Answering System accepts a question, preprocesses it, analyzes it, searches the available information, selects a suitable answer, and displays the response.",

    "what is a dataset":
        "A dataset is a collection of organized data used for processing, analysis, testing, or training.",

    "what is a knowledge base":
        "A knowledge base is a structured collection of information that can be searched to answer questions.",

    "what is text classification":
        "Text classification is the process of assigning text to predefined categories.",

    "what is a chatbot":
        "A chatbot is a software application that interacts with users through natural language.",

    "what is conversational ai":
        "Conversational AI uses artificial intelligence and NLP to communicate with users.",

    "what is semantic search":
        "Semantic search understands the meaning of a query and retrieves information based on that meaning.",

    "what is keyword extraction":
        "Keyword extraction identifies important words or phrases from a text.",

    "what is question analysis":
        "Question analysis examines the user's question to identify important terms and features.",

    "what is answer selection":
        "Answer selection is the process of choosing the most relevant answer from available information.",

    "what is response generation":
        "Response generation is the process of producing and displaying an appropriate answer.",

    "what are the applications of nlp":
        "NLP is used in question answering systems, chatbots, translation, sentiment analysis, search systems, and virtual assistants.",

    "what is natural language understanding":
        "Natural Language Understanding enables computers to understand the meaning and intent of human language.",

    "what is natural language generation":
        "Natural Language Generation is the process of producing human-readable language from data or system information.",

    "what are the benefits of a question answering system":
        "A Question Answering System provides quick access to information and reduces manual searching."
}


# ---------------------------------------
# TEXT PREPROCESSING
# ---------------------------------------

def preprocess(text):

    # Convert to lowercase
    text = text.lower()

    # Remove special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords and lemmatize
    processed_words = []

    for word in tokens:
        if word not in stop_words:
            word = lemmatizer.lemmatize(word)
            processed_words.append(word)

    return processed_words


# ---------------------------------------
# FIND ANSWER
# ---------------------------------------

def find_answer(question):

    question_words = set(preprocess(question))

    best_answer = None
    best_score = 0

    for stored_question, answer in qa_data.items():

        stored_words = set(preprocess(stored_question))

        common_words = question_words.intersection(stored_words)

        score = len(common_words)

        if score > best_score:
            best_score = score
            best_answer = answer

    if best_answer is not None:
        return best_answer

    return "Sorry, I could not find a suitable answer."


# ---------------------------------------
# MAIN PROGRAM
# ---------------------------------------

print()
print("==========================================")
print("       QUESTION ANSWERING SYSTEM")
print("==========================================")

print()
print("System is ready!")

question = input("Enter your question: ")

print()
print("Original Question:")
print(question)

print()
print("Tokens after preprocessing:")
print(preprocess(question))

print()
print("Answer:")
print(find_answer(question))

print()
print("==========================================")