"""
Kode.py
"""

import random
import re
import time
import numpy as np
import pandas as pd
import os
import requests
import gradio as gr
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 파일 없으면 다운로드
def download_if_not_exists(filename, url):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        r = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(r.content)
        print("Download complete.")

download_if_not_exists("Code.parquet", "https://huggingface.co/Yuchan5386/Kode/resolve/main/Code.parquet?download=true")
download_if_not_exists("vectorizer.joblib", "https://huggingface.co/Yuchan5386/Kode/resolve/main/vectorizer.joblib?download=true")
download_if_not_exists("response_vecs.npy", "https://huggingface.co/Yuchan5386/Kode/resolve/main/response_vecs.npy?download=true")
download_if_not_exists("vocab_vectors.npy", "https://huggingface.co/Yuchan5386/Kode/resolve/main/vocab_vectors.npy?download=true")
download_if_not_exists("vt_matrix.npy", "https://huggingface.co/Yuchan5386/Kode/resolve/main/vt_matrix.npy?download=true")

# ===== 유틸 함수들 =====
def remove_invalid_unicode(text):
    return re.sub(r'[\ud800-\udfff]', '', text)

def tokenize(text):
    return re.findall(r'\w+', text.lower())

def softmax(x, temperature=1.0):
    x = np.array(x)
    x = x - np.max(x)
    exp_x = np.exp(x / temperature)
    return exp_x / exp_x.sum()

def synonym_replace(text):
    replacements = {
        "좋아요": "괜찮아요",
        "도와드릴게요": "도와줄게요",
        "문제없어요": "걱정마세요",
        "할 수 있어요": "가능해요"
    }
    for word, synonym in replacements.items():
        text = text.replace(word, synonym)
    return text

def shuffle_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    if len(sentences) > 1:
        random.shuffle(sentences)
    return ' '.join(sentences)

def casual_tone(text):
    replacements = {
        "입니다": "예요",
        "할 수 있습니다": "할 수 있어요",
        "도와드리겠습니다": "도와줄게요",
        "감사합니다": "고마워요",
    }
    for formal, casual in replacements.items():
        text = text.replace(formal, casual)
    return text

def drop_redundant(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    if len(sentences) > 1:
        keep = random.sample(sentences, k=max(1, len(sentences)//2))
        return ' '.join(keep)
    return text

def reverse_phrase(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return ' '.join(sentences[::-1])

def random_cut(text):
    words = text.split()
    if len(words) > 5:
        cut_point = random.randint(5, len(words))
        return ' '.join(words[:cut_point])
    return text

def similar_word_replace(text, vocab_words, vocab_vectors, top_k=5):
    words = text.split()
    if not words:
        return text
    candidate = random.choice(words)
    if candidate not in vocab_words:
        return text

    idx = vocab_words.index(candidate)
    word_vec = vocab_vectors[idx].reshape(1, -1)
    sims = cosine_similarity(word_vec, vocab_vectors).flatten()
    sims[idx] = -1
    top_indices = sims.argsort()[-top_k:]
    similar_words = [vocab_words[i] for i in top_indices if sims[i] > 0]

    if not similar_words:
        return text

    replacement = random.choice(similar_words)
    new_words = [replacement if w == candidate else w for w in words]
    return ' '.join(new_words)

def remix_response(text, vocab_words=None, vocab_vectors=None):
    funcs = [
        synonym_replace,
        shuffle_sentences,
        casual_tone,
        drop_redundant,
        reverse_phrase,
        random_cut
    ]
    if vocab_words is not None and vocab_vectors is not None:
        funcs.append(lambda txt: similar_word_replace(txt, vocab_words, vocab_vectors))

    random.shuffle(funcs)
    selected_funcs = funcs[:random.randint(2, 5)]
    for func in selected_funcs:
        text = func(text)
    return text

# ===== 대화 로더 =====
from tqdm import tqdm

def load_conversations(parquet_path="Code.parquet"):
    df = pd.read_parquet(parquet_path)
    convs = []
    for conversations in tqdm(df["conversations"], desc="Loading conversations"):
        for i in range(0, len(conversations) - 1, 2):
            item1, item2 = conversations[i], conversations[i + 1]
            if item1.get("from") == "human" and item2.get("from") == "gpt":
                convs.append((item1.get("value", ""), item2.get("value", "")))
    return convs

# ===== ResponseGenerator 복원 클래스 =====
class ResponseGenerator:
    def __init__(self, convs, temperature=0.8):
        self.responses = [r for _, r in convs]
        self.vectorizer = joblib.load("vectorizer.joblib")
        self.response_vecs = np.load("response_vecs.npy")
        self.vocab_vectors = np.load("vocab_vectors.npy")
        self.VT = np.load("vt_matrix.npy")
        self.vocab_words = self.vectorizer.get_feature_names_out().tolist()
        self.temperature = temperature

    def generate_stream(self, user_input, top_k=5, delay=0.02):
        query_tfidf = self.vectorizer.transform([user_input])
        query_dense = query_tfidf @ self.VT.T
        query_vec = query_dense.astype(np.float32)

        sims = np.dot(self.response_vecs, query_vec.T).flatten()

        top_indices = np.argpartition(-sims, top_k)[:top_k]
        top_sims = sims[top_indices]

        temp = random.uniform(0.5, 1.2)
        probs = softmax(top_sims * 10, temperature=temp)

        chosen_idxs = np.random.choice(top_indices, size=2, replace=False, p=probs / probs.sum())
        mixed_resp = ' '.join([self.responses[i] for i in chosen_idxs])

        chosen_resp = remix_response(mixed_resp, self.vocab_words, self.vocab_vectors)

        accumulated = ""
        for ch in chosen_resp:
            accumulated += ch
            yield accumulated
            time.sleep(delay)

# ===== 메인 실행 =====
convs = load_conversations()
print(f"총 대화 페어 수: {len(convs)}")
generator = ResponseGenerator(convs, temperature=0.7)

def respond_stream(user_input):
    for partial in generator.generate_stream(user_input):
        cleaned = remove_invalid_unicode(partial)
        yield cleaned

def chat_fn(message, chat_history):
    for partial in respond_stream(message):
        yield partial

iface = gr.ChatInterface(fn=chat_fn)
iface.launch()
