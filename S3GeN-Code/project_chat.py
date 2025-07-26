import re
import random
import time
import math
import os
import json
import sqlite3
import asyncio
from collections import defaultdict, Counter
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download

# HF 허깅페이스 캐시 설정
os.environ["HF_HOME"] = "/tmp/hf_cache"
hf_token = os.getenv("HF_TOKEN")
path = hf_hub_download(
    repo_id="Yuchan5386/S3GeN",
    filename="ngram_model.db",
    repo_type="dataset",
    token=hf_token
)

# FastAPI 초기화
app = FastAPI()
origins = ["https://insect5386.github.io"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 토크나이저
def simple_tokenizer(text):
    return re.findall(r'[가-힣]+|[,.!?]', text)

# 온도 적용 함수
def apply_temperature(probs, temperature):
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if temperature == 1.0:
        return probs
    log_probs = [math.log(p) if p > 0 else -1e10 for p in probs]
    tempered = [math.exp(lp / temperature) for lp in log_probs]
    s = sum(tempered)
    return [t / s for t in tempered]

# SQLite 기반 생성기 클래스
class SQLiteStatSeqGenerator:
    def __init__(self, db_path=path):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._init_db()

    def _init_db(self):
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS next_word (
            w1 TEXT, w2 TEXT, w3 TEXT, next_word TEXT, count INTEGER,
            PRIMARY KEY (w1, w2, w3, next_word)
        )""")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_context ON next_word (w1, w2, w3)")

    def train(self, qa_pairs):
        insert_query = """
        INSERT INTO next_word (w1, w2, w3, next_word, count)
        VALUES (?, ?, ?, ?, 1)
        ON CONFLICT(w1, w2, w3, next_word) DO UPDATE SET count = count + 1
        """
        for q, a in qa_pairs:
            tokens = ['<BOS>', '<BOS>'] + simple_tokenizer(a) + ['<EOS>']
            for i in range(len(tokens) - 3):
                w1, w2, w3 = tokens[i], tokens[i+1], tokens[i+2]
                next_word = tokens[i+3]
                self.cursor.execute(insert_query, (w1, w2, w3, next_word))
        self.conn.commit()

    def get_next_word_probs(self, context):
        self.cursor.execute(
            "SELECT next_word, count FROM next_word WHERE w1=? AND w2=? AND w3=?",
            context
        )
        results = self.cursor.fetchall()
        if not results:
            return None
        words, counts = zip(*results)
        total = sum(counts)
        probs = [c / total for c in counts]
        return words, probs

    def generate(self, start_word, max_len=30, temperature=1.0):
        context = ('<BOS>', '<BOS>', start_word)
        for _ in range(max_len):
            if context[-1] == '<EOS>':
                break
            yield context[-1]
            result = self.get_next_word_probs(context)
            if not result:
                break
            words, base_probs = result
            tempered_probs = apply_temperature(base_probs, temperature)
            next_word = random.choices(words, tempered_probs)[0]
            context = (context[1], context[2], next_word)
        yield '<EOS>'

# QA 데이터 로드 함수
def load_qa_pairs_from_jsonl(path, max_pairs=200000000):
    qa_pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            q = obj.get("question", "").strip()
            a = obj.get("answer", "").strip()
            if q and a:
                qa_pairs.append((q, a))
                if len(qa_pairs) >= max_pairs:
                    break
    return qa_pairs

# 훈련 + 모델 객체 생성
if not os.path.exists(path):
    print("🔨 SQLite 모델 생성 중...")
    qa_pairs = load_qa_pairs_from_jsonl(path)
    gen = SQLiteStatSeqGenerator()
    gen.train(qa_pairs)
    del qa_pairs
    print("✅ SQLite 모델 저장 완료!")
else:
    gen = SQLiteStatSeqGenerator()

# API 라우트
@app.get('/chat')
async def chat_sse(message: str):
    async def event_generator():
        input_tokens = simple_tokenizer(message)
        start_word = input_tokens[-1] if input_tokens else random.choice(['안녕', '나', '오늘'])
        for partial in gen.generate(start_word, max_len=30, temperature=1.0):
            yield f"data: {partial}\n\n"
            await asyncio.sleep(0.1)
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
