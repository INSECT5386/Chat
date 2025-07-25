import re
import random
import time
from collections import defaultdict, Counter
import math

def simple_tokenizer(text):
    return re.findall(r'[가-힣]+|[,.!?]', text)

def apply_temperature(probs, temperature):
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if temperature == 1.0:
        return probs
    # log 확률에 온도 적용 후 다시 확률로 변환
    log_probs = [math.log(p) if p > 0 else -1e10 for p in probs]
    tempered = [math.exp(lp / temperature) for lp in log_probs]
    s = sum(tempered)
    return [t / s for t in tempered]

class SimpleStatSeqGenerator:
    def __init__(self):
        self.next_word_freq = defaultdict(Counter)  # {현재단어: Counter(다음단어)}

    def train(self, qa_pairs):
        for q, a in qa_pairs:
            tokens = simple_tokenizer(a) + ['<EOS>']
            for i in range(len(tokens) - 1):
                curr_w = tokens[i]
                next_w = tokens[i+1]
                self.next_word_freq[curr_w][next_w] += 1

    def generate(self, start_word, max_len=30, temperature=1.0):
        word = start_word
        for _ in range(max_len):
            if word == '<EOS>':
                break
            yield word
            next_words = self.next_word_freq.get(word)
            if not next_words:
                break
            total = sum(next_words.values())
            base_probs = [count / total for count in next_words.values()]
            words = list(next_words.keys())

            tempered_probs = apply_temperature(base_probs, temperature)
            word = random.choices(words, tempered_probs)[0]


def load_qa_pairs_from_file(path, max_pairs=2000000):
    qa_pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        lines_buffer = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines_buffer.append(line)
            if len(lines_buffer) == 2:
                q, a = lines_buffer
                qa_pairs.append((q, a))
                lines_buffer.clear()
                if len(qa_pairs) >= max_pairs:
                    break
    return qa_pairs

def main():
    print("단어 빈도 기반 간단 시퀀스 생성 챗봇 시작! 종료: exit")
    qa_pairs = load_qa_pairs_from_file("qa_pairs.txt")
    print(f"QA 쌍 {len(qa_pairs)}개 로드 완료!")

    gen = SimpleStatSeqGenerator()
    gen.train(qa_pairs)
    print("학습 완료!")

    while True:
        user_input = input("질문: ").strip()
        if user_input.lower() == 'exit':
            print("\n챗봇 종료합니다!")
            break

        input_tokens = simple_tokenizer(user_input)
        start_word = input_tokens[-1] if input_tokens else random.choice(list(gen.next_word_freq.keys()))
        
        print("답변:", end=' ', flush=True)
        first = True
        for token in gen.generate(start_word, temperature=0.8):
            # 문장부호면 바로 붙이고, 아니면 앞에 띄어쓰기
            if token in ',.!?':
                print(token, end='', flush=True)
            else:
                if first:
                    print(token, end='', flush=True)
                    first = False
                else:
                    print(' ' + token, end='', flush=True)
            time.sleep(0.05)  # 토큰 사이에 약간 딜레이 주기
        print()



if __name__ == "__main__":
    main()
