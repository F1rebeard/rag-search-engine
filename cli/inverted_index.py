import math
import os
import pickle
import string
from collections import Counter

from nltk.stem import PorterStemmer

BM25_K1 = 1.5
BM25_B = 0.75
CACHE_DIR = "cache"


def load_stopwords() -> list[str]:
    with open("data/stopwords.txt", "r", encoding="utf-8") as f:
        return f.read().splitlines()


def normalize_text(text: str) -> str:
    return text.lower().translate(str.maketrans("", "", string.punctuation))


def tokenize_text(text: str) -> list[str]:
    tokens = normalize_text(text).split()
    stop_words = set(load_stopwords())
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens if token not in stop_words]


class InvertedIndex:

    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = {}
        self.doc_length: dict[int, int] = {}
        self.doc_lengths_path: str = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        self.term_frequencies[doc_id] = Counter(tokens)
        self.doc_length[doc_id] = len(tokens)
        for token in set(tokens):
            self.index.setdefault(token, set()).add(doc_id)

    def __get_avg_doc_length(self) -> float:
        lengths = [l for l in self.doc_length.values() if l > 0]
        return sum(lengths) / len(lengths) if lengths else 1.0

    def get_document(self, term: str) -> list[int]:
        return sorted(self.index.get(term.lower(), set()))

    def get_bm25_tf(
        self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
    ) -> float:
        tf = self.get_tf(doc_id=doc_id, term=term)
        length_norm = 1 - b + b * (self.doc_length[doc_id] / self.__get_avg_doc_length())
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("Only single-word terms are supported.")
        return self.term_frequencies[doc_id][tokens[0]]

    def get_tfidf(self, doc_id: int, term: str) -> float:
        return self.get_tf(doc_id, term) * self.get_idf(term)

    def get_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("Only single-word terms are supported.")
        total = len(self.docmap)
        matches = len(self.get_document(tokens[0]))
        return math.log((total + 1) / (matches + 1))

    def bm25(self, doc_id: int, term: str) -> float:
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, query: str, limit: int = 5) -> list[tuple[int, float]]:
        tokens = tokenize_text(query)
        scores: dict[int, float] = {}
        for doc_id in self.docmap:
            scores[doc_id] = sum(self.bm25(doc_id, token) for token in tokens)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]

    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("Only single-word terms are supported.")
        N = len(self.docmap)
        n_q = len(self.get_document(tokens[0]))
        return math.log(((N - n_q + 0.5) / (n_q + 0.5)) + 1)

    def build(self) -> None:
        import json
        with open("data/movies.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        for m in data["movies"]:
            self.docmap[m["id"]] = m
            self.__add_document(m["id"], f"{m['title']} {m['description']}")

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(os.path.join(CACHE_DIR, "index.pkl"), "wb") as f:
            pickle.dump(self.index, f)
        with open(os.path.join(CACHE_DIR, "docmap.pkl"), "wb") as f:
            pickle.dump(self.docmap, f)
        with open(os.path.join(CACHE_DIR, "term_frequencies.pkl"), "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_length, f)

    def load(self) -> None:
        cache_files = {
            os.path.join(CACHE_DIR, "index.pkl"),
            os.path.join(CACHE_DIR, "docmap.pkl"),
            os.path.join(CACHE_DIR, "term_frequencies.pkl"),
            self.doc_lengths_path,
        }
        missing_files = {f for f in cache_files if not os.path.exists(f)}
        if missing_files:
            raise FileNotFoundError("Index not found. Run 'build' first.")
        with open(os.path.join(CACHE_DIR, "index.pkl"), "rb") as f:
            self.index = pickle.load(f)
        with open(os.path.join(CACHE_DIR, "docmap.pkl"), "rb") as f:
            self.docmap = pickle.load(f)
        with open(os.path.join(CACHE_DIR, "term_frequencies.pkl"), "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_length = pickle.load(f)
