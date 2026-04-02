import math
import os
import pickle
import string
from collections import Counter

from nltk.stem import PorterStemmer


def normalize_text(text: str) -> str:
    return text.lower().translate(str.maketrans("", "", string.punctuation))


def tokenize_text(text: str) -> set[str]:
    return set(text.split())


class InvertedIndex:

    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in normalize_text(text).split()]
        self.term_frequencies[doc_id] = Counter(tokens)
        for token in set(tokens):
            self.index.setdefault(token, set()).add(doc_id)

    def get_document(self, term: str) -> list[int]:
        return sorted(self.index.get(term.lower(), set()))

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = tokenize_text(normalize_text(term))
        if len(tokens) != 1:
            raise ValueError("Only single-word terms are supported.")
        return self.term_frequencies[doc_id][next(iter(tokens))]

    def get_tfidf(self, doc_id: int, term: str) -> float:
        return self.get_tf(doc_id, term) * self.get_idf(term)

    def get_idf(self, term: str) -> float:
        stemmer = PorterStemmer()
        stemmed = stemmer.stem(normalize_text(term))
        total = len(self.docmap)
        matches = len(self.get_document(stemmed))
        return math.log((total + 1) / (matches + 1))

    def build(self) -> None:
        import json
        with open("data/movies.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        for m in data["movies"]:
            self.docmap[m["id"]] = m
            self.__add_document(m["id"], f"{m['title']} {m['description']}")

    def save(self) -> None:
        os.makedirs("cache", exist_ok=True)
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)
        with open("cache/term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self) -> None:
        cache_files = {"cache/index.pkl", "cache/docmap.pkl", "cache/term_frequencies.pkl"}
        missing_files = {f for f in cache_files if not os.path.exists(f)}
        if missing_files:
            raise FileNotFoundError("Index not found. Run 'build' first.")
        with open("cache/index.pkl", "rb") as f:
            self.index = pickle.load(f)
        with open("cache/docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)
        with open("cache/term_frequencies.pkl", "rb") as f:
            self.term_frequencies = pickle.load(f)
