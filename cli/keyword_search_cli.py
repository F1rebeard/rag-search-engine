#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import string

from nltk.stem import PorterStemmer


class InvertedIndex:

    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        stemmer = PorterStemmer()
        tokens = {stemmer.stem(token) for token in tokenize_text(normalize_text(text))}
        for token in tokens:
            self.index.setdefault(token, set()).add(doc_id)

    def get_document(self, term: str) -> list[int]:
        return sorted(self.index.get(term.lower(), set()))

    def build(self) -> None:
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

    def load(self) -> None:
        if not os.path.exists("cache/index.pkl") or not os.path.exists("cache/docmap.pkl"):
            raise FileNotFoundError("Index not found. Run 'build' first.")
        with open("cache/index.pkl", "rb") as f:
            self.index = pickle.load(f)
        with open("cache/docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)


def normalize_text(text: str) -> str:
    return text.lower().translate(str.maketrans("", "", string.punctuation))


def tokenize_text(text: str) -> set[str]:
    return set(text.split())


def get_stopwords_list() -> set[str]:
    with open("data/stopwords.txt", "r", encoding="utf-8") as f:
        return set(f.read().splitlines())

def clean_query(query: str) -> str:
    stopwords = get_stopwords_list()
    return " ".join(token for token in query.split() if token not in stopwords)


def count_partial_token_matches(query: str, title: str) -> int:
    stemmer = PorterStemmer()
    cleaned_query = clean_query(query)

    query_tokens = tokenize_text(normalize_text(cleaned_query))
    stemmed_tokens = [stemmer.stem(token) for token in query_tokens]
    title_tokens = tokenize_text(normalize_text(title))

    return sum(
        any(query_token in title_token for title_token in title_tokens)
        for query_token in stemmed_tokens
    )


def search_movies(query: str) -> list[dict]:
    with open("data/movies.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    matches = []
    for movie in data["movies"]:
        score = count_partial_token_matches(query, movie["title"])
        if score > 0:
            matches.append((score, movie))

    matches.sort(key=lambda item: item[0], reverse=True)
    return [movie for _, movie in matches[:5]]


def print_results(query: str, results: list[dict]) -> None:
    print(f"Search results for: {query}")
    for idx, movie in enumerate(results, start=1):
        print(f"{idx}. {movie['title']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    subparsers.add_parser("build", help="Build inverted index and save to cache")

    args = parser.parse_args()

    match args.command:
        case "search":
            idx = InvertedIndex()
            try:
                idx.load()
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return
            stemmer = PorterStemmer()
            query_tokens = {stemmer.stem(token) for token in tokenize_text(normalize_text(args.query))}
            scores: dict[int, int] = {}
            for token in query_tokens:
                for doc_id in idx.get_document(token):
                    scores[doc_id] = scores.get(doc_id, 0) + 1
                if len(scores) >= 5:
                    break
            ranked = sorted(scores, key=lambda d: -scores[d])
            print(f"Search results for: {args.query}")
            for doc_id in ranked[:5]:
                movie = idx.docmap[doc_id]
                print(f"{movie['id']}. {movie['title']}")

        case "build":
            idx = InvertedIndex()
            idx.build()
            idx.save()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()