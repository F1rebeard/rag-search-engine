#!/usr/bin/env python3
import argparse
import json

from nltk.stem import PorterStemmer
from inverted_index import InvertedIndex, normalize_text, tokenize_text


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


def rank_documents(idx: InvertedIndex, query: str) -> list[int]:
    stemmer = PorterStemmer()
    query_tokens = {stemmer.stem(token) for token in tokenize_text(normalize_text(query))}
    scores: dict[int, int] = {}
    for token in query_tokens:
        for doc_id in idx.get_document(token):
            scores[doc_id] = scores.get(doc_id, 0) + 1
        if len(scores) >= 5:
            break
    return sorted(scores, key=lambda d: -scores[d])


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

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a term in a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to look up")

    idf_parser = subparsers.add_parser("idf", help="Get IDF value for a term")
    idf_parser.add_argument("term", type=str, help="Term to look up")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF score for a term in a document")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to look up")


    args = parser.parse_args()

    idx = InvertedIndex()
    if args.command != "build":
        idx.load()

    match args.command:
        case "search":
            ranked = rank_documents(idx, args.query)
            print(f"Search results for: {args.query}")
            for doc_id in ranked[:5]:
                movie = idx.docmap[doc_id]
                print(f"{movie['id']}. {movie['title']}")
        case "build":
            idx.build()
            idx.save()
        case "tf":
            print(idx.get_tf(args.doc_id, args.term))
        case "idf":
            idf = idx.get_idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            tf_idf = idx.get_tfidf(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()