# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run CLI search
uv run python cli/keyword_search_cli.py search "query here"

# Run main entry point
uv run python main.py

# Install dependencies
uv sync
```

## Architecture

A keyword-based movie search engine built as a learning project for basic information retrieval before advancing to RAG techniques.

**Core Components:**
- `cli/keyword_search_cli.py` — Main search implementation with CLI interface (argparse subcommands)
- `main.py` — Entry point placeholder
- `data/movies.json` — Movie dataset (~26MB, 25,000+ movies with id/title/description)
- `data/stopwords.txt` — 197 English stopwords used to filter query noise

**Search Pipeline (in `cli/keyword_search_cli.py`):**
1. Normalize query (lowercase, strip punctuation)
2. Tokenize into word set
3. Filter stopwords via `get_stopwords_list()` loading `data/stopwords.txt`
4. Stem tokens with NLTK PorterStemmer
5. `count_partial_token_matches()` — scores movies by stemmed substring matches in title tokens
6. `search_movies()` — loads `data/movies.json`, ranks results, returns top 5

**Dependencies:** Python >=3.13, nltk 3.9.1 (for stemming). Package manager: `uv`.
