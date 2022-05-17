import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
from itertools import chain
import scipy.sparse as sp

from dsets import AttributeSnippets
from util.globals import *


def get_tfidf_vectorizer(tfidf_dir: str):
    """
    Returns an sklearn TF-IDF vectorizer. See their website for docs.
    Loading hack inspired by some online blog post lol.
    """

    tfidf_dir = Path(tfidf_dir)

    idf = np.load(tfidf_dir / "idf.npy")
    with open(tfidf_dir / "tfidf_vocab.json", "r") as f:
        vocab = json.load(f)

    class MyVectorizer(TfidfVectorizer):
        TfidfVectorizer.idf_ = idf

    vec = MyVectorizer()
    vec.vocabulary_ = vocab
    vec._tfidf._idf_diag = sp.spdiags(idf, diags=0, m=len(idf), n=len(idf))

    return vec


def collect_stats(tfidf_dir: str):
    """
    Uses wikipedia snippets to collect statistics over a corpus of English text.
    Retrieved later when computing TF-IDF vectors.
    """

    tfidf_dir = Path(tfidf_dir)

    snips_list = AttributeSnippets().snippets_list
    documents = list(chain(*[[y["text"] for y in x["samples"]] for x in snips_list]))

    vec = TfidfVectorizer()
    vec.fit(documents)

    idfs = vec.idf_
    vocab = vec.vocabulary_

    np.save(tfidf_dir / "idf.npy", idfs)
    with open(tfidf_dir / "tfidf_vocab.json", "w") as f:
        json.dump(vocab, f, indent=1)


if __name__ == "__main__":
    collect_stats(TFIDF_DIR)
    get_tfidf_vectorizer(TFIDF_DIR)
