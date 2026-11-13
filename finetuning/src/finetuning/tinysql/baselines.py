"""Traditional reference approaches: nearest-neighbor and TF-IDF SQL retrieval + value copying."""

from __future__ import annotations

import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from .data import AMOUNTS, CATEGORIES, LIMITS, STATES, STATUS
from .utils import Record, read_jsonl


class RetrievalBaseline:
    def __init__(
        self, train_rows: list[Record], analyzer: str = "word", slots: bool = False
    ) -> None:
        self.rows = list(train_rows)
        self.slots = slots
        self.vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=(1, 2) if analyzer == "word" else (2, 5),
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform([x["question"] for x in self.rows])

    def predict_many(self, questions: list[str]) -> list[str]:
        Q = self.vectorizer.transform(questions)
        sims = Q @ self.matrix.T
        idx = (
            sims.argmax(axis=1).A.ravel()
            if hasattr(sims.argmax(axis=1), "A")
            else sims.argmax(axis=1).ravel()
        )
        return [
            self.copy_values(questions[i], self.rows[j]) if self.slots else self.rows[j]["sql"]
            for i, j in enumerate(idx)
        ]

    @staticmethod
    def copy_values(question: str, reference: Record) -> str:
        sql = str(reference["sql"])
        source = reference["values"]
        q = question.lower()
        # Only copy observed structured values. Family/SQL skeleton comes from nearest-
        # neighbor retrieval.
        lex = {"state": STATES, "status": STATUS, "category": CATEGORIES}
        for field, options in lex.items():
            if field not in source:
                continue
            found = [
                x for x in options if re.search(r"(?<![a-z_])" + re.escape(x) + r"(?![a-z_])", q)
            ]
            if len(found) == 1:
                sql = sql.replace("'" + str(source[field]) + "'", "'" + found[0] + "'")
        ints = [int(x) for x in re.findall(r"\b\d+\b", q)]
        if "amount" in source:
            amounts = [x for x in ints if x in AMOUNTS and x >= 10]
            if amounts:
                tgt = max(amounts)
                sql = re.sub(r"(?<!\d)" + str(source["amount"]) + r"(?!\d)", str(tgt), sql, count=1)
        if "limit" in source:
            limits = [x for x in ints if x in LIMITS]
            if limits:
                tgt = min(limits)
                sql = re.sub(
                    r"(?i)(LIMIT\s+)" + str(source["limit"]) + r"\b",
                    lambda m: m.group(1) + str(tgt),
                    sql,
                )
        return sql


def make_baselines(train_dir: str | Path = "data") -> dict[str, RetrievalBaseline]:
    rows = read_jsonl(Path(train_dir) / "train.jsonl")
    return {
        "word_tfidf_retrieval": RetrievalBaseline(rows, "word"),
        "char_tfidf_retrieval": RetrievalBaseline(rows, "char"),
        "word_tfidf_slots": RetrievalBaseline(rows, "word", True),
    }
