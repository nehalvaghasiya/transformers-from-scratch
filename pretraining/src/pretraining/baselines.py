"""Classical baselines fit only on training text; sklearn imports are local to fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .data import Ticket
from .inference import Prediction


@dataclass
class Baseline:
    name: str
    vectorizer: Any
    classifiers: dict[str, Any]

    def predict(self, texts: list[str]) -> list[Prediction]:
        features = (
            self.vectorizer.transform(texts)
            if self.vectorizer is not None
            else [[0]] * len(texts)
        )
        fields = {
            key: classifier.predict(features).tolist()
            for key, classifier in self.classifiers.items()
        }

        return [
            {key: values[index] for key, values in fields.items()}
            for index in range(len(texts))
        ]


def fit_baselines(rows: list[Ticket]) -> list[Baseline]:
    from sklearn.dummy import DummyClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    texts = [row["text"] for row in rows]
    specs: list[tuple[str, Any]] = [
        ("majority", None),
        (
            "word_tfidf",
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        ),
        (
            "char_tfidf",
            TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True
            ),
        ),
    ]
    models = []

    for name, vectorizer in specs:
        features = (
            vectorizer.fit_transform(texts)
            if vectorizer is not None
            else [[0]] * len(texts)
        )
        classifiers = {}

        for key in ("intent", "priority"):
            classifier = (
                DummyClassifier(strategy="most_frequent")
                if vectorizer is None
                else LogisticRegression(max_iter=1000, random_state=42)
            )
            classifier.fit(features, [row[key] for row in rows])
            classifiers[key] = classifier

        models.append(Baseline(name, vectorizer, classifiers))

    return models
