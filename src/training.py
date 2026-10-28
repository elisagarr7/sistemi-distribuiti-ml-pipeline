import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


REQUIRED_COLUMNS = {
    "product_id",
    "product_title",
    "cluster_id",
    "category_label",
}


def validate_training_dataframe(df: pd.DataFrame) -> None:
    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset processato non valido. "
            f"Colonne mancanti: {sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError("Il dataset processato è vuoto.")

    if df["product_title"].isna().any():
        raise ValueError("Sono presenti product_title mancanti.")

    if df["category_label"].isna().any():
        raise ValueError("Sono presenti category_label mancanti.")


def split_by_cluster(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42
):
    """
    Divide train e test senza permettere che lo stesso cluster_id
    compaia in entrambi gli insiemi.

    La stratificazione viene effettuata sulla categoria dei cluster.
    """

    cluster_targets = (
        df[["cluster_id", "category_label"]]
        .drop_duplicates()
        .copy()
    )

    # Un cluster deve appartenere ad una sola categoria
    categories_per_cluster = (
        cluster_targets
        .groupby("cluster_id")["category_label"]
        .nunique()
    )

    if (categories_per_cluster > 1).any():
        raise ValueError(
            "Sono presenti cluster associati a più categorie."
        )

    train_clusters, test_clusters = train_test_split(
        cluster_targets,
        test_size=test_size,
        random_state=random_state,
        stratify=cluster_targets["category_label"]
    )

    train_cluster_ids = set(train_clusters["cluster_id"])
    test_cluster_ids = set(test_clusters["cluster_id"])

    train_df = df[
        df["cluster_id"].isin(train_cluster_ids)
    ].copy()

    test_df = df[
        df["cluster_id"].isin(test_cluster_ids)
    ].copy()

    overlap = train_cluster_ids & test_cluster_ids

    if overlap:
        raise RuntimeError(
            "Errore nello split: alcuni cluster sono presenti "
            "sia nel train sia nel test."
        )

    return train_df, test_df


def train_and_evaluate(df: pd.DataFrame):
    """
    Addestra TF-IDF + Logistic Regression e restituisce:
    - pipeline addestrata
    - metriche principali
    - classification report
    """

    validate_training_dataframe(df)

    train_df, test_df = split_by_cluster(df)

    X_train = train_df["product_title"]
    y_train = train_df["category_label"]

    X_test = test_df["product_title"]
    y_test = test_df["category_label"]

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer()
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000
                )
            ),
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )
    )

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )

    metrics = {
        "input_rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_clusters": int(
            train_df["cluster_id"].nunique()
        ),
        "test_clusters": int(
            test_df["cluster_id"].nunique()
        ),
        "cluster_overlap": 0,
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "num_classes": int(
            df["category_label"].nunique()
        ),
    }

    return model, metrics, report