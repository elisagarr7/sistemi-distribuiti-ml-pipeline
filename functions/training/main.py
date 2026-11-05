import io
import json
import os
from urllib.parse import unquote_plus

from datetime import datetime, timezone

import joblib
import pandas as pd
from minio import Minio
import nuclio_sdk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.model_selection import (
    train_test_split,
    StratifiedGroupKFold,
)


REQUIRED_COLUMNS = {
    "product_id",
    "product_title",
    "cluster_id",
    "category_label",
}


def get_minio_client():
    endpoint = os.getenv("MINIO_ENDPOINT")
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")

    if not endpoint or not access_key or not secret_key:
        raise ValueError("Configurazione MinIO incompleta")

    return Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=False
    )


def parse_request_body(event):
    body = event.body

    if not body:
        return {}

    if isinstance(body, bytes):
        return json.loads(body.decode("utf-8"))

    if isinstance(body, str):
        return json.loads(body)

    if isinstance(body, dict):
        return body

    return {}


def extract_input_object(body):
    """
    Supporta sia eventi automatici MinIO
    sia chiamate HTTP manuali.
    """

    if "Records" in body and body["Records"]:
        record = body["Records"][0]

        try:
            bucket = record["s3"]["bucket"]["name"]
            object_name = record["s3"]["object"]["key"]
        except KeyError as exc:
            raise ValueError(
                "Evento MinIO non valido"
            ) from exc

        object_name = unquote_plus(object_name)

        event_name = record.get(
            "eventName",
            "unknown"
        )

        return bucket, object_name, event_name

    bucket = body.get(
        "bucket",
        "processed-data"
    )

    object_name = body.get(
        "object",
        "products_clean.csv"
    )

    return bucket, object_name, "manual-http"


def validate_dataframe(df):
    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Colonne mancanti: {sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError(
            "Il dataset processato è vuoto"
        )

    if df["product_title"].isna().any():
        raise ValueError(
            "Sono presenti product_title mancanti"
        )

    if df["category_label"].isna().any():
        raise ValueError(
            "Sono presenti category_label mancanti"
        )


def split_by_cluster(
    df,
    test_size=0.20,
    random_state=42
):
    cluster_targets = (
        df[
            ["cluster_id", "category_label"]
        ]
        .drop_duplicates()
        .copy()
    )

    categories_per_cluster = (
        cluster_targets
        .groupby("cluster_id")["category_label"]
        .nunique()
    )

    if (categories_per_cluster > 1).any():
        raise ValueError(
            "Alcuni cluster appartengono a più categorie"
        )

    train_clusters, test_clusters = train_test_split(
        cluster_targets,
        test_size=test_size,
        random_state=random_state,
        stratify=cluster_targets["category_label"]
    )

    train_ids = set(
        train_clusters["cluster_id"]
    )

    test_ids = set(
        test_clusters["cluster_id"]
    )

    overlap = train_ids & test_ids

    if overlap:
        raise RuntimeError(
            "Data leakage: cluster presenti sia "
            "nel train sia nel test"
        )

    train_df = df[
        df["cluster_id"].isin(train_ids)
    ].copy()

    test_df = df[
        df["cluster_id"].isin(test_ids)
    ].copy()

    return train_df, test_df


def build_classifier(
    algorithm_name,
    calibration_cv=None,
):
    if algorithm_name == "LogisticRegression":
        return LogisticRegression(
            max_iter=1000
        )

    if algorithm_name == "MultinomialNB":
        return MultinomialNB(
            alpha=1.0
        )

    if algorithm_name == "LinearSVM":
        return CalibratedClassifierCV(
            LinearSVC(
                C=1.0,
                random_state=42,
            ),
            method="sigmoid",
            cv=calibration_cv,
            ensemble=False,
        )

    raise ValueError(
        f"Algoritmo non supportato: {algorithm_name}"
    )


def build_pipeline(
    algorithm_name,
    calibration_cv=None,
):
    classifier = build_classifier(
        algorithm_name,
        calibration_cv,
    )

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer()
            ),
            (
                "classifier",
                classifier
            ),
        ]
    )


def train_model(df):
    validate_dataframe(df)

    # ==========================================
    # 1. Split principale TRAIN / TEST
    # ==========================================

    train_df, test_df = split_by_cluster(df)

    # Il test set viene tenuto completamente
    # fuori dalla selezione del modello.


    # ==========================================
    # 2. Split TRAIN / VALIDATION
    # ==========================================

    candidate_train_df, validation_df = (
        split_by_cluster(
            train_df,
            test_size=0.20,
            random_state=42,
        )
    )


    # ==========================================
    # 3. Dati per la selezione dei candidati
    # ==========================================

    X_candidate = candidate_train_df[
        "product_title"
    ]

    y_candidate = candidate_train_df[
        "category_label"
    ]

    candidate_groups = candidate_train_df[
        "cluster_id"
    ]

    X_validation = validation_df[
        "product_title"
    ]

    y_validation = validation_df[
        "category_label"
    ]


    # ==========================================
    # 4. Fold per calibrazione Linear SVM
    # ==========================================

    calibration_splitter = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    candidate_calibration_cv = list(
        calibration_splitter.split(
            X_candidate,
            y_candidate,
            groups=candidate_groups,
        )
    )


    # ==========================================
    # 5. Algoritmi candidati
    # ==========================================

    candidate_algorithms = [
        "LogisticRegression",
        "MultinomialNB",
        "LinearSVM",
    ]

    candidate_results = []

    best_algorithm = None
    best_macro_f1 = -1.0
    best_accuracy = -1.0


    # ==========================================
    # 6. Training e confronto dei candidati
    # ==========================================

    for algorithm_name in candidate_algorithms:

        calibration_cv = (
            candidate_calibration_cv
            if algorithm_name == "LinearSVM"
            else None
        )

        candidate_model = build_pipeline(
            algorithm_name,
            calibration_cv,
        )

        candidate_model.fit(
            X_candidate,
            y_candidate,
        )

        validation_predictions = (
            candidate_model.predict(
                X_validation
            )
        )

        accuracy = accuracy_score(
            y_validation,
            validation_predictions,
        )

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                y_validation,
                validation_predictions,
                average="macro",
                zero_division=0,
            )
        )

        result = {
            "algorithm": algorithm_name,
            "accuracy": float(accuracy),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(f1),
        }

        candidate_results.append(
            result
        )


        # ======================================
        # 7. Controllo miglior candidato
        # ======================================

        is_better = (
            f1 > best_macro_f1
            or (
                f1 == best_macro_f1
                and accuracy > best_accuracy
            )
        )

        if is_better:
            best_algorithm = algorithm_name
            best_macro_f1 = float(f1)
            best_accuracy = float(accuracy)


    if best_algorithm is None:
        raise RuntimeError(
            "Nessun modello candidato "
            "è stato selezionato"
        )


    # ==========================================
    # 8. Ordina risultati e marca vincitore
    # ==========================================

    candidate_results.sort(
        key=lambda item: (
            item["macro_f1"],
            item["accuracy"],
        ),
        reverse=True,
    )

    for result in candidate_results:
        result["selected"] = (
            result["algorithm"]
            == best_algorithm
        )


    # ==========================================
    # 9. Training definitivo del vincitore
    #    sull'intero TRAIN originale
    # ==========================================

    X_train = train_df[
        "product_title"
    ]

    y_train = train_df[
        "category_label"
    ]

    train_groups = train_df[
        "cluster_id"
    ]


    final_calibration_cv = None

    if best_algorithm == "LinearSVM":

        final_calibration_splitter = (
            StratifiedGroupKFold(
                n_splits=3,
                shuffle=True,
                random_state=42,
            )
        )

        final_calibration_cv = list(
            final_calibration_splitter.split(
                X_train,
                y_train,
                groups=train_groups,
            )
        )


    best_model = build_pipeline(
        best_algorithm,
        final_calibration_cv,
    )

    best_model.fit(
        X_train,
        y_train,
    )


    # ==========================================
    # 10. Valutazione finale sul TEST
    # ==========================================

    X_test = test_df[
        "product_title"
    ]

    y_test = test_df[
        "category_label"
    ]

    predictions = best_model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )
    )


    # ==========================================
    # 11. Metriche finali
    # ==========================================

    metrics = {
        "input_rows": int(len(df)),

        "train_rows": int(
            len(train_df)
        ),

        "test_rows": int(
            len(test_df)
        ),

        "train_clusters": int(
            train_df[
                "cluster_id"
            ].nunique()
        ),

        "test_clusters": int(
            test_df[
                "cluster_id"
            ].nunique()
        ),

        "cluster_overlap": 0,

        "accuracy": float(
            accuracy
        ),

        "macro_precision": float(
            precision
        ),

        "macro_recall": float(
            recall
        ),

        "macro_f1": float(
            f1
        ),

        "num_classes": int(
            df[
                "category_label"
            ].nunique()
        ),

        "algorithm": best_algorithm,

        "selection": {
            "metric": "macro_f1",

            "candidate_train_rows": int(
                len(candidate_train_df)
            ),

            "validation_rows": int(
                len(validation_df)
            ),

            "validation_clusters": int(
                validation_df[
                    "cluster_id"
                ].nunique()
            ),

            "candidate_models": (
                candidate_results
            ),
        },
    }

    return best_model, metrics


def handler(
    context: nuclio_sdk.Context,
    event: nuclio_sdk.Event
):
    try:
        body = parse_request_body(event)

        input_bucket, input_object, source_event = (
            extract_input_object(body)
        )

        models_bucket = "models"
        

        context.logger.info_with(
            "Training started",
            bucket=input_bucket,
            object=input_object,
            source_event=source_event
        )

        client = get_minio_client()

        # 1. Download dataset processato
        response = client.get_object(
            input_bucket,
            input_object
        )

        try:
            csv_bytes = response.read()
        finally:
            response.close()
            response.release_conn()

        # 2. Lettura con Pandas
        df = pd.read_csv(
            io.BytesIO(csv_bytes)
        )

        # 3. Training
        model, metrics = train_model(df)

        #  Timestamp e versioning
        training_time = datetime.now(timezone.utc)
        version_id = training_time.strftime(
            "%Y%m%dT%H%M%SZ"
        )
        trained_at = training_time.isoformat()

        # 4. Serializzazione modello in memoria
        model_buffer = io.BytesIO()

        joblib.dump(
            model,
            model_buffer
        )

        model_bytes = model_buffer.getvalue()

        # 5. Upload modello
 
        metrics["model_version"] = version_id
        metrics["trained_at"] = trained_at

        # 6. Serializzazione metriche
        metrics_bytes = json.dumps(
            metrics,
            indent=2
        ).encode("utf-8")

        version_model_object = (
            f"versions/{version_id}/"
            "product_classifier.joblib"
        )

        version_metrics_object = (
            f"versions/{version_id}/"
            "training_metrics.json"
        )

        # 7. Upload metriche

        context.logger.info_with(
            "Training completed",
            accuracy=metrics["accuracy"],
            macro_f1=metrics["macro_f1"],
            train_rows=metrics["train_rows"],
            test_rows=metrics["test_rows"]
        )

        client.put_object(
            "models",
            version_model_object,
            io.BytesIO(model_bytes),
            length=len(model_bytes),
            content_type="application/octet-stream",
        )

        client.put_object(
            "models",
            version_metrics_object,
            io.BytesIO(metrics_bytes),
            length=len(metrics_bytes),
            content_type="application/json",
        )

        # current.json
        current_model = {
            "active_version": version_id,
            "created_at": trained_at,
            "algorithm": metrics["algorithm"],
            "model_object": version_model_object,
            "metrics_object": version_metrics_object,
        }

        # serializzazione in json
        current_bytes = json.dumps(
            current_model,
            indent=2,
        ).encode("utf-8")

        # upload current.json
        client.put_object(
            "models",
            "current.json",
            io.BytesIO(current_bytes),
            length=len(current_bytes),
            content_type="application/json",
        )



        context.logger.info_with(
            "Model version created",
            model_version=version_id,
            algorithm=metrics["algorithm"],
            model_object=version_model_object,
        )

        return context.Response(
            body=json.dumps(
                {
                    "status": "success",
                    "input": (
                        f"{input_bucket}/"
                        f"{input_object}"
                    ),
                    "model": (
                        f"{models_bucket}/"
                        f"{version_model_object}"
                    ),
                    "metrics": metrics
                }
            ),
            headers={},
            content_type="application/json",
            status_code=200
        )

    except Exception as exc:
        context.logger.error_with(
            "Training failed",
            error=str(exc)
        )

        return context.Response(
            body=json.dumps(
                {
                    "status": "error",
                    "message": str(exc)
                }
            ),
            headers={},
            content_type="application/json",
            status_code=500
        )