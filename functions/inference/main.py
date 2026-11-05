import io
import json
import os
import re

import threading
import time

import joblib
import nuclio_sdk

from minio import Minio


MODELS_BUCKET = "models"

_CACHED_MODEL = None
_CACHED_VERSION = None
_CACHED_ALGORITHM = None

_CACHE_LOCK = threading.Lock()

def read_json_object(client, bucket_name, object_name):
    response = None

    try:
        response = client.get_object(
            bucket_name,
            object_name,
        )

        content = response.read().decode("utf-8")

        return json.loads(content)

    finally:
        if response is not None:
            response.close()
            response.release_conn()



def get_minio_client():
    endpoint = os.getenv("MINIO_ENDPOINT")
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")

    if not endpoint or not access_key or not secret_key:
        raise ValueError(
            "Configurazione MinIO incompleta"
        )

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
        return json.loads(
            body.decode("utf-8")
        )

    if isinstance(body, str):
        return json.loads(body)

    if isinstance(body, dict):
        return body

    return {}


def normalize_title(title):
    """
    Applica la stessa normalizzazione minima
    usata durante il preprocessing.
    """

    title = str(title)
    title = title.strip().lower()
    title = re.sub(r"\s+", " ", title)

    return title


def load_model():
    """
    Recupera il modello attivo.

    Se la versione è già presente in cache,
    riutilizza il modello in memoria.

    Se current.json indica una nuova versione,
    scarica il nuovo modello da MinIO e aggiorna
    automaticamente la cache.
    """

    global _CACHED_MODEL
    global _CACHED_VERSION
    global _CACHED_ALGORITHM

    client = get_minio_client()

    # Leggiamo sempre current.json per sapere
    # quale versione deve essere attiva.
    current_model = read_json_object(
        client,
        MODELS_BUCKET,
        "current.json"
    )

    model_version = current_model.get(
        "active_version"
    )

    algorithm = current_model.get(
        "algorithm"
    )

    model_object = current_model.get(
        "model_object"
    )

    if not model_version or not model_object:
        raise ValueError(
            "Informazioni del modello attivo non valide"
        )

    # ==========================================
    # CACHE HIT
    # ==========================================

    if (
        _CACHED_MODEL is not None
        and _CACHED_VERSION == model_version
        and _CACHED_ALGORITHM == algorithm
    ):
        return (
            _CACHED_MODEL,
            _CACHED_VERSION,
            _CACHED_ALGORITHM,
            True,
        )

    # ==========================================
    # CACHE MISS / MODELLO AGGIORNATO
    # ==========================================

    with _CACHE_LOCK:

        # Ricontrolliamo la cache perché un'altra
        # richiesta potrebbe averla aggiornata
        # mentre stavamo aspettando il lock.
        if (
            _CACHED_MODEL is not None
            and _CACHED_VERSION == model_version
            and _CACHED_ALGORITHM == algorithm
        ):
            return (
                _CACHED_MODEL,
                _CACHED_VERSION,
                _CACHED_ALGORITHM,
                True,
            )

        model_response = None

        try:
            model_response = client.get_object(
                MODELS_BUCKET,
                model_object,
            )

            model_bytes = model_response.read()

        finally:
            if model_response is not None:
                model_response.close()
                model_response.release_conn()

        model = joblib.load(
            io.BytesIO(model_bytes)
        )

        # Aggiornamento della cache
        _CACHED_MODEL = model
        _CACHED_VERSION = model_version
        _CACHED_ALGORITHM = algorithm

        return (
            _CACHED_MODEL,
            _CACHED_VERSION,
            _CACHED_ALGORITHM,
            False,
        )


def handler(
    context: nuclio_sdk.Context,
    event: nuclio_sdk.Event
):
    try:
        body = parse_request_body(event)

        product_title = body.get(
            "product_title"
        )

        if product_title is None:
            return context.Response(
                body=json.dumps(
                    {
                        "status": "error",
                        "message": (
                            "Campo 'product_title' mancante"
                        )
                    }
                ),
                headers={},
                content_type="application/json",
                status_code=400
            )

        if not str(product_title).strip():
            return context.Response(
                body=json.dumps(
                    {
                        "status": "error",
                        "message": (
                            "'product_title' non può essere vuoto"
                        )
                    }
                ),
                headers={},
                content_type="application/json",
                status_code=400
            )

        normalized_title = normalize_title(
            product_title
        )

        context.logger.info_with(
            "Inference started",
            product_title=normalized_title
        )


        model_load_start = time.perf_counter()

        (
            model,
            model_version,
            algorithm,
            cache_hit,
        ) = load_model()

        model_load_ms = (
            time.perf_counter() - model_load_start
        ) * 1000



        predicted_category = model.predict(
            [normalized_title]
        )[0]

        probabilities = model.predict_proba(
            [normalized_title]
        )[0]

        classifier = model.named_steps[
            "classifier"
        ]

        classes = classifier.classes_

        max_probability_index = (
            probabilities.argmax()
        )

        confidence = float(
            probabilities[
                max_probability_index
            ]
        )

        context.logger.info_with(
            "Inference completed",
            predicted_category=predicted_category,
            confidence=confidence,
            model_version=model_version,
            cache_hit=cache_hit,
            model_load_ms=round(
                model_load_ms,
                2
            ),
        )

        return context.Response(
            body=json.dumps(
                {
                    "status": "success",
                    "input": {
                        "product_title": product_title
                    },
                    "normalized_title": normalized_title,
                    "predicted_category": (
                        predicted_category
                    ),
                    "confidence": confidence,
                    "model_version": model_version,
                    "algorithm": algorithm,
                    "cache_hit": cache_hit,
                    "model_load_ms": round(
                        model_load_ms,
                        2
                    ),
                }
            ),
            headers={},
            content_type="application/json",
            status_code=200
        )

    except Exception as exc:

        context.logger.error_with(
            "Inference failed",
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