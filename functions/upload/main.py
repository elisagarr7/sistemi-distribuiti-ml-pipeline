import io
import json
import os

from minio import Minio


BUCKET_NAME = "raw-data"


def json_response(context, body, status_code=200):
    return context.Response(
        body=json.dumps(body),
        headers={},
        content_type="application/json",
        status_code=status_code,
    )


def get_header(headers, header_name):
    if not headers:
        return None

    target = header_name.lower()

    for key, value in headers.items():
        if isinstance(key, bytes):
            key = key.decode("utf-8")

        if str(key).lower() == target:
            if isinstance(value, bytes):
                value = value.decode("utf-8")

            return str(value)

    return None


def handler(context, event):
    try:
        method = event.method

        if isinstance(method, bytes):
            method = method.decode("utf-8")

        if str(method).upper() != "POST":
            return json_response(
                context,
                {
                    "status": "error",
                    "message": "Metodo HTTP non consentito",
                },
                405,
            )

        endpoint = os.getenv("MINIO_ENDPOINT")
        access_key = os.getenv("MINIO_ACCESS_KEY")
        secret_key = os.getenv("MINIO_SECRET_KEY")

        if not endpoint or not access_key or not secret_key:
            return json_response(
                context,
                {
                    "status": "error",
                    "message": "Configurazione MinIO incompleta",
                },
                500,
            )

        body = event.body

        if isinstance(body, str):
            body = body.encode("utf-8")
        elif isinstance(body, bytearray):
            body = bytes(body)

        if not body:
            return json_response(
                context,
                {
                    "status": "error",
                    "message": "Nessun file ricevuto",
                },
                400,
            )

        filename = get_header(
            event.headers,
            "X-Filename",
        )

        if not filename:
            filename = "pricerunner_aggregate.csv"

        filename = os.path.basename(filename)

        if not filename.lower().endswith(".csv"):
            return json_response(
                context,
                {
                    "status": "error",
                    "message": "Il file deve essere in formato CSV",
                },
                400,
            )

        client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=False,
        )

        if not client.bucket_exists(BUCKET_NAME):
            return json_response(
                context,
                {
                    "status": "error",
                    "message": f"Bucket {BUCKET_NAME} non disponibile",
                },
                500,
            )

        client.put_object(
            BUCKET_NAME,
            filename,
            io.BytesIO(body),
            length=len(body),
            content_type="text/csv",
        )

        context.logger.info_with(
            "Dataset uploaded",
            bucket=BUCKET_NAME,
            object_name=filename,
            size_bytes=len(body),
        )

        return json_response(
            context,
            {
                "status": "success",
                "message": "Dataset caricato correttamente",
                "bucket": BUCKET_NAME,
                "object_name": filename,
                "size_bytes": len(body),
            },
            201,
        )

    except Exception as exc:
        context.logger.error_with(
            "Dataset upload failed",
            error=str(exc),
        )

        return json_response(
            context,
            {
                "status": "error",
                "message": str(exc),
            },
            500,
        )