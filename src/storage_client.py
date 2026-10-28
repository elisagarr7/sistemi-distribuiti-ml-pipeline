import os
from pathlib import Path

from dotenv import load_dotenv
from minio import Minio


def create_minio_client():
    """
    Crea e restituisce un client MinIO utilizzando
    la configurazione presente nel file .env.
    """

    project_root = Path(__file__).resolve().parents[1]
    env_path = project_root / ".env"

    load_dotenv(env_path)

    endpoint = os.getenv("MINIO_ENDPOINT")
    access_key = os.getenv("MINIO_ROOT_USER")
    secret_key = os.getenv("MINIO_ROOT_PASSWORD")

    if not endpoint or not access_key or not secret_key:
        raise ValueError(
            "Configurazione MinIO incompleta nel file .env"
        )

    client = Minio(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=False
    )

    return client