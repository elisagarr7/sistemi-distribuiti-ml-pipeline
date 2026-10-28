import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.storage_client import create_minio_client


client = create_minio_client()

print("Connessione a MinIO riuscita!")

buckets = client.list_buckets()

print("\nBucket disponibili:")

for bucket in buckets:
    print(f"- {bucket.name}")


print("\nOggetti presenti in raw-data:")

objects = client.list_objects(
    "raw-data",
    recursive=True
)

for obj in objects:
    print(f"- {obj.object_name} ({obj.size} byte)")


download_path = (
    project_root
    / "data"
    / "download_test"
    / "pricerunner_aggregate_downloaded.csv"
)

client.fget_object(
    bucket_name="raw-data",
    object_name="pricerunner_aggregate.csv",
    file_path=str(download_path)
)

print(
    "\nFile scaricato correttamente in:",
    download_path
)


df_test = pd.read_csv(download_path)

print(
    "Dimensioni del dataset scaricato:",
    df_test.shape
)