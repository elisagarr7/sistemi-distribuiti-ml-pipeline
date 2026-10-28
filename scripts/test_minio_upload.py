import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.storage_client import create_minio_client


client = create_minio_client()

# Creazione di un piccolo file locale di test
test_directory = project_root / "data" / "upload_test"
test_directory.mkdir(parents=True, exist_ok=True)

test_file = test_directory / "test_upload.txt"

test_file.write_text(
    "Test di upload Python verso MinIO.",
    encoding="utf-8"
)

# Upload del file nel bucket processed-data
client.fput_object(
    bucket_name="processed-data",
    object_name="test_upload.txt",
    file_path=str(test_file),
    content_type="text/plain"
)

print("Upload completato.")

# Verifica dell'oggetto appena caricato
object_info = client.stat_object(
    bucket_name="processed-data",
    object_name="test_upload.txt"
)

print("Oggetto trovato in MinIO:")
print("Nome:", object_info.object_name)
print("Dimensione:", object_info.size, "byte")

client.remove_object(
    bucket_name="processed-data",
    object_name="test_upload.txt"
)

test_file.unlink()

print("File di test rimosso da MinIO e dal filesystem locale.")