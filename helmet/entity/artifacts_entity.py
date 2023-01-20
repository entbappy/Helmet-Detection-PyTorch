from dataclasses import dataclass

# Data ingestion artifacts
@dataclass
class DataIngestionArtifacts:
    train_file_path: str 
    test_file_path: str
    valid_file_path: str
