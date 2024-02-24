import os 
from chromadb.config import Settings 



#Define the chroma settings
CHROMA_SETTINGS = Settings(
    chroma_db_impl = '',
    persist_directory = "db",
    anonymized_telemetry = False
)