from file_ingestion.local_ingestor import LocalFileIngestor
from utils.chunker import chunk_text

if __name__ == "__main__":
    code_dir = "/Users/yash/Documents/Repos/StreamlitChat/T2H"  # Replace with your test folder path

    ingestor = LocalFileIngestor(code_dir)
    files = ingestor.read_files()

    # Prepare combined content
    full_code = "\n\n".join([f"# File: {f['path']}\n{f['content']}" for f in files])
    chunks = chunk_text(full_code)

    print(f"✅ Split into {len(chunks)} chunks")
    print(chunk_text(full_code))
