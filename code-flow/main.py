from file_ingestion.local_ingestor import LocalFileIngestor

if __name__ == "__main__":
    code_dir = "/Users/yash/Documents/Repos/StreamlitChat"  # Replace with your test folder path

    ingestor = LocalFileIngestor(code_dir)
    files = ingestor.read_files()

    print(f"Found {len(files)} code files:\n")
    for f in files:
        print(f"--- {f['path']} ---")
        print(f['content'][:50])  # Print preview only
        print("\n" + "-"*50 + "\n")
