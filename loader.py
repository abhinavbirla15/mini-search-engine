import os
def load_data(folder_path):
    documents={}
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filename=os.path.join(folder_path, filename)
            with open(filename, "r", encoding="utf-8", errors="replace") as file:
                documents[filename]=file.read()
    return documents