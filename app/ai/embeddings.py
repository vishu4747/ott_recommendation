from sentence_transformers import SentenceTransformer

# Free, offline AI model
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str):
    return model.encode(text).tolist()
