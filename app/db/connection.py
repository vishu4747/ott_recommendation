import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# Connect to MongoDB server
client = MongoClient(os.getenv("MONGO_URI"))

db = client["itap_recommendation"]

watch_collection = db["watch_history"]
content_collection = db["contents"]

print("MongoDB connected successfully")