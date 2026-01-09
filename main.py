from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from datetime import datetime
from app.data.data import CONTENTS
from app.db.connection import watch_collection, content_collection
from app.ai.embeddings import get_embedding
from app.recommender.ai_recommender import get_ai_recommendations
from app.recommender.trending import get_trending_content
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="OTT AI Recommendation Engine")

class Content(BaseModel):
    content_id: int
    title: str
    poster: str
    genres: list
    type: str
    is_reel: bool
    popularity: int = 0

# ---------------------
# Load contents + embeddings
# ---------------------
@app.on_event("startup")
def load_embeddings():
    for content in CONTENTS:
        exists = content_collection.find_one(
            {"content_id": content["content_id"]}
        )

        if exists and "embedding" in exists:
            continue

        text = f"{content['title']} {' '.join(content['genres'])}"
        content["embedding"] = get_embedding(text)
        content["popularity"] = content.get("popularity", 0)

        content_collection.update_one(
            {"content_id": content["content_id"]},
            {"$set": content},
            upsert=True
        )

# ---------------------
# Schemas
# ---------------------
class WatchRequest(BaseModel):
    user_id: int
    content_id: int

# ---------------------
# Watch API
# ---------------------
@app.post("/user/watch")
def add_user_watch(request: WatchRequest):
    valid_ids = [c["content_id"] for c in CONTENTS]
    if request.content_id not in valid_ids:
        raise HTTPException(status_code=400, detail="Invalid content_id")

    already_watched = watch_collection.find_one({
        "user_id": request.user_id,
        "watched": request.content_id
    })

    if not already_watched:
        watch_collection.update_one(
            {"user_id": request.user_id},
            {
                "$addToSet": {"watched": request.content_id},
                "$set": {"last_watched_at": datetime.utcnow()}
            },
            upsert=True
        )

        content_collection.update_one(
            {"content_id": request.content_id},
            {"$inc": {"popularity": 1}},
            upsert=True
        )

    user = watch_collection.find_one(
        {"user_id": request.user_id},
        {"_id": 0, "watched": 1}
    )

    return {
        "user_id": request.user_id,
        "watched": user.get("watched", []),
        "popularity_updated": not bool(already_watched)
    }

# ---------------------
# APIs
# ---------------------
@app.get("/contents")
def contents():
    return CONTENTS

@app.get("/recommend/{user_id}")
def recommend(user_id: int, limit: int = 5):
    user = watch_collection.find_one({"user_id": user_id})
    watched = user.get("watched", []) if user else []
    return get_ai_recommendations(watched, limit)

@app.get("/recommend/reels/{user_id}")
def recommend_reels(user_id: int, limit: int = 5):
    user = watch_collection.find_one({"user_id": user_id})
    watched = user.get("watched", []) if user else []
    return get_ai_recommendations(watched, limit, reels_only=True)

@app.get("/trending")
def trending(limit: int = 10):
    return get_trending_content(limit)

@app.get("/trending/reels")
def trending_reels(limit: int = 10):
    return get_trending_content(limit, reels_only=True)


# ---------------------------
# Content Management APIs
# ---------------------------

@app.delete("/content/delete/{content_id}")
def delete_content(content_id: int):
    """
    Delete content by content_id
    """
    result = content_collection.delete_one({"content_id": content_id})
    if result.deleted_count == 0:
        return {"status": "error", "message": "Content not found"}
    return {"status": "success", "message": f"Content {content_id} deleted"}


@app.post("/content/save")
def save_content(content: Content):
    """
    Add new content or update existing content based on content_id.
    """
    # Generate embedding from title + genres
    text = f"{content.title} {' '.join(content.genres)}"
    embedding = get_embedding(text)

    doc = content.dict()
    doc["embedding"] = embedding

    # Upsert: update if exists, else insert
    result = content_collection.update_one(
        {"content_id": content.content_id},
        {"$set": doc},
        upsert=True
    )

    # Fetch the updated/added document
    saved_doc = content_collection.find_one({"content_id": content.content_id}, {"_id": 0})

    return {"status": "success", "content": saved_doc}


@app.get("/content/check/{content_id}")
def check_content(content_id: int):
    """
    Check if a content exists in the database by content_id.
    """
    content = content_collection.find_one({"content_id": content_id}, {"_id": 0})

    if not content:
        return {"exists": False, "message": f"Content with id {content_id} not found."}

    return {"exists": True, "content": content}



class SortRecommendationRequest(BaseModel):
    user_id: int
    content_ids: list[int]



@app.post("/recommend/sort")
def sort_contents_by_user_recommendation(payload: SortRecommendationRequest):
    user_id = payload.user_id
    content_ids = payload.content_ids

    if not content_ids:
        raise HTTPException(status_code=400, detail="content_ids cannot be empty")

    # ---------------- USER WATCH HISTORY ----------------
    user = watch_collection.find_one({"user_id": user_id})
    watched_ids = user.get("watched", []) if user else []

    # ---------------- FALLBACK: NO WATCH HISTORY ----------------
    if not watched_ids:
        contents = list(
            content_collection.find(
                {"content_id": {"$in": content_ids}},
                {"_id": 0, "embedding": 0}  # remove embedding
            )
            .sort([
                ("popularity", -1),
                ("updated_at", -1)
            ])
        )

        return {
            "user_id": user_id,
            "count": len(contents),
            "data": contents
        }

    # ---------------- FETCH WATCHED CONTENT EMBEDDINGS ----------------
    watched_contents = list(
        content_collection.find(
            {
                "content_id": {"$in": watched_ids},
                "embedding": {"$exists": True}
            },
            {"embedding": 1}
        )
    )

    # If embeddings missing → fallback
    if not watched_contents:
        contents = list(
            content_collection.find(
                {"content_id": {"$in": content_ids}},
                {"_id": 0, "embedding": 0}
            )
            .sort([
                ("popularity", -1),
                ("updated_at", -1)
            ])
        )

        return {
            "user_id": user_id,
            "count": len(contents),
            "data": contents
        }

    # ---------------- USER VECTOR ----------------
    user_embedding = np.mean(
        [c["embedding"] for c in watched_contents],
        axis=0
    ).reshape(1, -1)

    # ---------------- FETCH CANDIDATES ----------------
    candidates = list(
        content_collection.find(
            {
                "content_id": {"$in": content_ids},
                "embedding": {"$exists": True}
            }
        )
    )

    scored_results = []

    for content in candidates:
        ai_score = cosine_similarity(
            user_embedding,
            [content["embedding"]]
        )[0][0]

        popularity_score = content.get("popularity", 0) / 100
        final_score = (0.7 * ai_score) + (0.3 * popularity_score)

        # 🔥 Remove embedding before response
        content.pop("embedding", None)
        content.pop("_id", None)

        content["score"] = round(float(final_score), 4)
        scored_results.append(content)

    # ---------------- FINAL SORT ----------------
    scored_results.sort(
        key=lambda x: (
            x["score"],
            x.get("popularity", 0),
            x.get("updated_at", 0)
        ),
        reverse=True
    )

    return {
        "user_id": user_id,
        "count": len(scored_results),
        "data": scored_results
    }