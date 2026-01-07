import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.db.connection import content_collection

def get_ai_recommendations(watched_ids, limit=5, reels_only=False):
    if not watched_ids:
        return []

    watched_contents = list(
        content_collection.find({"content_id": {"$in": watched_ids}})
    )

    if not watched_contents:
        return []

    # Create user profile embedding
    user_embedding = np.mean(
        [c["embedding"] for c in watched_contents], axis=0
    ).reshape(1, -1)

    # Candidate contents
    query = {
        "content_id": {"$nin": watched_ids}
    }
    if reels_only:
        query["is_reel"] = True
    else:
        query["is_reel"] = {"$ne": True}

    candidates = list(content_collection.find(query))

    results = []
    for content in candidates:
        score = cosine_similarity(
            user_embedding, [content["embedding"]]
        )[0][0]

        results.append({
            "content_id": content["content_id"],
            "title": content["title"],
            "poster": content["poster"],
            "genres": content["genres"],
            "type": content["type"],
            "score": float(score)
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)[:limit]
