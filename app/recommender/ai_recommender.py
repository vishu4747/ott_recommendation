import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.db.connection import content_collection
from app.recommender.trending import get_trending_content

def get_cold_start_content(limit=10, reels_only=False):
    query = {"is_reel": True} if reels_only else {"is_reel": {"$ne": True}}
    return list(
        content_collection.find(
            query,
            {"_id": 0, "embedding": 0}
        )
        .sort("popularity", -1)
        .limit(limit)
    )

# def get_ai_recommendations(watched_ids, limit=5, reels_only=False):
#     if not watched_ids:
#         return get_trending_content(limit, reels_only)

#     watched_contents = list(
#         content_collection.find({"content_id": {"$in": watched_ids}})
#     )

#     if not watched_contents:
#         return get_trending_content(limit, reels_only)

#     # Create user embedding
#     user_embedding = np.mean(
#         [c["embedding"] for c in watched_contents], axis=0
#     ).reshape(1, -1)

#     # Candidate query
#     query = {"content_id": {"$nin": watched_ids}}
#     if reels_only:
#         query["$or"] = [{"is_reel": True}, {"is_reel": "true"}]
#     else:
#         query["$or"] = [{"is_reel": False}, {"is_reel": "false"}, {"is_reel": {"$exists": False}}]

#     candidates = list(content_collection.find(query))

#     results = []
#     for content in candidates:
#         if "embedding" not in content:
#             continue
#         ai_score = cosine_similarity(
#             user_embedding, [content["embedding"]]
#         )[0][0]
#         popularity = content.get("popularity", 0) / 100
#         final_score = (0.7 * ai_score) + (0.3 * popularity)

#         results.append({
#             "content_id": content["content_id"],
#             "title": content["title"],
#             "poster": content["poster"],
#             "genres": content["genres"],
#             "type": content["type"],
#             "score": round(float(final_score), 4)
#         })

#     if not results:
#         return get_trending_content(limit, reels_only)

#     return sorted(results, key=lambda x: x["score"], reverse=True)[:limit]

def get_ai_recommendations(watched_ids, limit=5, reels_only=False):
    # If user has no watched ids, return default trending
    if not watched_ids:
        return get_trending_content(limit, reels_only)

    watched_contents = list(
        content_collection.find({"content_id": {"$in": watched_ids}})
    )

    # Fallback to trending if no embeddings found
    if not watched_contents:
        return get_trending_content(limit, reels_only)

    # User profile embedding
    user_embedding = np.mean(
        [c["embedding"] for c in watched_contents], axis=0
    ).reshape(1, -1)

    # Candidate query
    query = {"content_id": {"$nin": watched_ids}}
    if reels_only:
        query["$or"] = [{"is_reel": True}, {"is_reel": "true"}]
    else:
        query["$or"] = [{"is_reel": False}, {"is_reel": "false"}, {"is_reel": {"$exists": False}}]

    # Sort by _id descending (newer first)
    candidates = list(
        content_collection.find(query).sort("_id", -1)
    )

    results = []
    for content in candidates:
        if "embedding" not in content:
            continue
        ai_score = cosine_similarity(
            user_embedding, [content["embedding"]]
        )[0][0]
        popularity = content.get("popularity", 0) / 100
        final_score = (0.7 * ai_score) + (0.3 * popularity)

        results.append({
            "content_id": content["content_id"],
            "title": content["title"],
            "poster": content["poster"],
            "genres": content["genres"],
            "type": content["type"],
            "score": round(float(final_score), 4)
        })

    # Fallback to trending if AI results empty
    if not results:
        return get_trending_content(limit, reels_only)

    # Sort by final score DESC, but prioritize newer _id first
    results.sort(key=lambda x: (-x["score"], -x["content_id"]))

    return results[:limit]
