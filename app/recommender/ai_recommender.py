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


def get_ai_recommendations(
    watched_ids,
    limit=5,
    page=1,
    reels_only=False
):
    # Normalize page
    page = max(page, 1)
    offset = (page - 1) * limit

    # 🔹 Cold start
    if not watched_ids:
        data = get_trending_content(limit + 1, reels_only)
        return {
            "page": page,
            "next_page": page + 1 if len(data) > limit else 0,
            "data": data[:limit]
        }

    watched_contents = list(
        content_collection.find(
            {
                "content_id": {"$in": watched_ids},
                "embedding": {"$exists": True}
            }
        )
    )

    # 🔹 No embeddings → trending
    if not watched_contents:
        data = get_trending_content(limit + 1, reels_only)
        return {
            "page": page,
            "next_page": page + 1 if len(data) > limit else 0,
            "data": data[:limit]
        }

    # 🔹 User embedding
    user_embedding = np.mean(
        [c["embedding"] for c in watched_contents],
        axis=0
    ).reshape(1, -1)

    # 🔹 Candidate query
    query = {
        "content_id": {"$nin": watched_ids},
        "embedding": {"$exists": True}
    }

    if reels_only:
        query["$or"] = [{"is_reel": True}, {"is_reel": "true"}]
    else:
        query["$or"] = [
            {"is_reel": False},
            {"is_reel": "false"},
            {"is_reel": {"$exists": False}}
        ]

    candidates = list(
        content_collection.find(query).sort("_id", -1)
    )

    results = []

    for content in candidates:
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

    # 🔹 AI empty → trending
    if not results:
        data = get_trending_content(limit + 1, reels_only)
        return {
            "page": page,
            "next_page": page + 1 if len(data) > limit else 0,
            "data": data[:limit]
        }

    # 🔹 Sort by relevance, then freshness
    results.sort(key=lambda x: (-x["score"], -x["content_id"]))

    # 🔹 Pagination slice
    paginated = results[offset : offset + limit + 1]

    return {
        "page": page,
        "next_page": page + 1 if len(paginated) > limit else 0,
        "data": paginated[:limit]
    }
