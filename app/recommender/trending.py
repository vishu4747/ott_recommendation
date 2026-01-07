from app.db.connection import content_collection

# def get_trending_content(limit=10, reels_only=False):
#     query = {}

#     if reels_only:
#         # Match string "true" or boolean True just in case
#         query["$or"] = [
#             {"is_reel": True},
#             {"is_reel": "true"}
#         ]
#     else:
#         # Movies/Series: is_reel False, "false", or missing
#         query["$or"] = [
#             {"is_reel": False},
#             {"is_reel": "false"},
#             {"is_reel": {"$exists": False}}
#         ]

#     return list(
#         content_collection.find(
#             query,
#             {"_id": 0, "embedding": 0}
#         )
#         .sort("popularity", -1)
#         .limit(limit)
#     )

def get_trending_content(limit=10, reels_only=False):
    query = {}

    if reels_only:
        query["$or"] = [{"is_reel": True}, {"is_reel": "true"}]
    else:
        query["$or"] = [{"is_reel": False}, {"is_reel": "false"}, {"is_reel": {"$exists": False}}]

    # Sort by popularity DESC and _id DESC (newer first)
    return list(
        content_collection.find(
            query,
            {"_id": 0, "embedding": 0}
        )
        .sort([("popularity", -1), ("_id", -1)])
        .limit(limit)
    )


