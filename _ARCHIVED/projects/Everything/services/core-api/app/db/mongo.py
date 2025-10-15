from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase


MONGODB_URL = os.getenv("MONGODB_URL")
MONGODB_DB = os.getenv("MONGODB_DB", "scip")


@lru_cache(maxsize=1)
def get_mongo_client() -> Optional[AsyncIOMotorClient]:
    if not MONGODB_URL:
        return None
    return AsyncIOMotorClient(MONGODB_URL)


def get_mongo_db() -> Optional[AsyncIOMotorDatabase]:
    client = get_mongo_client()
    if not client:
        return None
    return client[MONGODB_DB]


async def ensure_mongo_indexes() -> None:
    db = get_mongo_db()
    if not db:
        return
    try:
        await db.get_collection("emails").create_index("messageId", unique=True)
        await db.get_collection("web_data").create_index("url", unique=True)
    except Exception:
        # best-effort
        pass
