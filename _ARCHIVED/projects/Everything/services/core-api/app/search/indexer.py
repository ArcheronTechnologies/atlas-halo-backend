from __future__ import annotations

import os
from typing import Any, Dict, Optional

ES_URL = os.getenv("ELASTICSEARCH_URL")

_client = None
_components_mapping = {"mappings": {"properties": {"manufacturerPartNumber": {"type": "keyword"}, "description": {"type": "text", "analyzer": "standard"}, "category": {"type": "keyword"}, "specifications": {"type": "nested"}, "datasheet": {"type": "text"}, "alternativeParts": {"type": "keyword"}, "suppliers": {"type": "nested", "properties": {"name": {"type": "keyword"}, "price": {"type": "double"}, "leadTime": {"type": "integer"}, "availability": {"type": "keyword"}}}, "priceHistory": {"type": "nested", "properties": {"date": {"type": "date"}, "price": {"type": "double"}, "quantity": {"type": "integer"}}}}}}
_documents_mapping = {"mappings": {"properties": {"title": {"type": "text", "analyzer": "standard"}, "content": {"type": "text", "analyzer": "standard"}, "url": {"type": "keyword"}, "domain": {"type": "keyword"}, "publishedAt": {"type": "date"}, "entities": {"type": "keyword"}, "relevanceScore": {"type": "double"}, "classification": {"type": "keyword"}, "extractedComponents": {"type": "nested", "properties": {"partNumber": {"type": "keyword"}, "context": {"type": "text"}}}}}}


def _get_client():
    global _client
    if _client is not None:
        return _client
    if not ES_URL:
        return None
    try:
        from elasticsearch import Elasticsearch

        _client = Elasticsearch(ES_URL)
        return _client
    except Exception:
        return None


def index_component(doc: Dict[str, Any]) -> None:
    client = _get_client()
    if not client:
        return
    try:
        client.index(index="components", id=doc.get("id"), document=doc, refresh=False)
    except Exception:
        # Best-effort; ignore indexing failures in MVP
        pass


def index_document(doc: Dict[str, Any]) -> None:
    client = _get_client()
    if not client:
        return
    try:
        client.index(index="documents", document=doc, refresh=False)
    except Exception:
        pass


def ensure_indices() -> None:
    client = _get_client()
    if not client:
        return
    try:
        if not client.indices.exists(index="components"):
            client.indices.create(index="components", body=_components_mapping)
        if not client.indices.exists(index="documents"):
            client.indices.create(index="documents", body=_documents_mapping)
    except Exception:
        # ignore if indices already exist or ES not reachable
        pass


def delete_component_index(component_id: str) -> None:
    client = _get_client()
    if not client:
        return
    try:
        client.delete(index="components", id=component_id, ignore=[404])
    except Exception:
        pass


def search_components(query: str, limit: int, offset: int) -> list[str]:
    client = _get_client()
    if not client:
        return []
    try:
        res = client.search(
            index="components",
            from_=offset,
            size=limit,
            query={
                "multi_match": {
                    "query": query,
                    "fields": ["manufacturerPartNumber^3", "description", "category"],
                }
            },
            _source=False,
        )
        hits = res.get("hits", {}).get("hits", [])
        return [h.get("_id") for h in hits if h.get("_id")]
    except Exception:
        return []
