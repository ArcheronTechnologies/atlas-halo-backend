import argparse
from typing import Iterator

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.db.models import Component as ComponentORM
from app.search.indexer import ensure_indices, index_component


def _iter_components(session: Session, page_size: int = 200) -> Iterator[ComponentORM]:
    offset = 0
    while True:
        rows = session.execute(select(ComponentORM).offset(offset).limit(page_size)).scalars().all()
        if not rows:
            break
        for r in rows:
            yield r
        offset += page_size


def cmd_ensure() -> None:
    ensure_indices()
    print("Ensured Elasticsearch indices (components, documents)")


def cmd_reindex_components() -> None:
    session = SessionLocal()
    try:
        ensure_indices()
        count = 0
        for c in _iter_components(session):
            doc = {
                "id": c.id,
                "manufacturerPartNumber": c.manufacturer_part_number,
                "description": c.description,
                "category": c.category,
                "datasheet": c.datasheet_url,
            }
            index_component(doc)
            count += 1
        print(f"Reindexed {count} components")
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description="SCIP Elasticsearch utilities")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("ensure", help="Ensure indices exist")
    sub.add_parser("reindex-components", help="Reindex all components")
    args = parser.parse_args()
    if args.cmd == "ensure":
        cmd_ensure()
    elif args.cmd == "reindex-components":
        cmd_reindex_components()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

