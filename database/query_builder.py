"""
Query builder utilities to safely apply projection minimization and retention constraints
to basic SELECT queries.

Scope: handles simple SELECT ... FROM <table> [WHERE ...] [ORDER BY ...] [LIMIT ...]
with a single base table in the FROM clause. For more complex queries, returns the SQL unchanged.
"""

from __future__ import annotations
from typing import List, Tuple
import re
from ..compliance.privacy_framework import ProcessingPurpose


_MIN_FIELDS: Dict[ProcessingPurpose, Dict[str, List[str]]] = {
    ProcessingPurpose.INVESTIGATION: {
        "persons": ["id", "first_name", "last_name", "date_of_birth", "address", "phone"],
        "incidents": ["id", "incident_type", "timestamp", "location", "severity_level"],
        "criminal_records": ["id", "subject_id", "charges", "conviction_date", "status"],
    },
    ProcessingPurpose.PREVENTION: {
        "persons": ["id", "risk_score", "last_known_location", "alert_flags"],
        "incidents": ["id", "incident_type", "location_general", "risk_indicators"],
        "criminal_records": ["id", "risk_category", "recidivism_score"],
    },
    ProcessingPurpose.PUBLIC_SAFETY: {
        "persons": ["id", "emergency_contacts", "medical_alerts", "location_current"],
        "incidents": ["id", "incident_type", "location", "emergency_level"],
        "criminal_records": ["id", "public_safety_flags", "restrictions"],
    },
    ProcessingPurpose.EMERGENCY_RESPONSE: {
        "persons": ["id", "emergency_contacts", "medical_info", "current_location"],
        "incidents": ["id", "emergency_type", "location", "response_required"],
        "criminal_records": [],
    },
}


_RETENTION_EXTRA = {
    # table -> extra temporal constraints
    "location_tracking": "timestamp > NOW() - INTERVAL '1 year'",
    "surveillance_footage": "timestamp > NOW() - INTERVAL '30 days'",
}


def _split_clauses(sql: str) -> tuple[str, str, str, str]:
    """Return (select_part, from_part, where_part, tail_part) for simple queries.
    tail_part contains ORDER BY / LIMIT if present.
    """
    sql_stripped = sql.strip()
    m = re.match(r"(?is)^\s*select\s+(.*?)\s+from\s+(.*)$", sql_stripped)
    if not m:
        return sql, "", "", ""
    select_part = m.group(1)
    rest = m.group(2)
    # Split where
    where_part = ""
    tail = ""
    where_match = re.search(r"(?is)\bwhere\b", rest)
    if where_match:
        from_part = rest[: where_match.start()].strip()
        where_and_tail = rest[where_match.end():].strip()
    else:
        # no where
        order_match = re.search(r"(?is)\b(order\s+by|limit)\b", rest)
        if order_match:
            from_part = rest[: order_match.start()].strip()
            tail = rest[order_match.start():].strip()
            return select_part, from_part, where_part, tail
        else:
            from_part = rest.strip()
            return select_part, from_part, where_part, tail
    # Separate tail from where clause
    order_match = re.search(r"(?is)\b(order\s+by|limit)\b", where_and_tail)
    if order_match:
        where_part = where_and_tail[: order_match.start()].strip()
        tail = where_and_tail[order_match.start():].strip()
    else:
        where_part = where_and_tail
    return select_part, from_part, where_part, tail


def _single_base_table(from_part: str) -> str | None:
    # accept formats: table, schema.table, table alias, table AS alias
    # Reject joins / commas
    if re.search(r"\bjoin\b|,", from_part, re.IGNORECASE):
        return None
    # first token is table (maybe schema.table)
    tokens = from_part.split()
    if not tokens:
        return None
    table = tokens[0]
    # strip quotes
    table = table.strip('"')
    return table


def build_select_sql(sql: str, tables: List[str], purpose: ProcessingPurpose) -> str:
    sel, frm, where, tail = _split_clauses(sql)
    if not frm:
        return sql
    base_table = _single_base_table(frm)
    # Only handle simple SELECT * cases with one base table
    if not base_table:
        # still add retention constraints if possible (simple from clause)
        return _apply_retention_only(sel, frm, where, tail)
    # Projection minimization
    if re.match(r"(?is)^\s*\*\s*$", sel):
        purpose_fields = _MIN_FIELDS.get(purpose, {})
        allowed = purpose_fields.get(base_table, [])
        if allowed:
            projection = ", ".join(f"{base_table}.{f}" for f in allowed)
        else:
            projection = sel  # leave as-is if no mapping
    else:
        projection = sel
    # Retention constraints
    constraints = [f"{base_table}.retention_status != 'deleted'"]
    extra = _RETENTION_EXTRA.get(base_table)
    if extra:
        constraints.append(f"{base_table}.{extra}") if not extra.startswith(base_table + ".") else constraints.append(extra)

    # Merge WHERE
    where_clause = where.strip()
    constraint_expr = " AND ".join(constraints)
    if where_clause:
        where_final = f"({where_clause}) AND {constraint_expr}"
    else:
        where_final = constraint_expr

    # Recompose
    out = f"SELECT {projection} FROM {frm}"
    if where_final:
        out += f" WHERE {where_final}"
    if tail:
        out += f" {tail.strip()}"
    return out


def _apply_retention_only(sel: str, frm: str, where: str, tail: str) -> str:
    # Attempt to add 'retention_status != 'deleted'' for each table token (best effort)
    tables = []
    for t in re.split(r"\s+", frm):
        tok = t.strip(',')
        if tok and tok.lower() not in {"as", "on", "inner", "left", "right", "full", "outer", "join"}:
            tables.append(tok.strip('"'))
    constraints = [f"{t}.retention_status != 'deleted'" for t in tables]
    constraint_expr = " AND ".join(constraints)
    where_clause = where.strip()
    if where_clause:
        where_final = f"({where_clause}) AND {constraint_expr}"
    else:
        where_final = constraint_expr
    out = f"SELECT {sel} FROM {frm}"
    if where_final:
        out += f" WHERE {where_final}"
    if tail:
        out += f" {tail.strip()}"
    return out

