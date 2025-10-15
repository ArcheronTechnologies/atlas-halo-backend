from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class Company(Base):
    __tablename__ = "companies"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    type: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    website: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class Component(Base):
    __tablename__ = "components"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    manufacturer_part_number: Mapped[str] = mapped_column(String(100), index=True)
    manufacturer_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("companies.id"), nullable=True)
    category: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    subcategory: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    datasheet_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    lifecycle_status: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    rohs_compliant: Mapped[Optional[bool]] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class RFQ(Base):
    __tablename__ = "rfqs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    customer_id: Mapped[str] = mapped_column(String(36), ForeignKey("companies.id"))
    rfq_number: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="open")
    required_date: Mapped[Optional[datetime]] = mapped_column(Date, nullable=True)
    total_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    currency: Mapped[Optional[str]] = mapped_column(String(3), default="USD")
    source: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    items: Mapped[list[RFQItem]] = relationship("RFQItem", back_populates="rfq", cascade="all, delete-orphan")


class RFQItem(Base):
    __tablename__ = "rfq_items"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    rfq_id: Mapped[str] = mapped_column(String(36), ForeignKey("rfqs.id"))
    component_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("components.id"), nullable=True)
    customer_part_number: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    quantity: Mapped[int] = mapped_column(Integer)
    target_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    lead_time_weeks: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    packaging: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    date_code_requirement: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    rfq: Mapped[RFQ] = relationship("RFQ", back_populates="items")


class RFQQuote(Base):
    __tablename__ = "rfq_quotes"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    rfq_id: Mapped[str] = mapped_column(String(36), ForeignKey("rfqs.id"))
    payload: Mapped[str] = mapped_column(Text)
    is_winner: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class PriceHistory(Base):
    __tablename__ = "price_history"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    component_id: Mapped[str] = mapped_column(String(36), ForeignKey("components.id"))
    supplier_id: Mapped[str] = mapped_column(String(36), ForeignKey("companies.id"))
    quantity_break: Mapped[int] = mapped_column(Integer)
    unit_price: Mapped[float] = mapped_column(Float)
    currency: Mapped[str] = mapped_column(String(3), default="USD")
    valid_from: Mapped[Optional[datetime]] = mapped_column(Date, nullable=True)
    valid_until: Mapped[Optional[datetime]] = mapped_column(Date, nullable=True)
    source_type: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    source_reference: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


# Auth tokens and revocation
class RefreshToken(Base):
    __tablename__ = "refresh_tokens"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(36))
    token_hash: Mapped[str] = mapped_column(String(128), index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    revoked: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class RevokedJTI(Base):
    __tablename__ = "revoked_jtis"
    jti: Mapped[str] = mapped_column(String(64), primary_key=True)
    revoked_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True)
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class Role(Base):
    __tablename__ = "roles"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(50), unique=True)


class UserRole(Base):
    __tablename__ = "user_roles"
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), primary_key=True)
    role_id: Mapped[str] = mapped_column(String(36), ForeignKey("roles.id"), primary_key=True)


class ApiKey(Base):
    __tablename__ = "api_keys"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"))
    key_hash: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    active: Mapped[bool] = mapped_column(default=True)


class RolePermission(Base):
    __tablename__ = "role_permissions"
    role_id: Mapped[str] = mapped_column(String(36), ForeignKey("roles.id"), primary_key=True)
    permission: Mapped[str] = mapped_column(String(100), primary_key=True)


class Inventory(Base):
    __tablename__ = "inventory"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    component_id: Mapped[str] = mapped_column(String(36), ForeignKey("components.id"))
    location: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    quantity_available: Mapped[int] = mapped_column(Integer, default=0)
    quantity_reserved: Mapped[int] = mapped_column(Integer, default=0)
    cost_per_unit: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    date_code: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    lot_code: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    expiry_date: Mapped[Optional[datetime]] = mapped_column(Date, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class PurchaseOrder(Base):
    __tablename__ = "purchase_orders"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    supplier_id: Mapped[str] = mapped_column(String(36), ForeignKey("companies.id"))
    po_number: Mapped[str] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(20), default="draft")
    total_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    currency: Mapped[str] = mapped_column(String(3), default="USD")
    payment_terms: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    delivery_address: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class POItem(Base):
    __tablename__ = "po_items"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    po_id: Mapped[str] = mapped_column(String(36), ForeignKey("purchase_orders.id"))
    component_id: Mapped[str] = mapped_column(String(36), ForeignKey("components.id"))
    quantity: Mapped[int] = mapped_column(Integer)
    unit_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    lead_time_weeks: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    manufacturer_lot_code: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    date_code: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    packaging: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class ProcessedEmail(Base):
    __tablename__ = "processed_emails"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    message_id: Mapped[str] = mapped_column(String(255), unique=True)
    sender_email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    subject: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    body_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    body_html: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    received_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    classification: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    confidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    extracted_entities: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    attachments: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    processed_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class MarketData(Base):
    __tablename__ = "market_data"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    component_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("components.id"), nullable=True)
    data_source: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    metric_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    metric_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    unit: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    collection_date: Mapped[Optional[datetime]] = mapped_column(Date, nullable=True)
    source_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    confidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    entity_type: Mapped[str] = mapped_column(String(50))
    entity_id: Mapped[str] = mapped_column(String(36))
    action: Mapped[str] = mapped_column(String(50))
    old_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    new_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
