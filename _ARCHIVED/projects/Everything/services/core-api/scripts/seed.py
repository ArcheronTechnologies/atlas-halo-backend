import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.db.base import Base
from app.db.models import Company
from app.repositories.components_repo import ComponentsRepository
from app.repositories.companies_repo import CompaniesRepository


def main():
    engine = create_engine(settings.database_url, future=True, connect_args={"check_same_thread": False} if settings.database_url.startswith("sqlite") else {})
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, future=True)
    session = SessionLocal()
    try:
        companies = CompaniesRepository(session)
        comps = ComponentsRepository(session)

        # Seed baseline entities if empty
        existing = companies.list()
        if not existing:
            # Companies
            cust = companies.create(name="Acme Corp", type_="customer", website="https://acme.example")
            cust2 = companies.create(name="Globex Inc", type_="customer", website="https://globex.example")
            mfg_st = companies.create(name="STMicroelectronics", type_="manufacturer", website="https://www.st.com")
            mfg_ti = companies.create(name="Texas Instruments", type_="manufacturer", website="https://www.ti.com")
            supp_dk = companies.create(name="Digi-Key", type_="supplier", website="https://www.digikey.com")
            supp_mou = companies.create(name="Mouser", type_="supplier", website="https://www.mouser.com")

            # Components
            c1 = comps.create(
                {
                    "manufacturerPartNumber": "STM32F429ZIT6",
                    "manufacturer": {"id": mfg_st.id},
                    "category": "Microcontrollers",
                    "description": "ARM Cortex-M4 MCU with FPU",
                    "lifecycleStatus": "active",
                    "rohsCompliant": True,
                    "datasheet": "https://www.st.com/resource/en/datasheet/stm32f429zit6.pdf",
                }
            )
            c2 = comps.create(
                {
                    "manufacturerPartNumber": "TPS7A4700RGWT",
                    "manufacturer": {"id": mfg_ti.id},
                    "category": "Power Management",
                    "description": "36-V, 1-A, ultra-low-noise LDO regulator",
                    "lifecycleStatus": "active",
                    "rohsCompliant": True,
                    "datasheet": "https://www.ti.com/lit/ds/symlink/tps7a4700.pdf",
                }
            )

            # RFQs minimal examples
            from app.repositories.rfqs_repo import RFQsRepository
            from app.db.models import PriceHistory
            import uuid as _uuid
            from datetime import datetime, timedelta

            rfqs = RFQsRepository(session)
            rfqs.create(
                {
                    "customerId": cust.id,
                    "rfqNumber": "RFQ-2024-001",
                    "items": [
                        {"componentId": c1.id, "quantity": 1000},
                        {"componentId": c2.id, "quantity": 500},
                    ],
                    "source": "seed",
                }
            )
            rfqs.create(
                {
                    "customerId": cust2.id,
                    "rfqNumber": "RFQ-2024-002",
                    "items": [{"componentId": c2.id, "quantity": 200}],
                    "source": "seed",
                }
            )
            # Price history samples
            now = datetime.utcnow()
            ph = [
                PriceHistory(
                    id=str(_uuid.uuid4()),
                    component_id=c1.id,
                    supplier_id=supp_dk.id,
                    quantity_break=1,
                    unit_price=15.99,
                    currency="USD",
                    created_at=now,
                ),
                PriceHistory(
                    id=str(_uuid.uuid4()),
                    component_id=c1.id,
                    supplier_id=supp_dk.id,
                    quantity_break=100,
                    unit_price=14.50,
                    currency="USD",
                    created_at=now,
                ),
                PriceHistory(
                    id=str(_uuid.uuid4()),
                    component_id=c1.id,
                    supplier_id=supp_mou.id,
                    quantity_break=1,
                    unit_price=16.10,
                    currency="USD",
                    created_at=now - timedelta(days=1),
                ),
            ]
            session.add_all(ph)
            session.commit()
            # Seed admin role permissions
            from app.repositories.users_repo import UsersRepository
            ur = UsersRepository(session)
            for perm in [
                "read:components","write:components","read:rfqs","write:rfqs",
                "read:companies","write:companies","read:users","write:users",
                "read:inventory","write:inventory","read:purchase_orders","write:purchase_orders",
                "read:audit","read:graph","write:graph"
            ]:
                ur.add_permission_to_role("admin", perm)
            print("Seeded companies, components, RFQs, price history, and admin role permissions.")
        else:
            print("Database already has companies; skipping seed.")
    finally:
        session.close()


if __name__ == "__main__":
    main()
