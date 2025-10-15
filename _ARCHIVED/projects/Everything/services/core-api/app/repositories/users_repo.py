from __future__ import annotations

import uuid
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import User, Role, UserRole, ApiKey, RolePermission
from ..core.security import generate_api_key, hash_api_key


class UsersRepository:
    def __init__(self, session: Session):
        self.session = session

    def list(self) -> List[User]:
        return self.session.execute(select(User).order_by(User.created_at.desc())).scalars().all()

    def create(self, email: str, name: Optional[str] = None) -> User:
        u = User(id=str(uuid.uuid4()), email=email, name=name)
        self.session.add(u)
        self.session.commit()
        self.session.refresh(u)
        return u

    def assign_role(self, user_id: str, role_name: str) -> None:
        role = self.session.execute(select(Role).where(Role.name == role_name)).scalar_one_or_none()
        if not role:
            role = Role(id=str(uuid.uuid4()), name=role_name)
            self.session.add(role)
            self.session.flush()
        link = UserRole(user_id=user_id, role_id=role.id)
        self.session.merge(link)
        self.session.commit()

    def create_api_key(self, user_id: str) -> dict:
        key_id, api_key = generate_api_key()
        hashed = hash_api_key(api_key)
        rec = ApiKey(id=str(uuid.uuid4()), user_id=user_id, key_hash=hashed, active=True)
        self.session.add(rec)
        self.session.commit()
        return {"id": rec.id, "key": api_key}

    def verify_api_key(self, api_key: str) -> Optional[User]:
        hashed = hash_api_key(api_key)
        rec = self.session.execute(select(ApiKey, User).join(User, User.id == ApiKey.user_id).where(ApiKey.key_hash == hashed, ApiKey.active == True)).first()  # noqa: E712
        if rec:
            _, user = rec
            return user
        return None

    def add_permission_to_role(self, role_name: str, permission: str) -> None:
        role = self.session.execute(select(Role).where(Role.name == role_name)).scalar_one_or_none()
        if not role:
            role = Role(id=str(uuid.uuid4()), name=role_name)
            self.session.add(role)
            self.session.flush()
        link = RolePermission(role_id=role.id, permission=permission)
        self.session.merge(link)
        self.session.commit()

    def permissions_for_user(self, user_id: str) -> list[str]:
        roles = self.session.execute(select(Role).join(UserRole, UserRole.role_id == Role.id).where(UserRole.user_id == user_id)).scalars().all()
        perms: set[str] = set()
        for r in roles:
            rows = self.session.execute(select(RolePermission).where(RolePermission.role_id == r.id)).scalars().all()
            perms.update([rp.permission for rp in rows])
        return sorted(perms)
