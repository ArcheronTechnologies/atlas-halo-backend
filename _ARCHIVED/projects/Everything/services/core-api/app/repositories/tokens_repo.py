from __future__ import annotations

import uuid
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import RefreshToken, RevokedJTI


class TokensRepository:
    def __init__(self, session: Session, secret: str):
        self.session = session
        self.secret = secret

    def _hash(self, token: str) -> str:
        return hashlib.sha256((self.secret + ":" + token).encode()).hexdigest()

    def issue_refresh(self, user_id: str, ttl_days: int = 30) -> str:
        token = uuid.uuid4().hex + uuid.uuid4().hex
        rec = RefreshToken(
            id=str(uuid.uuid4()),
            user_id=user_id,
            token_hash=self._hash(token),
            expires_at=datetime.now(timezone.utc) + timedelta(days=ttl_days),
            revoked=False,
        )
        self.session.add(rec)
        self.session.commit()
        return token

    def rotate_refresh(self, old_token: str, user_id: str) -> Optional[str]:
        rec = self._find_refresh(old_token)
        if not rec or rec.revoked or rec.user_id != user_id or rec.expires_at < datetime.now(timezone.utc):
            return None
        rec.revoked = True
        self.session.add(rec)
        self.session.commit()
        return self.issue_refresh(user_id)

    def _find_refresh(self, token: str) -> Optional[RefreshToken]:
        hashed = self._hash(token)
        return self.session.execute(select(RefreshToken).where(RefreshToken.token_hash == hashed)).scalar_one_or_none()

    def revoke_refresh(self, token: str) -> bool:
        rec = self._find_refresh(token)
        if not rec:
            return False
        rec.revoked = True
        self.session.add(rec)
        self.session.commit()
        return True

    def revoke_jti(self, jti: str) -> None:
        self.session.merge(RevokedJTI(jti=jti))
        self.session.commit()

    def is_jti_revoked(self, jti: str) -> bool:
        return self.session.get(RevokedJTI, jti) is not None

