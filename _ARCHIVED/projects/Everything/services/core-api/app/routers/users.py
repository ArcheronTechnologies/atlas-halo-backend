from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db.session import get_session
from ..repositories.users_repo import UsersRepository
from ..models.users import UserCreate, UserResponse, UsersListResponse, ApiKeyCreateResponse
from ..core.auth import require_scopes


router = APIRouter()


@router.get("/", response_model=UsersListResponse, dependencies=[Depends(require_scopes(["read:users"]))])
def list_users(session: Session = Depends(get_session)):
    repo = UsersRepository(session)
    items = repo.list()
    return UsersListResponse(
        data=[UserResponse(id=u.id, email=u.email, name=u.name, createdAt=u.created_at.isoformat() if u.created_at else None) for u in items]
    )


@router.post("/", response_model=UserResponse, dependencies=[Depends(require_scopes(["write:users"]))])
def create_user(body: UserCreate, session: Session = Depends(get_session)):
    repo = UsersRepository(session)
    u = repo.create(email=body.email, name=body.name)
    return UserResponse(id=u.id, email=u.email, name=u.name, createdAt=u.created_at.isoformat() if u.created_at else None)


@router.post("/{user_id}/roles/{role}", response_model=dict, dependencies=[Depends(require_scopes(["write:users"]))])
def assign_role(user_id: str, role: str, session: Session = Depends(get_session)):
    repo = UsersRepository(session)
    repo.assign_role(user_id, role)
    return {"assigned": True}


@router.post("/{user_id}/api-keys", response_model=ApiKeyCreateResponse, dependencies=[Depends(require_scopes(["write:users"]))])
def create_api_key(user_id: str, session: Session = Depends(get_session)):
    repo = UsersRepository(session)
    rec = repo.create_api_key(user_id)
    return ApiKeyCreateResponse(**rec)

