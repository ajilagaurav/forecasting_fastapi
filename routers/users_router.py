# app/routers/users_router.py
from fastapi import APIRouter, HTTPException
from app.models.users import User
from app.repositories import users_repository, roles_repository

router = APIRouter(prefix="/users", tags=["Users"])

@router.post("/", response_model=User)
async def create_user(user: User):
    created_user = users_repository.create_user(user)
    return created_user

@router.get("/", response_model=list[User])
async def get_users():
    return users_repository.get_users()

@router.get("/{user_id}", response_model=User)
async def get_user(user_id: int):
    user = users_repository.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.get("/{user_id}/roles", response_model=list[int])
async def get_roles_for_user(user_id: int):
    roles = roles_repository.get_roles_for_user(user_id)
    if not roles:
        raise HTTPException(status_code=404, detail="No roles assigned to this user")
    return roles

@router.get("/{user_id}/boards", response_model=list[int])
async def get_boards_for_user(user_id: int):
    boards = roles_repository.get_boards_for_user(user_id)
    if not boards:
        raise HTTPException(status_code=404, detail="No boards accessible for this user")
    return boards
