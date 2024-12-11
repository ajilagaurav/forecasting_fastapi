# app/routers/roles_router.py
from fastapi import APIRouter, HTTPException
from app.models.roles import Role
from app.models.role_assignment import RoleAssignment
from app.repositories import roles_repository

router = APIRouter(prefix="/roles", tags=["Roles"])

@router.post("/", response_model=Role)
async def create_role(role: Role):
    created_role = roles_repository.create_role(role)
    return created_role

@router.post("/assign-board")
async def assign_board_to_role(assignment: RoleAssignment):
    roles_repository.assign_board_to_role(assignment.role_id, assignment.board_id)
    return {"detail": "Board assigned to role successfully"}

@router.post("/assign-role-to-user")
async def assign_role_to_user(user_id: int, role_id: int):
    roles_repository.assign_role_to_user(user_id, role_id)
    return {"detail": "Role assigned to user successfully"}

@router.get("/{role_id}/boards", response_model=list[int])
async def get_boards_for_role(role_id: int):
    boards = roles_repository.get_boards_for_role(role_id)
    if not boards:
        raise HTTPException(status_code=404, detail="No boards assigned to this role")
    return boards
