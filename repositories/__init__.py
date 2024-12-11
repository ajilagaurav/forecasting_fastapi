# app/repositories/__init__.py
from app.repositories.users_repository import UsersRepository
from app.repositories.roles_repository import RolesRepository

users_repository = UsersRepository()
roles_repository = RolesRepository()
