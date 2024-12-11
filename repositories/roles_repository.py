class RolesRepository:
    def __init__(self):
        self.roles = []  # In-memory list to store roles
        self.role_assignments = []  # Track role-to-board assignments
        self.user_roles = []  # Track user-to-role assignments

    def create_role(self, role):
        if any(r["id"] == role.id for r in self.roles):
            raise ValueError("Role with this ID already exists")
        self.roles.append(role.dict())
        return role

    def assign_board_to_role(self, role_id, board_id):
        if not any(r["id"] == role_id for r in self.roles):
            raise ValueError("Role not found")
        self.role_assignments.append({"role_id": role_id, "board_id": board_id})

    def get_boards_for_role(self, role_id):
        return [a["board_id"] for a in self.role_assignments if a["role_id"] == role_id]

    def assign_role_to_user(self, user_id, role_id):
        if not any(r["id"] == role_id for r in self.roles):
            raise ValueError("Role not found")
        self.user_roles.append({"user_id": user_id, "role_id": role_id})

    def get_roles_for_user(self, user_id):
        return [ur["role_id"] for ur in self.user_roles if ur["user_id"] == user_id]

    def get_boards_for_user(self, user_id):
        user_roles = self.get_roles_for_user(user_id)
        boards = []
        for role_id in user_roles:
            boards.extend(self.get_boards_for_role(role_id))
        return list(set(boards))  # Remove duplicates
