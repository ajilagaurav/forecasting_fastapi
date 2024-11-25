from fastapi import FastAPI
from app.routers.forecast_router import router as forecast_router
from app.routers.board_router import router as board_router
from app.routers.main_board_router import router as main_board_router
from app.routers.users_router import router as users_router
from app.routers.roles_router import router as roles_router

app = FastAPI()

# Include routers
app.include_router(board_router)
app.include_router(main_board_router)
app.include_router(users_router)
app.include_router(roles_router)
app.include_router(forecast_router)


@app.get("/")
async def root():
    return {"message": "Welcome to the Forecasting API"}
