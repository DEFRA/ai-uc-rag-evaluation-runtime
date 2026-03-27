import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from logging import getLogger

import uvicorn
from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import JSONResponse

from app.common.mongo import get_mongo_client
from app.common.tracing import TraceIdMiddleware
from app.config import config
from app.evaluation import exceptions
from app.evaluation.queue_listener import listen
from app.evaluation.router import router as evaluation_router
from app.example.router import router as example_router
from app.health.router import router as health_router
from app.truth.router import router as truth_router

logger = getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    # Startup
    client = await get_mongo_client()
    logger.info("MongoDB client connected")
    listener_task = asyncio.create_task(listen())
    yield
    # Shutdown
    listener_task.cancel()
    await asyncio.gather(listener_task, return_exceptions=True)
    if client:
        await client.close()
        logger.info("MongoDB client closed")


app = FastAPI(lifespan=lifespan)


@app.exception_handler(exceptions.EvaluationDataServiceNotConfiguredError)
async def not_configured_handler(
    _: Request, exc: exceptions.EvaluationDataServiceNotConfiguredError
) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(ValueError)
async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(exceptions.EvaluationDataServiceError)
async def data_service_error_handler(
    _: Request, exc: exceptions.EvaluationDataServiceError
) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


# Setup middleware
app.add_middleware(TraceIdMiddleware)

# Setup Routes
app.include_router(health_router)
app.include_router(evaluation_router)
app.include_router(truth_router)
app.include_router(example_router)


def main() -> None:  # pragma: no cover
    uvicorn.run(
        "app.entrypoints.fastapi:app",
        host=config.host,
        port=config.port,
        log_config=config.log_config,
        reload=config.python_env == "development",
    )


if __name__ == "__main__":  # pragma: no cover
    main()
