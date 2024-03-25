# -*- coding: utf-8 -*-
from functools import wraps
import gc
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi_async_sqlalchemy import SQLAlchemyMiddleware
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_limiter import FastAPILimiter
from fastapi_pagination import add_pagination
from jose import jwt
from langchain.cache import RedisCache
from langchain.globals import set_llm_cache
from pydantic import ValidationError
from starlette.middleware.cors import CORSMiddleware

from app.api.deps import get_redis_client, get_redis_client_sync
from app.api.v1.api import api_router as api_router_v1
from app.core.config import settings, yaml_configs
from app.core.fastapi import FastAPIWithInternalModels
from app.utils.config_loader import load_agent_config, load_ingestion_configs
from app.utils.config_logging import LogConfig
from app.utils.fastapi_globals import GlobalsMiddleware, g


async def user_id_identifier(request: Request) -> str:
    """Identify the user from the request."""
    if request.scope["type"] == "http":
        # Retrieve the Authorization header from the request
        auth_header = request.headers.get("Authorization")

        if auth_header is not None:
            # Check that the header is in the correct format
            header_parts = auth_header.split()
            if len(header_parts) == 2 and header_parts[0].lower() == "bearer":
                token = header_parts[1]
                try:
                    payload = jwt.decode(
                        token,
                        settings.SECRET_KEY,
                        algorithms=["HS256"],
                    )
                except (
                    jwt.JWTError,
                    ValidationError,
                ):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Could not validate credentials",
                    )
                user_id = payload["sub"]
                print(
                    "here2",
                    user_id,
                )
                return user_id

    if request.scope["type"] == "websocket":
        return request.scope["path"]

    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0]

    ip = request.client.host if request.client else ""
    return ip + ":" + request.scope["path"]


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Start up and shutdown tasks."""
    # startup
    yaml_configs["agent_config"] = load_agent_config()
    yaml_configs["ingestion_config"] = load_ingestion_configs()

    redis_client = await get_redis_client()

    if settings.ENABLE_LLM_CACHE:
        set_llm_cache(RedisCache(redis_=get_redis_client_sync()))

    FastAPICache.init(
        RedisBackend(redis_client),
        prefix="fastapi-cache",
    )
    await FastAPILimiter.init(
        redis_client,
        identifier=user_id_identifier,
    )

    logging.info("Start up FastAPI [Full dev mode]")
    yield

    # shutdown
    await FastAPICache.clear()
    await FastAPILimiter.close()
    g.cleanup()
    gc.collect()
    yaml_configs.clear()

from logging.config import dictConfig
dictConfig(LogConfig().model_dump())

logger = logging.getLogger(__name__)

# Core Application Instance
app = FastAPIWithInternalModels(
    title=settings.PROJECT_NAME,
    version=settings.API_VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    lifespan=lifespan,
)


app.add_middleware(
    SQLAlchemyMiddleware,
    db_url=settings.ASYNC_DATABASE_URI,
    engine_args={
        "echo": False,
        "pool_pre_ping": True,
        "pool_size": settings.POOL_SIZE,
        "max_overflow": 64,
    },
)
app.add_middleware(GlobalsMiddleware)

# Set all CORS origins enabled
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
    logging.error(f"{request}: {exc_str}")
    content = {"status_code": 10422, "message": exc_str, "data": None}
    return JSONResponse(content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)


@app.get("/")
async def root() -> Dict[str, str]:
    """An example "Hello world" FastAPI route."""
    from opentelemetry import trace
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span('foo'):
        import time
        logger.debug("debug")
        logger.info("info")
        time.sleep(.1)

    return {"message": "FastAPI backend"}


# Add Routers
app.include_router(
    api_router_v1,
    prefix=settings.API_V1_STR,
)
add_pagination(app)


if settings.ENABLE_TRACE:

    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    FastAPIInstrumentor.instrument_app(app)


    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    trace.set_tracer_provider(
    TracerProvider(
            resource=Resource.create({SERVICE_NAME: "agentkit"})
        )
    )
    tracer = trace.get_tracer(__name__)

    # create a JaegerExporter
    jaeger_exporter = OTLPSpanExporter(endpoint="http://jaeger:4318/v1/traces")

    # Create a BatchSpanProcessor and add the exporter to it
    span_processor = BatchSpanProcessor(jaeger_exporter)

    # add to the tracer
    trace.get_tracer_provider().add_span_processor(span_processor)

