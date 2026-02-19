"""
Database Engine & Session Management
=============================================================================
CONCEPT: Async SQLAlchemy with Connection Pooling

SQLAlchemy is Python's most popular ORM (Object-Relational Mapper).
It lets you interact with databases using Python objects instead of raw SQL.

KEY CONCEPTS:
  1. Engine — The connection factory. Creates and manages DB connections.
  2. Session — A "workspace" for DB operations. Groups queries into transactions.
  3. Connection Pool — Reuses DB connections instead of creating new ones per request.
     This is critical for performance:
       - Creating a new TCP connection to PostgreSQL: ~5-10ms
       - Reusing a pooled connection: ~0.1ms (50-100x faster)
  4. Async — We use asyncpg driver for non-blocking I/O.
     While waiting for a DB query, the server can handle other requests.

POOL SETTINGS EXPLAINED:
  - pool_size=20: Keep 20 connections ready at all times
  - max_overflow=10: Allow up to 10 extra connections during traffic spikes
  - pool_pre_ping=True: Check if a connection is alive before using it
    (prevents "connection closed" errors after idle periods)
  - pool_recycle=3600: Replace connections after 1 hour
    (prevents stale connections from accumulating)
=============================================================================
"""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from src.config import settings

# Create the async engine with connection pooling
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,          # Log SQL queries in debug mode (very useful for learning!)
    pool_size=20,                 # Maintain 20 persistent connections
    max_overflow=10,              # Allow 10 extra connections under load
    pool_pre_ping=True,           # Verify connections before use
    pool_recycle=3600,            # Recycle connections every hour
)

# Session factory — creates new AsyncSession instances
# CONCEPT: async_sessionmaker is a factory pattern. Instead of creating sessions
# manually, we configure the factory once and call it to get new sessions.
# expire_on_commit=False means objects remain usable after commit
# (without this, accessing attributes after commit would trigger a lazy load error)
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# Base class for all ORM models
# CONCEPT: DeclarativeBase is the foundation of SQLAlchemy's ORM.
# All model classes inherit from this, and SQLAlchemy uses it to:
#   1. Track all table definitions
#   2. Generate CREATE TABLE statements
#   3. Map Python classes ↔ database tables
class Base(DeclarativeBase):
    pass


async def get_db_session() -> AsyncSession:
    """
    FastAPI dependency that provides a database session per request.

    CONCEPT: Dependency Injection
    FastAPI calls this function automatically when a route needs a DB session.
    The `async with` ensures the session is properly closed after the request,
    even if an error occurs (like a try/finally block).

    Usage in a route:
        @app.get("/employees")
        async def list_employees(db: AsyncSession = Depends(get_db_session)):
            result = await db.execute(select(Employee))
            return result.scalars().all()
    """
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()
