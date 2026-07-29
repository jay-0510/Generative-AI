"""
config/db_init.py
==================

WHY THIS FILE EXISTS
---------------------
The SQL agent needs a *real* database to text-to-SQL against, and the RAG
agent needs a *real* vector index to retrieve from. Both are "test fixtures
that look like production data" — small enough to set up in a few seconds,
but structured enough that the classifier/SQL/RAG nodes exercise genuine
retrieval and generation logic (which is exactly what we want traced in
LangSmith/LangFuse — an empty demo wouldn't produce any interesting spans).

Both setup functions are IDEMPOTENT: calling them twice does not duplicate
rows or rebuild the vector store from scratch, so `main.py` and the test
suite can call them freely without worrying about ordering.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from config.settings import get_settings

# ---------------------------------------------------------------------- #
# SQL agent's database: a tiny e-commerce schema (customers/products/orders)
# ---------------------------------------------------------------------- #

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id   INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    city          TEXT NOT NULL,
    signup_date   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS products (
    product_id    INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    category      TEXT NOT NULL,
    price_usd     REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    order_id      INTEGER PRIMARY KEY,
    customer_id   INTEGER NOT NULL REFERENCES customers(customer_id),
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    quantity      INTEGER NOT NULL,
    order_date    TEXT NOT NULL
);
"""

_SEED_CUSTOMERS = [
    (1, "Asha Rao", "Ahmedabad", "2025-01-14"),
    (2, "Vikram Shah", "Mumbai", "2025-02-02"),
    (3, "Meera Iyer", "Bengaluru", "2025-03-19"),
    (4, "Rohan Mehta", "Ahmedabad", "2025-04-05"),
]

_SEED_PRODUCTS = [
    (1, "Wireless Mouse", "Electronics", 799.0),
    (2, "Mechanical Keyboard", "Electronics", 3499.0),
    (3, "Standing Desk", "Furniture", 12999.0),
    (4, "Office Chair", "Furniture", 8999.0),
    (5, "USB-C Hub", "Electronics", 1299.0),
]

_SEED_ORDERS = [
    (1, 1, 1, 2, "2025-05-01"),
    (2, 1, 5, 1, "2025-05-01"),
    (3, 2, 3, 1, "2025-05-10"),
    (4, 3, 2, 1, "2025-06-02"),
    (5, 4, 4, 2, "2025-06-15"),
    (6, 2, 2, 1, "2025-07-01"),
]


def init_sql_database() -> Path:
    """
    Create (if needed) and seed (if empty) the SQLite database used by the
    SQL agent. Returns the resolved DB path so callers can log/inspect it.

    WHY SQLite specifically: this is a *practical about tracing*, not about
    database infra, so we want zero external services for the SQL half —
    SQLite is a single file, needs no server, and `sqlite3` is stdlib.
    """
    settings = get_settings()
    db_path = Path(settings.sqlite_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_SCHEMA_SQL)
        (existing,) = conn.execute("SELECT COUNT(*) FROM customers").fetchone()
        if existing == 0:
            conn.executemany("INSERT INTO customers VALUES (?, ?, ?, ?)", _SEED_CUSTOMERS)
            conn.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", _SEED_PRODUCTS)
            conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?)", _SEED_ORDERS)
            conn.commit()
    finally:
        conn.close()
    return db_path


def get_schema_description() -> str:
    """
    Human-readable schema summary handed to the LLM as part of the
    text-to-SQL prompt.

    WHY a hand-written string instead of `PRAGMA table_info` dumped raw:
    the model writes noticeably more reliable SQL when column *intent* is
    spelled out (e.g. "price_usd" vs. just "price_usd REAL"), and keeping
    this static means the prompt is identical across runs — which matters
    when you are comparing trace latency/cost between LangSmith and
    LangFuse, since prompt-token count is one of the variables you want to
    hold constant.
    """
    return (
        "Table customers(customer_id INTEGER, name TEXT, city TEXT, signup_date TEXT)\n"
        "Table products(product_id INTEGER, name TEXT, category TEXT, price_usd REAL)\n"
        "Table orders(order_id INTEGER, customer_id INTEGER, product_id INTEGER, "
        "quantity INTEGER, order_date TEXT)\n"
        "orders.customer_id references customers.customer_id; "
        "orders.product_id references products.product_id."
    )


# ---------------------------------------------------------------------- #
# RAG agent's vector store: a handful of FAQ/policy documents
# ---------------------------------------------------------------------- #

_FAQ_DOCS: list[tuple[str, str]] = [
    (
        "return_policy.txt",
        "Our return policy allows customers to return any unused product within "
        "30 days of delivery for a full refund. Furniture items must be returned "
        "in their original packaging. Electronics are eligible for replacement "
        "instead of refund if a manufacturing defect is reported within 7 days.",
    ),
    (
        "shipping_policy.txt",
        "Standard shipping takes 3-5 business days within the same state and "
        "5-8 business days for other states. Orders above 2000 rupees qualify "
        "for free standard shipping. Expedited shipping is available at "
        "checkout for an additional fee and delivers within 1-2 business days.",
    ),
    (
        "warranty_policy.txt",
        "All electronics carry a 1-year manufacturer warranty covering hardware "
        "defects. Furniture items carry a 2-year warranty against structural "
        "defects. Warranty claims require the original order ID and are "
        "processed within 5 business days of submission.",
    ),
    (
        "loyalty_program.txt",
        "Customers earn 1 loyalty point for every 100 rupees spent. Points can "
        "be redeemed at checkout at a rate of 1 point = 1 rupee. Loyalty tier "
        "upgrades (Silver, Gold, Platinum) are evaluated at the start of every "
        "quarter based on trailing-twelve-month spend.",
    ),
]


def _write_faq_source_files(knowledge_base_dir: Path) -> None:
    """Materialize the FAQ strings above as .txt files on disk, once."""
    knowledge_base_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in _FAQ_DOCS:
        file_path = knowledge_base_dir / filename
        if not file_path.exists():
            file_path.write_text(content, encoding="utf-8")


def init_vector_store():
    """
    Build (or load, if it already exists on disk) the Chroma vector store
    used by the RAG agent, embedding documents with Bedrock Titan Embed.

    WHY Chroma specifically, and not OpenSearch: this project's stack (see
    README) uses OpenSearch elsewhere for production-scale hybrid search,
    but for a single-machine tracing practical with 4 short documents,
    Chroma needs no cluster, persists to a local folder, and keeps the
    example runnable with `docker compose up` only for LangFuse — not for
    the vector store too.

    Returns a `langchain_chroma.Chroma` instance ready for `.as_retriever()`.
    """
    # Imported lazily so `config/db_init.py` can be imported by tests that
    # only touch the SQL half without requiring langchain-aws/chroma to be
    # importable (keeps unit tests fast and dependency-light).
    from langchain_aws import BedrockEmbeddings
    from langchain_chroma import Chroma
    from langchain_core.documents import Document

    settings = get_settings()
    knowledge_base_dir = Path(settings.knowledge_base_dir)
    _write_faq_source_files(knowledge_base_dir)

    embeddings = BedrockEmbeddings(model_id=settings.titan_embed_model_id, region_name=settings.aws_region)

    persist_dir = Path(settings.chroma_persist_dir)
    already_built = persist_dir.exists() and any(persist_dir.iterdir())

    store = Chroma(
        collection_name="faq_knowledge_base",
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    if not already_built:
        documents = [
            Document(page_content=content, metadata={"source": filename}) for filename, content in _FAQ_DOCS
        ]
        store.add_documents(documents)

    return store
