"""
config package
--------------
Environment-driven configuration and one-time setup for the project.

    settings.py -> centralized constants: model IDs, AWS region, file
                   paths, and RAG tuning values (see settings.py docstring
                   for why these all live in one place)
    db_init.py   -> creates and seeds the sample SQLite database used by
                   the SQL agent path
"""
