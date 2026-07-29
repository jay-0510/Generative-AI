"""
config package
==============
Startup-time configuration: environment settings (settings.py) and
one-time data-store initialization (db_init.py). Nothing in here should
depend on `src/`, so that config can be imported first, before any
LangGraph/LangChain object is constructed.
"""
