"""
src package
-----------
This project's application code.

    config/       -> agents.yaml and tasks.yaml: declarative definitions
                     of the crew's two agents and two tasks (see each
                     file's own header comment for why role/goal/backstory
                     and description/expected_output live in YAML rather
                     than inline Python)
    main.py         -> builds the crew (wiring YAML config to Amazon Nova
                     via CrewAI's @CrewBase pattern), runs it end to end,
                     inspects whether delegation occurred, and saves the
                     combined result as research_output.md
"""
