# Generative AI Practical Learning

A hands-on Generative AI learning repository that progresses from foundational NLP and LLM concepts to retrieval-augmented generation (RAG), agent workflows, observability, and deployable applications. The work is organized as **12 focused practicals** and **3 cumulative milestones**. Units are self-contained, with setup, configuration, execution, and validation guidance maintained in their project documentation where available.

## Purpose

This repository is a practical record of building and evaluating Generative AI systems with an emphasis on:

- Moving from NLP representations and embeddings to production-oriented LLM applications.
- Using Amazon Bedrock and related AWS services responsibly for model inference and retrieval workloads.
- Designing reliable RAG, tool-use, and conditional agent workflows.
- Applying software-engineering practices including typed interfaces, modular design, tests, safety controls, and tracing.

## Learning Path

The practicals build the supporting concepts and implementation skills used by the milestones. The milestones consolidate that work into end-to-end applications.

Detailed documentation intentionally lives with the projects. Consult the relevant project's documentation before installing dependencies or running commands.

## Repository Layout

```text
GenAI-Practical_Learning/
├── milestone_01_llm_microservice/   # Cumulative milestone
├── milestone_2_rag_architecture/    # Cumulative milestone
├── practical_*/                     # Independent learning exercises
└── README.md                         # Repository overview
```

This is a collection of independent Python projects, not a single installable package. Dependencies, environment variables, data, notebooks, services, and test commands are scoped to the project that needs them.

## Getting Started

1. Clone the repository and enter the working directory.

   ```bash
   git clone https://github.com/Jay-Patel-ai-ak/GenAI_practical_learning.git GenAI-Practical_Learning
   cd GenAI-Practical_Learning
   ```

2. Choose a practical or milestone based on your current learning goal.
3. Read the selected project's available documentation completely, including prerequisites, cloud requirements, configuration, and cost notes.
4. Create a dedicated virtual environment in the selected project and install only its declared dependencies.
5. Run its tests before using cloud-backed or containerized workflows where available.

## Prerequisites

Requirements vary by project. Across the repository, you may need:

- Python 3.10 or later for most current projects.
- `pip` and a virtual-environment tool such as `venv`.
- Jupyter for notebook-based exercises.
- Docker Engine and Docker Compose for containerized workflows.
- An AWS account, AWS CLI credentials, appropriate IAM permissions, and Amazon Bedrock model access for AWS-backed exercises.
- Optional tracing-service credentials where observability exercises require them.

Some exercises run completely locally or use mocks for tests. Do not assume that a project requires AWS, Docker, or external credentials until its documentation says so.

## Working Conventions

- Treat each directory as an isolated project; do not install all dependency files into one shared environment.
- Run commands from the project directory unless its documentation specifies otherwise.
- Keep generated artifacts, local datasets, caches, and service volumes out of version control when they are ignored by the project.
- Use the documented test command for the selected project. Many test suites mock cloud dependencies, but integration runs may incur AWS or infrastructure costs.
- Review service cleanup instructions after cloud or Docker experiments to avoid unnecessary charges.

## Security And Cost Awareness

- Never commit API keys, AWS access keys, `.env` files, or sensitive source documents.
- Follow least-privilege IAM practices and grant only the permissions required for the selected exercise.
- Review regional model availability and Bedrock access before running model calls.
- Amazon Bedrock, Textract, OpenSearch, and other managed services can incur charges. Use the cleanup guidance in the applicable project documentation after experimentation.

## Quality Approach

Projects in this repository use the tools appropriate to their scope, including unit tests, mocked external dependencies, validation models, API contracts, retrieval evaluation, SQL safety guards, and workflow tracing. Refer to the applicable project documentation for the exact coverage and limitations of its verification strategy.

## Contributing

Contributions and improvements are welcome. Keep changes scoped to the relevant practical or milestone, preserve its local setup conventions, update its documentation when behavior changes, and run the checks documented for that project before opening a pull request.
