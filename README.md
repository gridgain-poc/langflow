# [![Langflow](./docs/static/img/hero.png)](https://www.langflow.org)

<p align="center" style="font-size: 12px;">
    Langflow is a low-code app builder for RAG and multi-agent AI applications. It's Python-based and agnostic to any model, API, or database.
</p>

<p align="center" style="font-size: 12px;">
    <a href="https://docs.langflow.org" style="text-decoration: underline;">Docs</a> -
    <a href="https://astra.datastax.com/signup?type=langflow" style="text-decoration: underline;">Free Cloud Service</a> -
    <a href="https://docs.langflow.org/getting-started-installation" style="text-decoration: underline;">Self Managed</a>
</p>

<div align="center">
  <a href="./README.md"><img alt="README in English" src="https://img.shields.io/badge/English-d9d9d9"></a>
  <a href="./README.PT.md"><img alt="README in Portuguese" src="https://img.shields.io/badge/Portuguese-d9d9d9"></a>
  <a href="./README.ES.md"><img alt="README in Spanish" src="https://img.shields.io/badge/Spanish-d9d9d9"></a>  
  <a href="./README.zh_CN.md"><img alt="README in Simplified Chinese" src="https://img.shields.io/badge/简体中文-d9d9d9"></a>
  <a href="./README.ja.md"><img alt="README in Japanese" src="https://img.shields.io/badge/日本語-d9d9d9"></a>
  <a href="./README.KR.md"><img alt="README in KOREAN" src="https://img.shields.io/badge/한국어-d9d9d9"></a>
</div>

## ✨ Core features

1. **Python-based** and agnostic to models, APIs, data sources, or databases.
2. **Visual IDE** for drag-and-drop building and testing of workflows.
3. **Playground** to immediately test and iterate workflows with step-by-step control.
4. **Multi-agent** orchestration and conversation management and retrieval.
5. **Free cloud service** to get started in minutes with no setup.
6. **Publish as an API** or export as a Python application.
7. **Observability** with LangSmith, LangFuse, or LangWatch integration.
8. **Enterprise-grade** security and scalability with free DataStax Langflow cloud service.
9. **Customize workflows** or create flows entirely just using Python.
10. **Ecosystem integrations** as reusable components for any model, API or database.

## GridGain Vector Store Component

Langflow includes a custom GridGain Vector Store component that provides enhanced CSV handling capabilities. This component allows you to:

1. Ingest CSV files directly into the GridGain Vector Store.
2. Automatically process the CSV content and metadata to create LangChain Documents.
3. Perform similarity searches on the ingested data using the GridGain Vector Store.

To use the GridGain Vector Store component, follow these steps:

1. Install the required dependencies:

```
pip install langflow 
pip install lib\gg_langchain-0.6.1.tar.gz
pip install -e .
```

2. In the Langflow Visual IDE, add the GridGain Vector Store component to your workflow.
3. Configure the component with the necessary settings, such as the cache name, API endpoint, and CSV file (if applicable).
4. Connect the component to other components in your workflow, such as an Embedding component and a Search Query component.
5. Run the workflow to ingest the CSV data and perform similarity searches.

The GridGain Vector Store component is designed to seamlessly integrate with the Langflow platform, allowing you to leverage the power of GridGain's vector search capabilities within your low-code AI applications.

## 📦 Quickstart

- **Install with pip** (Python 3.10 or greater):

```shell
pip install langflow
```

- **Cloud:** DataStax Langflow is a hosted environment with zero setup. [Sign up for a free account.](https://astra.datastax.com/signup?type=langflow)
- **Self-managed:** Run Langflow in your environment. Follow these steps:

1. Install Langflow:

```shell
pip install langflow
pip install -e .
```

2. Run the Langflow backend:

```shell
uv run langflow run --env-file local.env
```

3. Set up the Langflow frontend:

```shell
cd src/frontend
npm install
npm run build
```

- **Hugging Face:** [Clone the space using this link](https://huggingface.co/spaces/Langflow/Langflow?duplicate=true) to create a Langflow workspace.

[![Getting Started](https://github.com/user-attachments/assets/f1adfbe7-3c35-43a4-b265-661f3d4f875f)](https://www.youtube.com/watch?v=kinngWhaUKM)

## ⭐ Stay up-to-date

Star Langflow on GitHub to be instantly notified of new releases.

![Star Langflow](https://github.com/user-attachments/assets/03168b17-a11d-4b2a-b0f7-c1cce69e5a2c)

## 👋 Contribute

We welcome contributions from developers of all levels. If you'd like to contribute, please check our [contributing guidelines](./CONTRIBUTING.md) and help make Langflow more accessible.

---

[![Star History Chart](https://api.star-history.com/svg?repos=langflow-ai/langflow&type=Timeline)](https://star-history.com/#langflow-ai/langflow&Date)

## ❤️ Contributors

[![langflow contributors](https://contrib.rocks/image?repo=langflow-ai/langflow)](https://github.com/langflow-ai/langflow/graphs/contributors)