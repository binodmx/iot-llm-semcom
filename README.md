# AGL-IDS: Agentic LLM-Driven Semantic Policy Interpretation and Enforcement for Guided Learning in IoT Intrusion Detection Systems

## Introduction

The convergence of large-scale Internet of Things (IoT) deployments with interoperable infrastructures introduces significant challenges in maintaining security compliance under highly dynamic and heterogeneous conditions. Existing IoT security frameworks predominantly rely on static or manually updated policies, limiting their ability to adapt to evolving threats and operational contexts. To address these limitations, this paper proposes an agentic large language model (LLM) driven semantic policy interpretation and enforcement framework for managing IoT intrusion detection systems (IDS). The proposed framework leverages LLMs to semantically interpret high-level security policies and translate them into enforcement configurations that drive guided learning model updates for IDS. By embedding semantic policies into the learning pipeline, the framework enables guided model adaptation, improves decision transparency, and reduces the retraining overhead.

## Getting Started

1. Clone the repository.
2. Install required Python libraries using `pip install -r requirements.txt`.
3. Create `.env` file in the root directory and add API keys as follows.

    ```bash
    OPENAI_API_KEY=
    GOOGLE_API_KEY=
    ANTHROPIC_API_KEY=
    LANGCHAIN_API_KEY=
    LANGCHAIN_PROJECT=
    LANGCHAIN_TRACING_V2=
    ```

4. Create `data` directory in the root directory and subdirectories for datasets as follows.

    ```
    data
    |-cic-iot
        |-sample.csv
        └-population.csv
    |-wustl-iiot
    |-ton-iot
    └-bot-iot
    ```

5. Place downloaded datasets in the relevant subdirectories.

6. Make sure preferred LLMs are available (either locally using Ollama or through API).

7. Run `gradio app.py` to start the Gradio interface.

## Datasets

| Name       | Paper(s) | Year |
|------------| - | - |
| CICIoT2023 | CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment | 2023 |
| WUSTL-IIoT | WUSTL-IIOT-2021 Dataset for IIoT Cybersecurity Research | 2021 |
| TON_IoT    | TON_IoT telemetry dataset: a new generation dataset of IoT and IIoT for data-driven Intrusion Detection Systems | 2020 |
| Bot-IoT    | Towards the development of realistic botnet dataset in the internet of things for network forensic analytics: Bot-iot dataset | 2019 |
