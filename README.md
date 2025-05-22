# LLM-Enhanced Semantic Policy Interpretation and Enforcement for Secure and Compliant Communication in 6G-Enabled IoT

## Introduction

In this research we try to find how Large Language Models (LLMs) perform semantic communication to improve the IoT security and access control. 
In particular, we investigate whether LLMs can generate semantically extracted information from intrusion detection models and 
distil that knowledge into smaller client models.
This experiment integrates LLM agents to build a novel policy enforcement framework for IoT Intrusion Detection Systems.

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
    |-wustl-iiot
    |-ton-iot
    |-bot-iot
    └-unsw-nb15
    ```

5. Place downloaded datasets in the relevant subdirectories.

6. Run `main.ipynb` Python notebook for each dataset.

## Datasets

| Name       | Paper(s) | Year |
|------------| - | - |
| CICIoT2023 | CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment | 2023 |
| WUSTL-IIoT | WUSTL-IIOT-2021 Dataset for IIoT Cybersecurity Research | 2021 |
| TON_IoT    | TON_IoT telemetry dataset: a new generation dataset of IoT and IIoT for data-driven Intrusion Detection Systems | 2020 |
| Bot-IoT    | Towards the development of realistic botnet dataset in the internet of things for network forensic analytics: Bot-iot dataset | 2019 |
