<img align="right" src="logo.png" alt="Matryoshka Logo" width="150" height="150">


# Matryoshka
> A semantic-aware log parser generator powered by Large Language Models

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [LLM API Configuration](#llm-api-configuration)
  - [Usage](#usage)
- [Data](#additional-data)
- [License](#license)

---

## Overview

Security analysts struggle to quickly and efficiently query and correlate log data due to the heterogeneity and lack of structure in real-world logs. Existing AI-based parsers focus on learning syntactic log templates but lack the semantic interpretation needed for querying. Directly querying large language models on raw logs is impractical at scale and vulnerable to prompt injection attacks. 

**Matryoshka** is the first end-to-end system leveraging LLMs to automatically generate semantically-aware structured log parsers. Matryoshka consists of three complementary stages:

1. **Syntax Parsing**: Learns log syntax to extract relevant fields
2. **Schema Creator**: Associates templates with semantically meaningful schemas for querying
3. **Mapping**: Maps custom schemas to standardized taxonomies: [OCSF](https://schema.ocsf.io/) or [UDM](https://docs.cloud.google.com/chronicle/docs/event-processing/udm-overview)

While Matryoshka uses LLMs for generating parsers, these parsers only rely on regular expressions for conveting log data to structured formats, making them suitable for high volume log sources. Matryoshka’s tempate generator outperforms prior works. Its semantic steps achieve an F_1 score of 0.99 on realistic security queries, and equals human-written parsers in head to head comparisons.

### Research Background

Based on the paper *"Semantic-Aware Parsing for Security Logs"* by Julien Piet, Vivian Fang, Rishi Khare, Scott Coull, Vern Paxson, Raluca Ada Popa, and David Wagner from UC Berkeley.

**[📖 Read the Paper](https://people.eecs.berkeley.edu/~julien.piet/matryoshka.pdf)**

> ⚠️ **Disclaimer**: This is research code and may contain bugs. For production use cases, please contact [Julien Piet](https://people.eecs.berkeley.edu/~julien.piet/).

---

## Getting Started

### Prerequisites

- **Python**: 3.9 or higher
- **API Access**: At least one of Gemini, OpenAI, or Anthropic API keys

### Installation

#### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/julien-piet/matryoshka
cd matryoshka

# Create virtual environment
python3.9 -m venv env
source env/bin/activate

# Install Matryoshka
pip install .
```

### LLM API Configuration

Choose your preferred LLM provider and follow the setup instructions:

#### Gemini

The original paper uses Gemini models. In order to setup the Gemini API, you will need to follow instructions [here](https://ai.google.dev/palm_docs/oauth_quickstart) to login to your GCP account. At the time of writing, the instructions for this were:

1. **Enable the VertexAI API in Google Cloud, and create OAuth credentials**
2. **Install gcloud on your server**
3. **Setup Application Default Credentials**

```sh
gcloud auth application-default login
```
4. **Export your GCP project name**

```
export GEMINI_PROJECT=YOUR_GCP_PROJECT_NAME
```


- **Usage**: Specify your model using ``--model MODEL`` flag
- **Supported Models**: Any Gemini generative model  
- **Embeddings**: `text-embedding-005`

#### OpenAI

```bash
export OPENAI_API_KEY=YOUR_API_KEY
```

- **Usage**: Add `--backend openai` flag and specify your model using ``--model MODEL`` flag
- **Supported Models**: Any OpenAI generative model  
- **Embeddings**: `text-embedding-3-large`

#### Anthropic

```bash
export ANTHROPIC_API_KEY=YOUR_API_KEY
export OPENAI_API_KEY=YOUR_OPENAI_KEY  # Required for embeddings
```

- **Usage**: Add `--backend anthropic` flag and specify your model using ``--model MODEL`` flag
- **Supported Models**: Any Anthropic generative model  
- **Note**: Uses OpenAI embeddings as Anthropic doesn't provide embedding models

### Usage

#### Parser Generation

##### Available Commands

| Command | Description |
|---------|-------------|
| `matryoshka-syntax` | Run template generation step |
| `matryoshka-schema` | Run schema generation step |
| `matryoshka-map` | Run mapping step |

##### Configuration File

Create a JSON configuration file:

```json
{
  "data_path": "path/to/your/log/file.log",
  "results_path": "path/to/output/folder",
}
```

**Configuration Fields:**
- `data_path`: **Required** - Path to your log file. An example sshd log file is provided in `examples/sshd.txt`. 
- `results_path`: **Required** - Output directory.

##### Command Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config_file` | - | **Required** - Configuration file path |
| `--model` | `gemini-2.5-flash` | LLM model to use |
| `--backend` | `google` | Provider: `google`, `openai`, `anthropic` |
| `--thread_count` | `16` | Number of processing threads |
| `--file_percent` | `1.0` | Percentage of log file to process |
| `--buffer_size` | `2500` | Log processing buffer size |
| `--force_overlap` | - | Skip overlap confirmation (recommended) |
| `--use_fewshot` | - | Use human-provided few-shot examples |
| `--cache_dir` | `.cache/` | Cache directory path |
| `--parser` | None | Path to saved template generation dill file. <br> Use if template generation was halted to recover progress. |

##### Step-by-Step Execution (Recommended)

```bash
# 1. Generate templates
matryoshka-syntax --config_file config.json --model gemini-2.5-flash --force_overlap

# 2. Create semantic schema
matryoshka-schema --config_file config.json --model gemini-2.5-flash

# 3. Map to OCSF standards
matryoshka-map --config_file config.json --model gemini-2.5-flash
```

#### Parser Curation

Launch the interactive web interface for parser editing and querying:

```bash
# With specific files
matryoshka-edit --log_file logs/sshd.log --parser_file parsers/sshd_parser.dill

# With configuration file
matryoshka-edit --config_file config.json
```

**Web Interface Options:**
- `--listen_addr`: Bind address (default: `127.0.0.1`)
- `--listen_port`: Port number (default: `8080`)

#### Log Ingestion

Convert logs to structured JSON format:

```bash
# Basic ingestion
matryoshka-ingest logs/input.log parsers/parser.dill output.json

# With OCSF mapping
matryoshka-ingest logs/input.log parsers/parser.dill output.json --OCSF
```

---

## Additional Data

### SecurityLogs Dataset

Matryoshka was evaluated on both [LogHub 2.0](https://github.com/logpai/loghub-2.0) and a curated dataset from public RedHat bug reports.

**[📥 Download SecurityLogs Dataset](https://drive.google.com/file/d/1dG-skZ_g47DUqsX8BgCiS03stXpawVz5/view?usp=sharing)**

**Included Log Types:**
- Audit logs
- SSH Server logs  
- DHCP Client logs
- CRON logs
- Puppet logs

### OCSF Descriptions

OCSF attributes are only described in the context of their parent attribute. Matryoshka augments OCSF attributes with a description relative to the event they are part of. For instance, field `authentication.src_endpoint.ip` is described as "The IP address of the endpoint, in either IPv4 or IPv6 format.", while our generated description for this field is "The IP address (IPv4 or IPv6) of the source endpoint involved in the authentication event (e.g., a user's login or logout attempt).", taking into account the role of its parent.

Generating these descriptions is costly, but only needs to be done once every time OCSF releases a new version. These descriptions are saved in a cache directory. When running the mapping algorithm, this cache will be created automatically. To avoid doing this yourself, you can download cached descriptions **[📥 here](https://drive.google.com/file/d/19nLz10RCMXLeK6fC9jSgJkcp-F3BbTqG/view?usp=share_link)**, extract them, and put both json files in an `OCSF` directory at the `.cache/OCSF` path.

---

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE.txt).

---

## Contributing & Contact

For questions, issues, or collaboration opportunities, please contact **Julien Piet** 
🌐 [Personal Website](https://people.eecs.berkeley.edu/~julien.piet/)
