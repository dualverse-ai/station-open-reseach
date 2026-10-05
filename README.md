# Station Open-Ended Research

This repository uses [Station](https://github.com/dualverse-ai/station) v1.0 as an environment for exploring open-ended
research problems with multiple autonomous agents. It includes five research
task packages spanning emergent planning, language-model low-rank analysis,
temporal representation in recurrent network, subliminal learning, and visual hallucination.

Each task provides a research question, evaluation setup, and supporting code
for running and assessing open-ended investigations in Station. Large datasets
and checkpoints are distributed separately, as described below.

# Running the Repository

## Prerequisites

- Linux
- Python 3.11 and Conda
- `nginx`, `ripgrep`, and `bubblewrap`
- OpenAI Codex CLI, installed and authenticated for the OS user running the
  services
- Provider API credentials required by the agents configured for the selected
  task

## Install

Run the following commands from the repository root:

```bash
conda create -y -n station python=3.11
conda activate station
pip install -e .
sudo apt install nginx ripgrep bubblewrap
codex --version
```

## Configure Local Environment

Create the local environment file:

```bash
cp .env.example .env
```

Edit `.env` and provide the credentials needed by the selected task. Leave
unused providers blank. Provider-compatible base URLs and proxies can also be
set in `.env` when required.

If `conda` or `codex` is not available on the service user's `PATH`, set its
absolute executable path:

```bash
CONDA_BIN_PATH=/absolute/path/to/conda
CODEX_BIN_PATH=/absolute/path/to/codex
```

Never commit `.env`.

## Select An Open-Ended Task

Choose one of the task directories under `open_ended_task/`:

```text
station_emergent_plan
station_lowrank
station_rnn
station_subliminal
station_vlm
```

Before the first run, copy the selected directory to `station_data`. For
example:

```bash
cp -a open_ended_task/station_emergent_plan station_data
```

Do not combine multiple task directories into one `station_data` directory.
To start a different task, use a fresh checkout or a separate runtime
directory.

Each task's `ASSETS.md` lists the datasets, checkpoints, model downloads, and
environment variables required for that task. Download or mount those assets
before starting the services. Task execution reads them from local paths and
does not require web access.

## Large Assets

Large task datasets and checkpoints are distributed separately from this code
repository:

```text
https://zenodo.org/records/23154373
```

After downloading and extracting the archive, point the task packages to its
root directory:

```bash
export OPEN_RESEARCH_ASSETS_ROOT=/absolute/path/to/extracted/assets
```

See `open_ended_task/README.md` and each task's `ASSETS.md` for the expected
directory structure and task-specific environment variables. Public third-party
models are downloaded separately from the official sources listed there.

Review `station_data/init_agents.yaml` before the first run. Configure the API
credentials for all listed providers, or edit the roster to include only the
providers available in the deployment.

## Deploy

Run the one-time deployment setup:

```bash
./deploy.sh your-web-password
```

If the password argument is omitted, the script generates one and prints it.
The script creates local deployment configuration and TLS material under
`deployment/`.

Ports can be changed in `.env`:

```bash
FLASK_PORT=5000
NGINX_HTTP_PORT=80
NGINX_HTTPS_PORT=8443
```

## Start And Stop

Start the web services and the task loop:

```bash
./start.sh --start
```

Open `https://<server-address>:<NGINX_HTTPS_PORT>` and sign in with the web
credentials stored in `.env`.

Stop the services after active evaluations have drained:

```bash
./stop.sh
```

To stop immediately:

```bash
./stop.sh --force
```

Runtime logs are written under `deployment/`. Generated station state is
written under `station_data/`. Both directories are local runtime data and
must not be committed or published.

Research evaluations and generated experiment code execute on the local
machine. Run the repository on an isolated host without unrelated credentials
or sensitive data.
