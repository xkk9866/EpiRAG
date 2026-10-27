# EpiRAG / QuoteNet

QuoteNet studies how readers with different evidence can answer a question together. Each reader selects a short sequence of tokens from its own documents and sends those tokens to its peers. The receiver uses the messages alongside its local evidence to predict an answer span.

The current implementation is in `quotenet/`. The repository also contains the EpiNet and EpiRAG implementations that share its data adapters, reader components and evaluation utilities.

## Code map

| Path | Purpose |
| --- | --- |
| `quotenet/model.py` | Reader, token selection, communication variants and answer loss |
| `quotenet/views.py` | Split and colocated evidence assignments |
| `quotenet/encode.py` | Episode encoding, answer supervision and batching |
| `quotenet/train.py` | Training, checkpoint loading and evaluation |
| `quotenet/llm_protocols.py` | Language-model communication comparisons |
| `epinet/data/` | Shared HotpotQA, 2WikiMultihopQA and MuSiQue adapters |
| `epinet/`, `epirag/` | Earlier models and retrieval-based comparisons |
| `tests/` | Evidence assignment, communication and gradient checks |

## Setup

Use Python 3.10 or newer. Install a PyTorch build suitable for your hardware, then install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Run module commands from the repository root. The default encoder is `microsoft/deberta-v3-base`; `--encoder` also accepts a local model directory. Dataset adapters download their inputs through Hugging Face Datasets. Download access is needed on the first run; an existing local cache can be used afterward. Model weights and benchmark corpora are not bundled.

## Train and evaluate

Train the independent reader first, then initialize the communicating reader from that checkpoint:

```bash
python -m quotenet.train --name base_s13 --mode indep --seed 13 --n-train 4000 --epochs 2 --out experiments/q2
python -m quotenet.train --name quote_s13 --mode quote --k 32 --seed 13 --train-seed 113 --n-train 3000 --epochs 1 --lr 2e-5 --head-lr 2e-4 --init experiments/q2/base_s13/model.pt --out experiments/q2
```

`--n-train`, `--n-dev` and `--n-test` specify examples per dataset. `--mode` selects a communication variant; use `python -m quotenet.train --help` for the available choices. The default device is `cuda:0`; select another device with `--device`.

To evaluate an existing run, reuse its architecture and run name with `--eval-only`. Add `--probe swap`, `--probe drop` or `--probe random` to measure how the prediction depends on exchanged messages. Each run writes its configuration, checkpoint, prediction rows and aggregate metrics under `--out/--name`.

For a small synthetic run:

```bash
python -m quotenet.train --name synthetic_check --datasets synth2 --mode quote --k 8 --max-len 128 --n-train 32 --n-dev 8 --n-test 8 --epochs 1 --max-steps 2 --workers 0
```

This still loads the pretrained encoder. Tests such as `python -m tests.check_quotenet` additionally use benchmark data and exercise the model's gradients. The shell launchers preserve the experiment schedules and accept `PYTHON` to select an interpreter; review GPU assignments before running multiple jobs.

## Release contents

This source release includes model, training, evaluation and diagnostic code. Manuscripts, submission materials, publication artwork, generated results and private machine configuration are excluded. `RELEASE_MANIFEST.json` records the included files and their hashes. Dependency lists describe the source requirements; they are not a lockfile for the original experiment environment.
