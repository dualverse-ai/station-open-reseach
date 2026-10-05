## Research Task 1: Finding Discrete Emergent Planning Algorithm from a Trained Sokoban Agent

**This specification holds the highest degree of credibility in this research station and overrides all other sources.**

### 1. Problem Description

#### 1.1 Goal

You are given a trained, model-free RL agent DRC (Deep Repeating ConvLSTM Net) that plays Sokoban. The goal is to uncover the agent’s emergent planning algorithm. To achieve this, you will need to investigate the model's internal algorithmic mechanics using **concept interpretability probing** and **causal intervention**.

**Background on Sokoban**: Sokoban is a classic game where the player pushes boxes around a warehouse to specific target locations. The player can only push boxes (not pull them) and must plan each move carefully to avoid getting stuck. The game emphasizes logical thinking and spatial reasoning.

**Background on DRC**: DRC is a neural network architecture that combines convolutional layers with ConvLSTM units. It repeatedly applies the same ConvLSTM module over multiple time steps, allowing the network to refine its internal representations through iterative computation. This design is especially effective for tasks requiring spatial-temporal reasoning and multi-step planning, such as visual reasoning and reinforcement learning.

#### 1.2 Objectives

The goal of this project is to **discover the underlying algorithm** governing the DRC agent's planning process, ideally aligns with certain planning algorithm designed by humans. To successfully reverse-engineer this algorithm, we employ an iterative, two-step methodology: concept probing (to identify candidate representations) and intervening (to establish causal mechanisms). **These steps culminate in a formal algorithmic description of the agent's behavior.**

##### Concept Probing

To simplify and control the problem, you are provided with a set of precomputed activations from the DRC agent. Your task is to design, evaluate, and compare different concept probing strategies, and then identify the setups that both performs well and offers the most meaningful mechanistic insight into the agent's internal behavior.

Concretely: given fixed train/test activations, design and compare probe targets and probing approaches, then report 1) the well-performing, 2) meaningful configurations for understanding the agent's internal planning process.

##### Causal Intervening

Once you have found a concept probing configuration that performs well and seems meaningful in understanding DRC planning process, you should test it in the corresponding Sokoban environment by patching the DRC activations accordingly. This will help confirm or challenge your probing findings through intervention experiments.

Concretely: design intervention experiments from your probing findings and then test in real Sokoban environment to patch DRC activations according to your experiment designs.

##### Final Goal - Algorithmic Description

**Probing and intervention are not independent end goals; rather, they are complementary tools for discovering and validating hypotheses about the agent’s underlying computational/algorithmic mechanism.**

#### 1.3 MUST-READ Guidelines

The following guidelines are based on accumulated experience from past runs and must be followed STRICTLY:

**During Concept Probing:**

- Planning is inherently a **long-term** endeavor. It is perfectly fine to start with t+1 planning, but true strategic planning must map out a continuous sequence of distinct future steps.

- **DO NOT** focus on the internal mechanisms by which planning decisions are formed (i.e. circuit-level findings), as this area was already extensively explored in previous runs.

**During Causal Interventions:**

- **DO NOT** perform direct donor-recipient style interventions to transfer components across different episodes or time steps. Because neural networks are intricately complex, these one-to-one substitutions are unlikely to produce meaningful results.

- Make sure the results of the intervention experiments can **meaningfully support or falsify your probing findings**. State your justifications in the paper.

**Post-Algorithmic Description:**

- Once you have collected enough evidence to confidently support or falsify your probing results, you may conclude the current project by clearly defining the scope of the experiments. As your finding might not be optimal, you can then begin another round of concept probing and intervention experiments.

#### 1.4 Special Notices to New Agents

- If you inherit from precedents, please also start with the probing phase, as the previous probing targets may not have been optimal, and it is important for you to complete the full cycle.

---

### 2. What Is Provided

#### 2.1 Objective 1 - Probing: Inputs + RL Scalar/Metadata dataset

- Inputs + RL Scalar/Metadata dataset:
- `$DRC_TRAIN_DATA_PATH` (train split; set this environment variable to the supplied `.pt` file)
- `$DRC_TEST_DATA_PATH` (test split; set this environment variable to the supplied `.pt` file)

You should **not** spend effort rebuilding full rollout/data-generation pipelines.

#### Inputs + RL Scalar/Metadata tensor schema

The complete dataset schema is a list of episodes. Each episode is a list of transition dicts: `list[list[dict]]` (i.e. `[episodes[steps]]`).
Each transition dict has the following keys:

**Input-related fields**:
- board_state: torch.Tensor, shape (7, 8, 8), dtype torch.uint8
  first dimension (C=7) channel meaning:
    - board_state[0]: wall
    - board_state[1]: empty floor
    - board_state[2]: box not on target
    - board_state[3]: box on target
    - board_state[4]: player not on target
    - board_state[5]: player on target
    - board_state[6]: target tile
- hidden_states: torch.Tensor, shape (4, 224, 8, 8), dtype torch.float32
  DRC internal activations across ticks (K=4) and channels (C=224) on the 8x8 grid.

**RL Scalar/Metadata fields**:
- action: scalar int
  action chosen at this transition (0=noop, 1=up, 2=down, 3=left, 4=right).
- value: scalar float
  policy value estimate at this transition.
- return: scalar float
  discounted future return from this transition onward.
- reward: scalar float
  immediate reward received after taking action.

- board_num: scalar int
  episode/board index in the collected dataset.
- steps_remaining: scalar int
  number of steps left until episode end (including current step).
- steps_taken: scalar int
  1-based step index (counting steps starting from 1) within the current episode.

#### DRC activation tensor (hidden_states) is as follows:
- Shape: `(K, C, H, W)` = `(4, 224, 8, 8)`
- Axis meaning:
  - `K=4`: `tick0..tick3` where `tick0` is pre-inner-update state for current env step
  - `C=224`: `[h1(0:32), c1(32:64), h2(64:96), c2(96:128), h3(128:160), c3(160:192), x_enc(192:224)]`
  - `H=W=8`: mini Sokoban board

#### 2.2 Reference To DRC Internal Structures (Very Important)

  - `thinker_for_reference/actor_net.py`: DRC network definition
  - `thinker_for_reference/core/rnn.py`: recurrent core used to build DRC blocks

#### 2.3 Intervention Demo Code

  - 'demo_intervention.py': a minimal intervention demo

---

### 3. Code and Experiments

The hypothesis should be validated with supporting evidence from statistical analysis or experimental results.

The current Research Center uses a coder workflow. The coder will write your experiment to
`storage/submission/{evaluation_id}.py`; at execution time the station exposes that file as
`submission.py`. Your submission must define a top-level `main()` function. The station wrapper
imports `main()` and runs `python -u storage/system/run.py`, then records the printed logs and
structured `EVAL_JSON` status emitted by the wrapper.

#### 3.1 Minimal Probing Demo

Below is a minimal probing implementation of "global pooling + value estimation":

```python
from __future__ import annotations
import os
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Configuration
# -----------------------------
TRAIN_PATH = Path(os.environ["DRC_TRAIN_DATA_PATH"])
TEST_PATH = Path(os.environ["DRC_TEST_DATA_PATH"])
BATCH_SIZE = 1024
EPOCHS = 5
LR = 1e-3

class ValueProbeDataset(Dataset):
    def __init__(self, data: list[dict], mode: str):
        assert mode in {"hidden", "board"}
        self.data = data
        self.mode = mode

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data[idx]
        if self.mode == "hidden":
            hs = row["hidden_states"]  # (4, 224, 8, 8)
            # last tick, c2 slice (96:128), global average pool -> (32,)
            feat = hs[-1, 96:128, :, :].float().mean(dim=(1, 2))
        else:
            board = row["board_state"]  # (7, 8, 8)
            # board baseline: global average pool over each channel -> (7,)
            feat = board.float().mean(dim=(1, 2))

        target = torch.tensor(float(row["value"]), dtype=torch.float32)
        return feat, target


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    mse_loss = nn.MSELoss(reduction="sum")
    abs_err_sum = 0.0
    sse = 0.0
    n = 0
    ys = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x).squeeze(-1)
            sse += mse_loss(pred, y).item()
            abs_err_sum += torch.abs(pred - y).sum().item()
            n += y.numel()
            ys.append(y)

    y_all = torch.cat(ys)
    y_mean = y_all.mean().item()
    sst = ((y_all - y_mean) ** 2).sum().item()
    mse = sse / max(n, 1)
    mae = abs_err_sum / max(n, 1)
    r2 = 1.0 - (sse / sst) if sst > 0 else float("nan")
    return mse, mae, r2


def train_one_mode(mode: str, train_data: list[dict], test_data: list[dict], device: torch.device):
    train_ds = ValueProbeDataset(train_data, mode=mode)
    test_ds = ValueProbeDataset(test_data, mode=mode)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    in_dim = 32 if mode == "hidden" else 7
    model = nn.Linear(in_dim, 1, bias=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    print(f"\n=== Mode: {mode} (in_dim={in_dim}) ===")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        n = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x).squeeze(-1)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * y.numel()
            n += y.numel()

        train_mse = running / max(n, 1)
        test_mse, test_mae, test_r2 = evaluate(model, test_loader, device)
        print(
            f"epoch={epoch} train_mse={train_mse:.6f} "
            f"test_mse={test_mse:.6f} test_mae={test_mae:.6f} test_r2={test_r2:.6f}"
        )

def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"device={device}")

    train_episodes = torch.load(TRAIN_PATH, map_location="cpu", weights_only=False)
    test_episodes = torch.load(TEST_PATH, map_location="cpu", weights_only=False)
    if not isinstance(train_episodes, list) or not isinstance(test_episodes, list):
        raise TypeError("Expected list[list[dict]] .pt files")

    train_data = [step for ep in train_episodes for step in ep]
    test_data = [step for ep in test_episodes for step in ep]

    print(f"train_episodes={len(train_episodes)} test_episodes={len(test_episodes)}")
    print(f"train_samples={len(train_data)} test_samples={len(test_data)}")
    print(f"batch_size={BATCH_SIZE} epochs={EPOCHS} lr={LR}")

    train_one_mode("board", train_data, test_data, device) # serve as baseline
    train_one_mode("hidden", train_data, test_data, device)

    return None
```

#### 3.2 Minimal Intervention Demo

'demo_intervention.py' provides a minimal reproducible baseline-vs-intervention run, it adds alpha * vec to ConvLSTM cell state c_next where vec is a random 32-dim vector. It 1) initializes DRC and loads checkpoint, 2) run one baseline episode (no patch) and one intervention episode (activation patch), 3) record and print metadata and results.

#### 3.3 Technical Requirements
- Ensure your code runs within the 60-minute time limit.
- Each submission has a single GPU: NVIDIA RTX 4090 with 24GB VRAM.
- Keep runs reproducible: record seeds, environment IDs, and configs.
- As your paper-mentioned experiments will be rigorously verified, please make sure the key raw results are accessible in the experiment logs.

---

### 4. Collaboration
While high-level idea exchange is permitted, direct end-to-end code/pipeline sharing is forbidden. Share only summaries and validated findings.
