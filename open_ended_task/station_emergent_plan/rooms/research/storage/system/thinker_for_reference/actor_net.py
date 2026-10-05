"""
Reference-only actor network: DRCNet.
This file intentionally keeps only `DRCNet`.
"""
from __future__ import annotations
from collections import namedtuple
from typing import Any, Sequence, Tuple
import torch
from torch import nn
from torch.nn import functional as F

try:
    # External dependency (not vendored in this repo). Kept as-is since this file is
    # meant as a reference implementation.
    from thinker.core.rnn import ConvAttnLSTM
except Exception:  # pragma: no cover
    ConvAttnLSTM = None  # type: ignore[assignment]

class ActorBaseNet(nn.Module):
    """Minimal base class, only providing what DRCNet uses."""

    def __init__(
        self,
        obs_space: Any,
        action_space: Any,
        flags: Any,
        tree_rep_meaning: Any = None,
        record_state: bool = False,
    ) -> None:
        super().__init__()
        self.flags = flags
        self.record_state = record_state
        self.dim_actions = int(getattr(flags, "dim_actions", 1))
        self.tuple_action = self.dim_actions > 1
        self.num_actions = _infer_num_actions(action_space)
        self.real_states_shape = _infer_real_states_shape(obs_space)
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        # Kept intentionally minimal: assume env_out.real_states are already normalized
        # (or normalization is handled in the wrapper).
        return x

class DRCNet(ActorBaseNet):
    def __init__(
        self,
        obs_space: Any,
        action_space: Any,
        flags: Any,
        tree_rep_meaning: Any = None,
        record_state: bool = False,
        num_layers: int = 3,
        num_ticks: int = 3,
        input_dim: int = 7,
    ) -> None:
        super().__init__(obs_space, action_space, flags, tree_rep_meaning, record_state)
        assert flags.wrapper_type == 1
        if ConvAttnLSTM is None:  # pragma: no cover
            raise ImportError(
                "ConvAttnLSTM could not be imported (thinker is not installed). "
                "This file is reference-only; install thinker to instantiate DRCNet."
            )

        k1, s1, p1 = (3, 1, 1) if flags.mini else (8, 4, 2)
        k2, s2, p2 = 4, 2, 1

        if flags.mini:
            if flags.mini_unqtar and flags.mini_unqbox:
                self.in_channels = 13
            elif flags.mini_unqtar:
                self.in_channels = 10
            else:
                self.in_channels = input_dim
        else:
            self.in_channels = 3

        encoder_layers = [
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=32,
                kernel_size=k1,
                stride=s1,
                padding=p1,
            )
        ] + (
            []
            if flags.mini
            else [
                nn.Conv2d(
                    in_channels=32,
                    out_channels=32,
                    kernel_size=k2,
                    stride=s2,
                    padding=p2,
                ),
            ]
        )
        self.encoder = nn.Sequential(*encoder_layers)
        def output_shape(h: int, w: int, kernel: int, stride: int, padding: int) -> Tuple[int, int]:
            return (
                ((h + 2 * padding - kernel) // stride + 1),
                ((w + 2 * padding - kernel) // stride + 1),
            )
        h, w = output_shape(self.real_states_shape[1], self.real_states_shape[2], k1, s1, p1)
        if not flags.mini:
            h, w = output_shape(h, w, k2, s2, p2)
        self.num_layers = num_layers
        self.num_ticks = num_ticks
        self.core = ConvAttnLSTM(
            input_dim=32,
            hidden_dim=32,
            num_layers=self.num_layers,
            attn=False,
            h=h,
            w=w,
            kernel_size=3,
            mem_n=None,
            num_heads=8,
            attn_mask_b=None,
            tran_t=self.num_ticks,
            pool_inject=True,
        )
        last_out_size = 32 * h * w * 2
        self.final_layer = nn.Linear(last_out_size, 256)
        self.policy = nn.Linear(256, self.num_actions * self.dim_actions)
        self.baseline = nn.Linear(256, 1)
        self.record_core_output = False
        self.record_gates = False
        if getattr(flags, "ppo_k", 1) > 1:
            kl_beta = torch.tensor(1.0)
            self.register_buffer("kl_beta", kl_beta)

    def initial_state(self, batch_size: int, device: torch.device | None = None) -> Any:
        return self.core.initial_state(batch_size, device=device)

    def forward(
        self,
        env_out: Any,
        core_state: Sequence[Any] = (),
        clamp_action: torch.Tensor | None = None,
        compute_loss: bool = False,
        greedy: bool = False,
        ood: bool = False,
    ) -> Any:
        done = env_out.done
        assert len(done.shape) == 2, f"done shape should be (T, B) instead of {done.shape}"
        T, B = done.shape

        x = self.normalize(env_out.real_states.float())
        x = torch.flatten(x, 0, 1)
        x_enc = self.encoder(x)
        core_input = x_enc.view(*((T, B) + x_enc.shape[1:]))
        core_output, core_state = self.core(
            core_input,
            done,
            core_state,
            record_state=self.record_state,
            record_output=self.record_core_output,
            record_gates=self.record_gates,
        )
        if self.record_state:
            # hidden_state stores the LSTM states with x_enc concatenated along channels.
            self.hidden_state = torch.cat(
                [self.core.hidden_state, torch.stack([x_enc] * (1 + self.num_ticks), dim=1)],
                dim=2,
            )

        core_output = torch.flatten(core_output, 0, 1)
        core_output = torch.cat([x_enc, core_output], dim=1)
        core_output = torch.flatten(core_output, 1)

        if ood:
            return core_state

        final_out = F.relu(self.final_layer(core_output))

        pri_logits = self.policy(final_out)
        pri_logits = pri_logits.view(T * B, self.dim_actions, self.num_actions)

        if compute_loss:
            # -H(pi) using the soft-target variant of cross entropy.
            entropy_loss = -torch.nn.CrossEntropyLoss(reduction="none")(
                input=torch.flatten(pri_logits, 0, 1),
                target=torch.flatten(F.softmax(pri_logits, dim=-1), 0, 1),
            )
            entropy_loss = entropy_loss.view(T, B, self.dim_actions)
            entropy_loss = torch.sum(entropy_loss, dim=-1)
        else:
            entropy_loss = None

        pri = sample(pri_logits, greedy=greedy, dim=-1)
        pri_logits = pri_logits.view(T, B, self.dim_actions, self.num_actions)
        pri = pri.view(T, B, self.dim_actions)

        if clamp_action is not None:
            pri[: clamp_action.shape[0]] = clamp_action

        c_action_log_prob = compute_discrete_log_prob(pri_logits, pri)

        pri_env = pri[-1, :, 0] if not self.tuple_action else pri[-1]
        action = pri_env
        action_prob = F.softmax(pri_logits, dim=-1)
        if not self.tuple_action:
            action_prob = action_prob[:, :, 0]

        baseline = self.baseline(final_out).view(T, B, 1)

        if compute_loss:
            reg_loss = (
                1e-3 * torch.sum(torch.square(pri_logits), dim=(-2, -1))
                + 1e-5 * torch.sum(torch.square(self.baseline.weight))
                + 1e-5 * torch.sum(torch.square(self.policy.weight))
            )
        else:
            reg_loss = None

        actor_out = ActorOut(
            pri=pri,
            pri_param=pri_logits,
            reset=None,
            reset_logits=None,
            action=action,
            action_prob=action_prob,
            c_action_log_prob=c_action_log_prob,
            baseline=baseline,
            baseline_enc=None,
            entropy_loss=entropy_loss,
            reg_loss=reg_loss,
            misc={},
        )
        return actor_out, core_state

ActorOut = namedtuple(
    "ActorOut",
    [
        "pri",  # sampled primary action
        "pri_param",  # parameters for primary action distribution (logits)
        "reset",
        "reset_logits",
        "action",
        "action_prob",
        "c_action_log_prob",
        "baseline",
        "baseline_enc",
        "entropy_loss",
        "reg_loss",
        "misc",
    ],
)

def compute_discrete_log_prob(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    assert len(logits.shape) == len(actions.shape) + 1
    has_dim = len(actions.shape) == 3
    end_dim = 2 if has_dim else 1
    log_prob = -torch.nn.CrossEntropyLoss(reduction="none")(
        input=torch.flatten(logits, 0, end_dim),
        target=torch.flatten(actions, 0, end_dim),
    )
    log_prob = log_prob.view_as(actions)
    if has_dim:
        log_prob = torch.sum(log_prob, dim=-1)
    return log_prob

def _infer_num_actions(action_space: Any) -> int:
    # Common cases: gym.spaces.Discrete, nested tuples/lists of spaces, or an int.
    if hasattr(action_space, "n"):
        return int(action_space.n)
    if isinstance(action_space, int):
        return action_space
    if isinstance(action_space, (list, tuple)) and action_space:
        cur: Any = action_space
        while isinstance(cur, (list, tuple)) and cur:
            cur = cur[0]
        if hasattr(cur, "n"):
            return int(cur.n)
        if isinstance(cur, int):
            return cur
    raise ValueError(f"Unsupported action_space for DRCNet reference: {type(action_space)!r}")

def _infer_real_states_shape(obs_space: Any) -> Tuple[int, ...]:
    # Expected to be (C, H, W) (or occasionally (H, W, C) in some wrappers).
    if isinstance(obs_space, dict) and "real_states" in obs_space:
        rs = obs_space["real_states"]
        if hasattr(rs, "shape"):
            return tuple(int(x) for x in rs.shape)

    spaces = getattr(obs_space, "spaces", None)
    if isinstance(spaces, dict) and "real_states" in spaces:
        rs = spaces["real_states"]
        if hasattr(rs, "shape"):
            return tuple(int(x) for x in rs.shape)

    if hasattr(obs_space, "shape"):
        return tuple(int(x) for x in obs_space.shape)

    if isinstance(obs_space, (list, tuple)) and obs_space and all(isinstance(x, int) for x in obs_space):
        return tuple(int(x) for x in obs_space)

    raise ValueError(f"Unsupported obs_space for DRCNet reference: {type(obs_space)!r}")

def sample(logits: torch.Tensor, greedy: bool, dim: int = -1) -> torch.Tensor:
    if not greedy:
        gumbel_noise = (
            torch.empty_like(logits)
            .uniform_()
            .clamp(1e-10, 1)
            .log()
            .neg_()
            .clamp(1e-10, 1)
            .log()
            .neg_()
        )
        sampled_action = (logits + gumbel_noise).argmax(dim=dim)
        return sampled_action.detach()
    return torch.argmax(logits, dim=dim)
