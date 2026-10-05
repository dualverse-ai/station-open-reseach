#!/usr/bin/env python3
"""Minimal DRC + checkpoint + intervention demo (hardcoded config).

Reference implementation for spec Section 3.2:
`demo_intervention.py`

This demo intentionally avoids paper-specific probe directions (no spoilers).
It shows:
1) initialize a DRC agent
2) load checkpoint
3) run env loop for return
4) apply a simple activation intervention and compare output
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import os

import gym_sokoban  # noqa: F401 (registers env ids)
import thinker
import thinker.util as util
from thinker.actor_net import DRCNet, sample


# -----------------------------
# Hardcoded demo config
# -----------------------------
ENV_ID = "Sokoban-valid-v0" # there are multiple envs
STATION_CKPT = Path(
    os.environ.get("DRC_CHECKPOINT_PATH", "checkpoints/ckp_actor_realstep250m.tar")
)
NUM_LAYERS = 3
NUM_TICKS = 3
MAX_STEPS = 200

# Intervention config
ALPHA = 0.5
PATCH_LAYER = 0
PATCH_STEPS = 25
VEC_SEED = 0
PATCH_LOCS = [(3, 3), (3, 4), (4, 3), (4, 4)]

@dataclass
class EpisodeResult:
    ep_return: float
    ep_len: int


class PatchDRC:
    """Patch ConvLSTM cell state c_next at chosen layer/locations."""

    def __init__(self, net: DRCNet):
        self.net = net

    def act(self, env_out, rnn_state, *, patch_info: dict, activ_ticks: list[int]):
        done = env_out.done
        t_dim, b_dim = done.shape

        x = self.net.normalize(env_out.real_states.float())
        x = torch.flatten(x, 0, 1)
        x_enc = self.net.encoder(x)
        core_input = x_enc.view(*((t_dim, b_dim) + x_enc.shape[1:]))

        reset = done.float()
        for x_single, reset_single in zip(core_input.unbind(), reset.unbind()):
            for tick in range(self.net.core.tran_t):
                if tick > 0:
                    reset_single = torch.zeros_like(reset_single)
                reset_single = reset_single.view(-1)

                if patch_info and (tick in activ_ticks):
                    out, rnn_state = self._forward_single_patch(
                        x=x_single,
                        core_state=rnn_state,
                        reset=reset_single,
                        patch_info=patch_info,
                    )
                else:
                    out, rnn_state = self.net.core.forward_single(
                        x_single, rnn_state, reset_single, reset_single
                    )

        # Same policy head path as DRCNet.forward.
        out = torch.flatten(out, 0, 1)
        out = torch.cat([x_enc, out], dim=1)
        out = torch.flatten(out, 1)
        final_out = torch.nn.functional.relu(self.net.final_layer(out))
        pri_logits = self.net.policy(final_out).view(t_dim * b_dim, self.net.dim_actions, self.net.num_actions)
        pri = sample(pri_logits, greedy=True, dim=-1).view(t_dim, b_dim, self.net.dim_actions)
        action = pri[-1, :, 0] if not self.net.tuple_action else pri[-1]
        return action, rnn_state

    def _forward_single_patch(self, *, x, core_state, reset, patch_info: dict):
        reset = reset.float()
        layer_n = 2
        out = core_state[(self.net.core.num_layers - 1) * layer_n] * (1 - reset).view(x.shape[0], 1, 1, 1)

        new_core_state = []
        for layer_i, cell in enumerate(self.net.core.layers):
            cell_input = torch.concat([x, out], dim=1)
            h_cur = core_state[layer_i * layer_n + 0] * (1 - reset.view(x.shape[0], 1, 1, 1))
            c_cur = core_state[layer_i * layer_n + 1] * (1 - reset.view(x.shape[0], 1, 1, 1))

            if layer_i in patch_info:
                h_next, c_next = self._forward_cell_patch(
                    convlstm_cell=cell,
                    input=cell_input,
                    h_cur=h_cur,
                    c_cur=c_cur,
                    layer_patch_info=patch_info[layer_i],
                )
            else:
                h_next, c_next, _, _ = cell(cell_input, h_cur, c_cur, None, None, None)

            new_core_state.append(h_next)
            new_core_state.append(c_next)
            out = h_next

        return out.unsqueeze(0), tuple(new_core_state)

    def _forward_cell_patch(self, *, convlstm_cell, input, h_cur, c_cur, layer_patch_info):
        combined = torch.cat([input, h_cur], dim=1)
        if convlstm_cell.pool_inject:
            combined = torch.cat([combined, convlstm_cell.proj_max_mean(h_cur)], dim=1)

        if convlstm_cell.linear:
            combined_conv = convlstm_cell.main(combined[:, :, 0, 0]).unsqueeze(-1).unsqueeze(-1)
        else:
            combined_conv = convlstm_cell.main(combined)

        cc_i, cc_f, cc_o, cc_g, _ = torch.split(combined_conv, convlstm_cell.embed_dim, dim=1)
        i_gate = torch.sigmoid(cc_i)
        f_gate = torch.sigmoid(cc_f)
        o_gate = torch.sigmoid(cc_o)
        g_gate = torch.tanh(cc_g)
        c_next = f_gate * c_cur + i_gate * g_gate

        for interv in layer_patch_info:
            vec = interv["vec"].to(c_next.device)
            alpha = float(interv["alpha"])
            for (y_idx, x_idx) in interv["locs"]:
                c_next[0, :, y_idx, x_idx] += alpha * vec

        h_next = o_gate * torch.tanh(c_next)
        return h_next, c_next


def make_flags():
    flags = util.create_setting(args=[], save_flags=False, wrapper_type=1)
    flags.mini = True
    flags.mini_unqtar = False
    flags.mini_unqbox = False
    return flags


def make_env(env_id: str, *, gpu: bool):
    return thinker.make(
        name=env_id,
        env_n=1,
        gpu=gpu,
        wrapper_type=1,
        has_model=False,
        train_model=False,
        parallel=False,
        save_flags=False,
        mini=True,
        mini_unqtar=False,
        mini_unqbox=False,
    )


@torch.no_grad()
def run_episode(*, env, net: DRCNet, flags, intervene: bool) -> EpisodeResult:
    rnn_state = net.initial_state(batch_size=1, device=env.device)
    try:
        state = env.reset()
    except EOFError as err:
        raise RuntimeError(
            "env.reset() failed (EOFError). Try a compatible mini-sokoban env id."
        ) from err

    env_out = util.init_env_out(state, flags, dim_actions=1, tuple_action=False)

    # No-spoiler direction: fixed random vector in hidden-channel space (32-dim).
    rng = torch.Generator(device="cpu").manual_seed(VEC_SEED)
    vec = torch.randn(32, generator=rng)
    patch_info = {PATCH_LAYER: [{"vec": vec, "locs": PATCH_LOCS, "alpha": ALPHA}]}
    activ_ticks = list(range(net.num_ticks))
    patcher = PatchDRC(net)

    ep_ret = 0.0
    ep_len = 0
    done = False

    while not done and ep_len < MAX_STEPS:
        if intervene and ep_len < PATCH_STEPS:
            action, rnn_state = patcher.act(
                env_out,
                rnn_state,
                patch_info=patch_info,
                activ_ticks=activ_ticks,
            )
        else:
            actor_out, rnn_state = net(env_out, rnn_state, greedy=True)
            action = actor_out.action

        state, reward, done, info = env.step(action)
        ep_ret += float(reward.item())
        ep_len += 1
        env_out = util.create_env_out(action, state, reward, done, info, flags)

    return EpisodeResult(ep_return=ep_ret, ep_len=ep_len)


def main() -> None:
    env = make_env(ENV_ID, gpu=torch.cuda.is_available())
    flags = make_flags()

    net = DRCNet(
        obs_space=env.observation_space,
        action_space=env.action_space,
        flags=flags,
        record_state=False,
        num_layers=NUM_LAYERS,
        num_ticks=NUM_TICKS,
    ).to(env.device)
    net.eval()

    ckpt_path = STATION_CKPT
    ckp = torch.load(ckpt_path, map_location=env.device, weights_only=False)
    net.load_state_dict(ckp["actor_net_state_dict"], strict=False)

    print(f"env_id={ENV_ID}")
    print(f"device={env.device}")
    print(f"ckpt={ckpt_path}")
    print(
        f"patch: layer={PATCH_LAYER}, alpha={ALPHA}, steps<{PATCH_STEPS}, "
        f"locs={PATCH_LOCS}, vec_seed={VEC_SEED}"
    )

    baseline = run_episode(env=env, net=net, flags=flags, intervene=False)
    print(f"baseline   return={baseline.ep_return:.3f} len={baseline.ep_len}")

    intervened = run_episode(env=env, net=net, flags=flags, intervene=True)
    print(f"intervene  return={intervened.ep_return:.3f} len={intervened.ep_len}")

    delta_return = abs(intervened.ep_return - baseline.ep_return)
    delta_len = abs(intervened.ep_len - baseline.ep_len)
    demo_success = (delta_return >= 0.1) or (delta_len >= 1)
    print(f"delta_return={delta_return:.3f} delta_len={delta_len}")
    print(f"demo_success={demo_success}")
    if not demo_success:
        print("note=no visible effect in this seed")
