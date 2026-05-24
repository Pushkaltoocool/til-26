"""Run the latest trained AE policy in the simulator with a pygame viewer."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import pygame
import torch

from train_rl_viewer import (
    ACTION_HISTORY_LEN,
    ACTION_NAMES,
    DQN,
    ROOT,
    TrainingViewer,
    adapt_state_dict_input_dim,
    as_scalar,
    default_config,
    legal_actions,
    obs_to_vector,
    select_action,
)
from til_environment.bomberman_env import Bomberman


class ObservationViewer(TrainingViewer):
    def __init__(self, env: Bomberman, fps: int, panel_width: int = 760) -> None:
        super().__init__(env, fps, panel_width=panel_width)
        self.json_font = pygame.font.Font("freesansbold.ttf", 11)

    def _draw_panel(
        self,
        x0: int,
        height: int,
        episode: int,
        epsilon: float,
        loss: float | None,
        replay_size: int,
    ) -> None:
        assert self.screen is not None
        pygame.draw.rect(self.screen, (18, 21, 25), (x0, 0, self.panel_width, height))
        pygame.draw.line(self.screen, (61, 67, 76), (x0, 0), (x0, height), 2)

        obs = self.env.observe(self.selected_agent)
        agent_view = np.asarray(obs["agent_viewcone"], dtype=np.float32)
        base_view = np.asarray(obs["base_viewcone"], dtype=np.float32)

        left_x = x0 + 18
        right_x = x0 + (self.panel_width // 2) + 10
        y = 18
        y = self._text(f"Selected: {self.selected_agent}", left_x, y, (238, 242, 247))
        y = self._text(f"Episode: {episode}", left_x, y, (195, 205, 216))
        y = self._text(
            f"Step: {int(as_scalar(obs['step']))}  Dir: {int(obs['direction'])}",
            left_x,
            y,
            (195, 205, 216),
        )
        y = self._text(
            f"Loc: {tuple(np.asarray(obs['location'], dtype=np.int64).reshape(-1)[:2])}",
            left_x,
            y,
            (195, 205, 216),
        )
        y = self._text(
            f"Base: {tuple(np.asarray(obs['base_location'], dtype=np.int64).reshape(-1)[:2])}",
            left_x,
            y,
            (195, 205, 216),
        )
        y = self._text(
            f"HP: {as_scalar(obs['health']):.0f}  Base HP: {as_scalar(obs['base_health']):.0f}",
            left_x,
            y,
            (195, 205, 216),
        )
        y = self._text(
            f"Frozen: {int(as_scalar(obs['frozen_ticks']))}  Bombs: {int(as_scalar(obs['team_bombs']))}",
            left_x,
            y,
            (195, 205, 216),
        )
        y = self._text(
            f"Resources: {as_scalar(obs['team_resources']):.2f}",
            left_x,
            y,
            (195, 205, 216),
        )
        mask_values = np.asarray(obs["action_mask"], dtype=np.int64).reshape(-1).tolist()
        legal = [ACTION_NAMES[i] for i in legal_actions(obs["action_mask"]).tolist()]
        y = self._text(f"Mask: {mask_values}", left_x, y + 4, (164, 174, 186))
        y = self._text("Legal: " + ", ".join(legal), left_x, y, (164, 174, 186))
        y = self._text(
            f"agent_viewcone: {tuple(agent_view.shape)}", left_x, y + 8, (238, 242, 247)
        )
        self._draw_viewcone(agent_view, left_x, y + 6, (self.panel_width // 2) - 34)

        right_y = 18
        right_y = self._text(
            f"base_viewcone: {tuple(base_view.shape)}", right_x, right_y, (238, 242, 247)
        )
        self._draw_viewcone(base_view, right_x, right_y + 6, (self.panel_width // 2) - 34)

        json_y = right_y + 6 + max(18, min(((self.panel_width // 2) - 34) // base_view.shape[1], 44)) * base_view.shape[0] + 42
        observation_json = json.dumps(self._serialise_observation(obs), indent=2)
        self._draw_multiline_text(observation_json, right_x, json_y, (186, 194, 204), height - json_y - 12)

    def _serialise_observation(self, obs: dict[str, object]) -> dict[str, object]:
        serialised: dict[str, object] = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                serialised[key] = value.tolist()
            else:
                serialised[key] = value
        return serialised

    def _draw_multiline_text(
        self,
        text: str,
        x: int,
        y: int,
        colour: tuple[int, int, int],
        max_height: int,
    ) -> None:
        assert self.screen is not None
        line_height = 14
        max_lines = max(max_height // line_height, 0)
        for index, line in enumerate(text.splitlines()[:max_lines]):
            rendered = self.json_font.render(line, True, colour)
            self.screen.blit(rendered, (x, y + index * line_height))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to load. Defaults to the newest .pt file under ae/models.",
    )
    parser.add_argument("--episodes", type=int, default=0, help="0 means run until you quit.")
    parser.add_argument("--novice", action="store_true", help="Use fixed novice map layout.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--window-size", type=int, default=768)
    return parser.parse_args()


def resolve_checkpoint(path: Path | None) -> Path:
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        return path

    models_dir = ROOT / "ae" / "models"
    checkpoints = sorted(models_dir.glob("*.pt"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint files found under {models_dir}")
    return checkpoints[0]


def make_env(args: argparse.Namespace) -> Bomberman:
    cfg = default_config()
    cfg.env.novice = bool(args.novice)
    cfg.env.render_mode = "rgb_array"
    cfg.renderer.window_size = int(args.window_size)
    cfg.renderer.render_fps = int(args.fps)
    return Bomberman(cfg)


def load_model_from_checkpoint(
    checkpoint_path: Path,
    input_dim: int,
    output_dim: int,
    device: torch.device,
) -> tuple[DQN, dict, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict, _ = adapt_state_dict_input_dim(checkpoint["model_state"], input_dim)
    first_layer = state_dict.get("net.0.weight")
    if first_layer is None:
        first_layer = next(iter(state_dict.values()))
    hidden_dim = int(first_layer.shape[0])
    model = DQN(input_dim, hidden_dim, output_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint, checkpoint_path.stat().st_mtime_ns


def maybe_reload_model(
    checkpoint_arg: Path | None,
    loaded_path: Path,
    loaded_mtime_ns: int,
    input_dim: int,
    output_dim: int,
    device: torch.device,
    current_model: DQN,
) -> tuple[DQN, Path, int, dict | None]:
    candidate = resolve_checkpoint(checkpoint_arg).resolve()
    candidate_mtime_ns = candidate.stat().st_mtime_ns
    if candidate == loaded_path and candidate_mtime_ns == loaded_mtime_ns:
        return current_model, loaded_path, loaded_mtime_ns, None
    try:
        model, checkpoint, new_mtime_ns = load_model_from_checkpoint(
            candidate, input_dim, output_dim, device
        )
    except Exception as exc:
        print(f"reload skipped (checkpoint busy/incomplete): {exc}")
        return current_model, loaded_path, loaded_mtime_ns, None
    return model, candidate, new_mtime_ns, checkpoint


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_checkpoint(args.checkpoint).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(args)
    env.reset(seed=args.seed)
    sample_obs = env.observe(env.possible_agents[0])
    input_dim = int(obs_to_vector(sample_obs, env.cfg, deque(maxlen=ACTION_HISTORY_LEN)).shape[0])
    output_dim = env.action_space().n
    model, checkpoint, loaded_mtime_ns = load_model_from_checkpoint(
        checkpoint_path, input_dim, output_dim, device
    )

    viewer = ObservationViewer(env, fps=args.fps)
    episode = 0
    running = True

    print(f"loaded checkpoint: {checkpoint_path}")
    print(f"checkpoint episode: {checkpoint.get('episode', '?')}")
    print("close the pygame window or press Q / Esc to stop")

    try:
        while running and (args.episodes == 0 or episode < args.episodes):
            model, checkpoint_path, loaded_mtime_ns, reloaded_checkpoint = maybe_reload_model(
                args.checkpoint,
                checkpoint_path,
                loaded_mtime_ns,
                input_dim,
                output_dim,
                device,
                model,
            )
            if reloaded_checkpoint is not None:
                print(f"reloaded checkpoint: {checkpoint_path}")
                print(f"checkpoint episode: {reloaded_checkpoint.get('episode', '?')}")

            episode += 1
            seed = None if args.seed is None else args.seed + episode
            env.reset(seed=seed)
            cumulative_rewards = {agent_id: 0.0 for agent_id in env.possible_agents}
            action_histories: dict[str, deque[int]] = {
                agent_id: deque(maxlen=ACTION_HISTORY_LEN) for agent_id in env.possible_agents
            }

            for agent_id in env.agent_iter():
                obs, _reward, termination, truncation, _info = env.last()
                was_last = env.agent_selector.is_last()

                if termination or truncation:
                    action = None
                else:
                    state = obs_to_vector(obs, env.cfg, action_histories.get(agent_id))
                    action = select_action(model, state, obs["action_mask"], 0.0, device)
                    action_histories[agent_id].append(int(action))

                env.step(action)

                if was_last:
                    for current_agent in env.possible_agents:
                        cumulative_rewards[current_agent] += float(env.rewards.get(current_agent, 0.0))
                    running = viewer.draw(episode, 0.0, None, 0)
                    if not running:
                        break

                if all(env.truncations.values()) or all(env.terminations.values()):
                    break

            print(
                f"episode={episode} total_reward={sum(cumulative_rewards.values()):.2f} "
                f"agent0_reward={cumulative_rewards.get(env.possible_agents[0], 0.0):.2f}"
            )
    finally:
        env.close()
        viewer.close()


if __name__ == "__main__":
    main()
