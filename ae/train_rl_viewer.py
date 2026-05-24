"""Train a shared DQN policy for AE while visualising the game with pygame.

This is a development tool. It is not used by the competition inference
server; trained weights still need to be wired into ae/src/ae_manager.py if you
want to submit them.
"""

from __future__ import annotations

import argparse
import heapq
import math
import multiprocessing as mp
import random
import sys
import threading
from collections import deque, namedtuple
from dataclasses import dataclass
from multiprocessing.connection import wait
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TIL_AE_ROOT = ROOT / "til-26-ae"
if str(TIL_AE_ROOT) not in sys.path:
    sys.path.insert(0, str(TIL_AE_ROOT))

import pygame
import torch
from torch import nn
from torch.nn import functional as F

from til_environment.actions import Action
from til_environment.bomberman_env import Bomberman
from til_environment.config import default_config, viewcone_tuple
from til_environment.entities import Agent
from til_environment.helpers import idx_to_view, view_to_world
from til_environment.observation import ViewChannel
from til_environment.types import Direction


Transition = namedtuple(
    "Transition",
    ["state", "action", "reward", "next_state", "done", "next_mask"],
)

ACTION_NAMES = {
    int(Action.FORWARD): "FORWARD",
    int(Action.BACKWARD): "BACKWARD",
    int(Action.LEFT): "LEFT",
    int(Action.RIGHT): "RIGHT",
    int(Action.STAY): "STAY",
    int(Action.PLACE_BOMB): "BOMB",
}
ACTION_HISTORY_LEN = 3
NAV_TARGET_COUNT = 3
NAV_FEATURE_DIM = NAV_TARGET_COUNT * (len(Action) + 1)
UNKNOWN_WALL = -1
NO_WALL = 0
HAS_WALL = 1
DIRECTION_VECTORS = {
    int(Direction.RIGHT): (1, 0),
    int(Direction.DOWN): (0, 1),
    int(Direction.LEFT): (-1, 0),
    int(Direction.UP): (0, -1),
}
DIRECTION_FROM_DELTA = {delta: direction for direction, delta in DIRECTION_VECTORS.items()}


@dataclass
class PendingTransition:
    state: np.ndarray
    action: int


@dataclass
class EvalResult:
    avg_reward: float
    avg_agent0_reward: float
    avg_freezes: float


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self._items: deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._items)

    def add(self, *transition: Any) -> None:
        self._items.append(Transition(*transition))

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self._items, batch_size)


class DQN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def as_scalar(value: Any, default: float = 0.0) -> float:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return default
    return float(arr[0])


def encode_action_history(action_history: deque[int] | list[int] | None) -> np.ndarray:
    encoded = np.zeros((ACTION_HISTORY_LEN, len(Action)), dtype=np.float32)
    if action_history:
        history = list(action_history)[-ACTION_HISTORY_LEN:]
        offset = ACTION_HISTORY_LEN - len(history)
        for idx, action in enumerate(history):
            action_int = int(action)
            if 0 <= action_int < len(Action):
                encoded[offset + idx, action_int] = 1.0
    return encoded.reshape(-1)


class InternalMap:
    def __init__(self, grid_size: int) -> None:
        self.grid_size = int(grid_size)
        self.seen = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.walls = np.full((self.grid_size, self.grid_size, 4), UNKNOWN_WALL, dtype=np.int8)
        self.enemy_agents: set[tuple[int, int]] = set()
        self.enemy_bases: set[tuple[int, int]] = set()
        self._path_cache: dict[tuple[tuple[int, int], tuple[int, int]], list[tuple[int, int]] | None] = {}

    def reset(self) -> None:
        self.seen.fill(False)
        self.walls.fill(UNKNOWN_WALL)
        self.enemy_agents.clear()
        self.enemy_bases.clear()
        self._path_cache.clear()

    def update(self, obs: dict[str, Any], cfg: Any) -> None:
        agent_loc = np.asarray(obs["location"], dtype=np.int64).reshape(-1)[:2]
        base_loc = np.asarray(obs["base_location"], dtype=np.int64).reshape(-1)[:2]
        direction = int(obs["direction"])
        agent_viewcone = np.asarray(obs["agent_viewcone"], dtype=np.float32)
        base_viewcone = np.asarray(obs["base_viewcone"], dtype=np.float32)

        self._update_agent_view(agent_viewcone, agent_loc, direction, viewcone_tuple(cfg.dynamics.vision))
        radius = max((base_viewcone.shape[0] - 1) // 2, 0)
        self._update_radius_view(base_viewcone, base_loc, radius)

    def navigation_features(self, obs: dict[str, Any]) -> np.ndarray:
        start = _as_int_xy(obs["location"])
        base = _as_int_xy(obs["base_location"])
        grid_norm = max(float(self.grid_size - 1), 1.0)
        direction = int(obs["direction"])
        target_groups = [
            [base],
            sorted(self.enemy_agents, key=lambda target: _manhattan(start, target)),
            sorted(self.enemy_bases, key=lambda target: _manhattan(start, target)),
        ]

        features: list[float] = []
        for targets in target_groups:
            action, distance = self._next_action_to_nearest(start, direction, targets)
            action_one_hot = np.zeros(len(Action), dtype=np.float32)
            if action is not None:
                action_one_hot[int(action)] = 1.0
            features.extend(float(value) for value in action_one_hot)
            features.append(distance / grid_norm if distance is not None else 1.0)
        return np.asarray(features, dtype=np.float32)

    def _update_agent_view(
        self,
        view: np.ndarray,
        agent_loc: np.ndarray,
        direction: int,
        viewcone: tuple[int, int, int, int],
    ) -> None:
        for idx in np.ndindex(view.shape[:2]):
            if view[idx][ViewChannel.VISIBLE] < 0.5:
                continue
            world = view_to_world(agent_loc, Direction(direction), idx_to_view(np.array(idx), viewcone))
            self._update_cell(view[idx], int(world[0]), int(world[1]))

    def _update_radius_view(self, view: np.ndarray, center: np.ndarray, radius: int) -> None:
        if view.size == 0:
            return
        for x, y in np.ndindex(view.shape[:2]):
            if view[x, y, ViewChannel.VISIBLE] < 0.5:
                continue
            wx = int(center[0]) + x - radius
            wy = int(center[1]) + y - radius
            self._update_cell(view[x, y], wx, wy)

    def _update_cell(self, tile: np.ndarray, x: int, y: int) -> None:
        if not self._in_bounds((x, y)):
            return
        self.seen[x, y] = True
        self._set_wall((x, y), int(Direction.RIGHT), tile[ViewChannel.WALL_RIGHT] > 0.5)
        self._set_wall((x, y), int(Direction.DOWN), tile[ViewChannel.WALL_DOWN] > 0.5)
        self._set_wall((x, y), int(Direction.LEFT), tile[ViewChannel.WALL_LEFT] > 0.5)
        self._set_wall((x, y), int(Direction.UP), tile[ViewChannel.WALL_UP] > 0.5)

        key = (x, y)
        if tile[ViewChannel.ENEMY_AGENT] > 0.5:
            self.enemy_agents.add(key)
        else:
            self.enemy_agents.discard(key)
        if tile[ViewChannel.ENEMY_BASE] > 0.5:
            self.enemy_bases.add(key)
        else:
            self.enemy_bases.discard(key)

    def _set_wall(self, cell: tuple[int, int], direction: int, has_wall: bool) -> None:
        if not self._in_bounds(cell):
            return
        x, y = cell
        old_value = self.walls[x, y, direction]
        new_value = HAS_WALL if has_wall else NO_WALL
        if old_value != new_value:
            self._path_cache.clear()
        self.walls[x, y, direction] = new_value
        dx, dy = DIRECTION_VECTORS[direction]
        other = (x + dx, y + dy)
        if self._in_bounds(other):
            self.walls[other[0], other[1], (direction + 2) % 4] = new_value

    def _next_action_to_nearest(
        self,
        start: tuple[int, int],
        direction: int,
        targets: list[tuple[int, int]],
    ) -> tuple[int | None, float | None]:
        best_path: list[tuple[int, int]] | None = None
        for target in targets:
            path = self._a_star(start, target)
            if path is None:
                continue
            if best_path is None or len(path) < len(best_path):
                best_path = path
        if best_path is None:
            return None, None
        distance = float(max(len(best_path) - 1, 0))
        if len(best_path) <= 1:
            return int(Action.STAY), distance
        next_cell = best_path[1]
        step_dir = DIRECTION_FROM_DELTA.get((next_cell[0] - start[0], next_cell[1] - start[1]))
        if step_dir is None:
            return int(Action.STAY), distance
        if step_dir == direction:
            return int(Action.FORWARD), distance
        if step_dir == (direction + 2) % 4:
            return int(Action.BACKWARD), distance
        if step_dir == (direction + 3) % 4:
            return int(Action.LEFT), distance
        return int(Action.RIGHT), distance

    def _a_star(self, start: tuple[int, int], target: tuple[int, int]) -> list[tuple[int, int]] | None:
        if not self._in_bounds(start) or not self._in_bounds(target):
            return None
        cache_key = (start, target)
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]

        frontier: list[tuple[int, int, tuple[int, int]]] = [(0, 0, start)]
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        cost_so_far: dict[tuple[int, int], int] = {start: 0}
        order = 0

        while frontier:
            _, _, current = heapq.heappop(frontier)
            if current == target:
                break
            for neighbor in self._neighbors(current):
                new_cost = cost_so_far[current] + 1
                if neighbor in cost_so_far and new_cost >= cost_so_far[neighbor]:
                    continue
                cost_so_far[neighbor] = new_cost
                order += 1
                priority = new_cost + _manhattan(neighbor, target)
                heapq.heappush(frontier, (priority, order, neighbor))
                came_from[neighbor] = current

        if target not in came_from:
            self._path_cache[cache_key] = None
            return None
        path: list[tuple[int, int]] = []
        current: tuple[int, int] | None = target
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        self._path_cache[cache_key] = path
        return path

    def _neighbors(self, cell: tuple[int, int]) -> list[tuple[int, int]]:
        neighbors: list[tuple[int, int]] = []
        x, y = cell
        for direction, (dx, dy) in DIRECTION_VECTORS.items():
            next_cell = (x + dx, y + dy)
            if not self._in_bounds(next_cell):
                continue
            if self.walls[x, y, direction] == HAS_WALL:
                continue
            neighbors.append(next_cell)
        return neighbors

    def _in_bounds(self, cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] < self.grid_size and 0 <= cell[1] < self.grid_size


def _as_int_xy(value: Any) -> tuple[int, int]:
    arr = np.asarray(value, dtype=np.int64).reshape(-1)
    return int(arr[0]), int(arr[1])


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def obs_to_vector(
    obs: dict[str, Any],
    cfg: Any,
    action_history: deque[int] | list[int] | None = None,
    internal_map: InternalMap | None = None,
) -> np.ndarray:
    direction = int(obs["direction"])
    direction_one_hot = np.zeros(4, dtype=np.float32)
    if 0 <= direction < 4:
        direction_one_hot[direction] = 1.0

    grid_norm = max(float(cfg.env.grid_size - 1), 1.0)
    scalars = np.array(
        [
            *direction_one_hot,
            *(np.asarray(obs["location"], dtype=np.float32).reshape(-1) / grid_norm),
            *(np.asarray(obs["base_location"], dtype=np.float32).reshape(-1) / grid_norm),
            as_scalar(obs["health"]) / max(float(cfg.entities.agent.max_health), 1.0),
            as_scalar(obs["frozen_ticks"]) / max(float(cfg.entities.agent.freeze_turns), 1.0),
            as_scalar(obs["base_health"]) / max(float(cfg.entities.base.max_health), 1.0),
            as_scalar(obs["team_resources"]) / max(float(cfg.resources.max_team_resources), 1.0),
            as_scalar(obs["team_bombs"]) / max(float(cfg.resources.max_team_bombs), 1.0),
            as_scalar(obs["step"]) / max(float(cfg.env.num_iters), 1.0),
            *np.asarray(obs["action_mask"], dtype=np.float32).reshape(-1),
            *encode_action_history(action_history),
            *(
                internal_map.navigation_features(obs)
                if internal_map is not None
                else np.zeros(NAV_FEATURE_DIM, dtype=np.float32)
            ),
        ],
        dtype=np.float32,
    )

    return np.concatenate(
        [
            np.asarray(obs["agent_viewcone"], dtype=np.float32).reshape(-1),
            np.asarray(obs["base_viewcone"], dtype=np.float32).reshape(-1),
            scalars,
        ]
    )


def legal_actions(mask: Any) -> np.ndarray:
    mask_arr = np.asarray(mask, dtype=np.float32).reshape(-1)
    actions = np.flatnonzero(mask_arr > 0)
    if actions.size == 0:
        return np.array([int(Action.STAY)], dtype=np.int64)
    return actions.astype(np.int64)


def select_action(
    model: DQN,
    state: np.ndarray,
    mask: Any,
    epsilon: float,
    device: torch.device,
) -> int:
    choices = legal_actions(mask)
    if random.random() < epsilon:
        return int(random.choice(choices))

    with torch.no_grad():
        q_values = model(torch.from_numpy(state).float().unsqueeze(0).to(device))[0]
    mask_tensor = torch.zeros_like(q_values, dtype=torch.bool)
    mask_tensor[torch.as_tensor(choices, device=device)] = True
    q_values = q_values.masked_fill(~mask_tensor, -1e9)
    return int(torch.argmax(q_values).item())


def optimise(
    policy: DQN,
    target: DQN,
    optimiser: torch.optim.Optimizer,
    replay: ReplayBuffer,
    batch_size: int,
    gamma: float,
    device: torch.device,
) -> float | None:
    if len(replay) < batch_size:
        return None

    batch = replay.sample(batch_size)
    states = torch.tensor(np.stack([t.state for t in batch]), dtype=torch.float32, device=device)
    actions = torch.tensor([t.action for t in batch], dtype=torch.int64, device=device)
    rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device)
    next_states = torch.tensor(
        np.stack([t.next_state for t in batch]), dtype=torch.float32, device=device
    )
    dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device)
    next_masks = torch.tensor(
        np.stack([np.asarray(t.next_mask, dtype=np.float32).reshape(-1) for t in batch]),
        dtype=torch.float32,
        device=device,
    )

    q_values = policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        policy_next_q = policy(next_states).masked_fill(next_masks <= 0, -1e9)
        next_actions = policy_next_q.argmax(dim=1, keepdim=True)
        has_next_action = (next_masks > 0).any(dim=1)
        next_q = target(next_states).gather(1, next_actions).squeeze(1)
        max_next_q = torch.where(has_next_action, next_q, torch.zeros_like(rewards))
        targets = rewards + gamma * max_next_q * (1.0 - dones)

    loss = F.smooth_l1_loss(q_values, targets)
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    optimiser.step()
    return float(loss.item())


def adapt_state_dict_input_dim(
    model_state: dict[str, torch.Tensor], input_dim: int
) -> tuple[dict[str, torch.Tensor], bool]:
    weight_key = "net.0.weight"
    if weight_key not in model_state:
        return model_state, False
    first_weight = model_state[weight_key]
    old_input_dim = int(first_weight.shape[1])
    if old_input_dim == input_dim:
        return model_state, False

    adjusted = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in model_state.items()
    }
    if old_input_dim < input_dim:
        pad = torch.zeros(
            (first_weight.shape[0], input_dim - old_input_dim),
            dtype=first_weight.dtype,
            device=first_weight.device,
        )
        adjusted[weight_key] = torch.cat([first_weight, pad], dim=1)
    else:
        adjusted[weight_key] = first_weight[:, :input_dim]
    return adjusted, True


def cpu_state_dict(model: DQN) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


class TrainingViewer:
    def __init__(self, env: Bomberman, fps: int, panel_width: int = 360) -> None:
        pygame.init()
        self.env = env
        self.fps = fps
        self.panel_width = panel_width
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font("freesansbold.ttf", 13)
        self.small_font = pygame.font.Font("freesansbold.ttf", 10)
        self.screen: pygame.Surface | None = None
        self.selected_agent = env.possible_agents[0]

    def handle_events(self, frame_width: int) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                return False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                x, y = event.pos
                if x < frame_width:
                    entity = self.env.renderer.hit_test(x, y, self.env.dynamics.registry)
                    if isinstance(entity, Agent):
                        self.selected_agent = entity.entity_id
        return True

    def draw(
        self,
        episode: int,
        epsilon: float,
        loss: float | None,
        replay_size: int,
    ) -> bool:
        frame = self.env.render(selected_agent_id=self.selected_agent)
        if frame is None:
            return True

        height, frame_width = frame.shape[:2]
        if self.screen is None:
            self.screen = pygame.display.set_mode((frame_width + self.panel_width, height))
            pygame.display.set_caption("AE DQN training viewer")

        if not self.handle_events(frame_width):
            return False

        surface = pygame.surfarray.make_surface(np.swapaxes(frame, 0, 1))
        self.screen.blit(surface, (0, 0))
        self._draw_panel(frame_width, height, episode, epsilon, loss, replay_size)
        pygame.display.flip()
        self.clock.tick(self.fps)
        return True

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
        y = 18
        y = self._text(f"Selected: {self.selected_agent}", x0 + 18, y, (238, 242, 247))
        y = self._text(f"Episode: {episode}", x0 + 18, y, (195, 205, 216))
        y = self._text(f"Epsilon: {epsilon:.3f}", x0 + 18, y, (195, 205, 216))
        loss_text = "warming up" if loss is None else f"{loss:.4f}"
        y = self._text(f"Loss: {loss_text}", x0 + 18, y, (195, 205, 216))
        y = self._text(f"Replay: {replay_size}", x0 + 18, y, (195, 205, 216))
        y = self._text(
            f"HP {as_scalar(obs['health']):.0f}  Frozen {int(as_scalar(obs['frozen_ticks']))}",
            x0 + 18,
            y + 6,
            (195, 205, 216),
        )
        mask_names = [
            ACTION_NAMES[i]
            for i in legal_actions(obs["action_mask"]).tolist()
        ]
        y = self._text("Legal: " + ", ".join(mask_names), x0 + 18, y, (164, 174, 186))

        self._draw_viewcone(
            np.asarray(obs["agent_viewcone"], dtype=np.float32),
            x0 + 18,
            y + 16,
            self.panel_width - 36,
        )

    def _text(self, text: str, x: int, y: int, colour: tuple[int, int, int]) -> int:
        assert self.screen is not None
        rendered = self.font.render(text, True, colour)
        self.screen.blit(rendered, (x, y))
        return y + 19

    def _draw_viewcone(self, view: np.ndarray, x: int, y: int, width: int) -> None:
        assert self.screen is not None
        rows, cols, _channels = view.shape
        cell = max(18, min(width // cols, 44))
        panel_w = cell * cols
        panel_h = cell * rows

        pygame.draw.rect(self.screen, (10, 12, 15), (x - 2, y - 2, panel_w + 4, panel_h + 4))
        for row in range(rows):
            for col in range(cols):
                tile = view[row, col]
                rect = pygame.Rect(x + col * cell, y + row * cell, cell, cell)
                colour = self._view_cell_colour(tile)
                pygame.draw.rect(self.screen, colour, rect)
                pygame.draw.rect(self.screen, (48, 54, 61), rect, 1)
                self._draw_view_cell_glyph(tile, rect)
                self._draw_view_cell_walls(tile, rect)

        legend_y = y + panel_h + 12
        self._legend_item("A", (75, 190, 210), x, legend_y)
        self._legend_item("E", (215, 76, 76), x + 58, legend_y)
        self._legend_item("B", (230, 134, 70), x + 116, legend_y)
        self._legend_item("R", (86, 179, 118), x + 174, legend_y)
        self._legend_item("M", (225, 199, 82), x + 232, legend_y)

    def _legend_item(self, label: str, colour: tuple[int, int, int], x: int, y: int) -> None:
        assert self.screen is not None
        pygame.draw.rect(self.screen, colour, (x, y, 18, 18))
        self.screen.blit(self.small_font.render(label, True, (238, 242, 247)), (x + 24, y + 3))

    def _view_cell_colour(self, tile: np.ndarray) -> tuple[int, int, int]:
        if tile[ViewChannel.VISIBLE] <= 0:
            return (31, 35, 42)
        if tile[ViewChannel.TILE_MISSION] > 0:
            return (129, 112, 42)
        if tile[ViewChannel.TILE_RECON] > 0:
            return (48, 92, 139)
        if tile[ViewChannel.TILE_RESOURCE] > 0:
            return (43, 104, 69)
        return (48, 55, 63)

    def _draw_view_cell_glyph(self, tile: np.ndarray, rect: pygame.Rect) -> None:
        label = ""
        colour = (238, 242, 247)
        if tile[ViewChannel.ALLY_AGENT] > 0:
            label, colour = "A", (75, 190, 210)
        elif tile[ViewChannel.ENEMY_AGENT] > 0:
            label, colour = "E", (215, 76, 76)
        elif tile[ViewChannel.ALLY_BASE] > 0 or tile[ViewChannel.ENEMY_BASE] > 0:
            label, colour = "B", (230, 134, 70)
        elif tile[ViewChannel.ALLY_BOMB] > 0 or tile[ViewChannel.ENEMY_BOMB] > 0:
            label, colour = "*", (235, 122, 57)
        elif tile[ViewChannel.TILE_RESOURCE] > 0:
            label, colour = "R", (172, 228, 175)
        elif tile[ViewChannel.TILE_MISSION] > 0:
            label, colour = "M", (245, 220, 112)
        elif tile[ViewChannel.TILE_RECON] > 0:
            label, colour = "C", (112, 181, 246)

        if not label:
            return
        rendered = self.font.render(label, True, colour)
        x = rect.centerx - rendered.get_width() // 2
        y = rect.centery - rendered.get_height() // 2
        self.screen.blit(rendered, (x, y))

    def _draw_view_cell_walls(self, tile: np.ndarray, rect: pygame.Rect) -> None:
        wall_colour = (225, 230, 236)
        destr_colour = (245, 171, 74)
        wall_pairs = [
            (ViewChannel.WALL_RIGHT, ViewChannel.DESTR_WALL_RIGHT, rect.topright, rect.bottomright),
            (ViewChannel.WALL_DOWN, ViewChannel.DESTR_WALL_DOWN, rect.bottomleft, rect.bottomright),
            (ViewChannel.WALL_LEFT, ViewChannel.DESTR_WALL_LEFT, rect.topleft, rect.bottomleft),
            (ViewChannel.WALL_UP, ViewChannel.DESTR_WALL_UP, rect.topleft, rect.topright),
        ]
        for wall_ch, destr_ch, start, end in wall_pairs:
            if tile[wall_ch] > 0:
                colour = destr_colour if tile[destr_ch] > 0 else wall_colour
                pygame.draw.line(self.screen, colour, start, end, 3)

    def close(self) -> None:
        pygame.quit()


def finish_round_transitions(
    env: Bomberman,
    cfg: Any,
    replay: ReplayBuffer,
    pending: dict[str, PendingTransition],
    action_histories: dict[str, deque[int]],
    internal_maps: dict[str, InternalMap],
    forward_backward_penalty: float,
    left_right_penalty: float,
    same_turn_penalty: float,
) -> float:
    transitions = collect_round_transitions(
        env,
        cfg,
        pending,
        action_histories,
        internal_maps,
        forward_backward_penalty,
        left_right_penalty,
        same_turn_penalty,
    )
    for transition in transitions:
        replay.add(*transition)
    pending.clear()
    return float(sum(float(transition.reward) for transition in transitions))


def collect_round_transitions(
    env: Bomberman,
    cfg: Any,
    pending: dict[str, PendingTransition],
    action_histories: dict[str, deque[int]],
    internal_maps: dict[str, InternalMap],
    forward_backward_penalty: float,
    left_right_penalty: float,
    same_turn_penalty: float,
) -> list[Transition]:
    done = all(env.truncations.values()) or all(env.terminations.values())
    transitions: list[Transition] = []
    for agent_id, item in pending.items():
        next_obs = env.observations[agent_id]
        internal_map = internal_maps.setdefault(agent_id, InternalMap(int(cfg.env.grid_size)))
        internal_map.update(next_obs, cfg)
        reward = float(env.rewards.get(agent_id, 0.0))
        history = action_histories.get(agent_id)
        if history is not None and len(history) >= 2:
            prev_action = int(history[-2])
            curr_action = int(history[-1])
            if (
                (prev_action == int(Action.FORWARD) and curr_action == int(Action.BACKWARD))
                or (prev_action == int(Action.BACKWARD) and curr_action == int(Action.FORWARD))
            ):
                reward += forward_backward_penalty
            if (
                (prev_action == int(Action.LEFT) and curr_action == int(Action.RIGHT))
                or (prev_action == int(Action.RIGHT) and curr_action == int(Action.LEFT))
            ):
                reward += left_right_penalty
            if prev_action == curr_action and curr_action in (int(Action.LEFT), int(Action.RIGHT)):
                reward += same_turn_penalty

        transitions.append(
            Transition(
                item.state,
                item.action,
                reward,
                obs_to_vector(next_obs, cfg, action_histories.get(agent_id), internal_map),
                done,
                np.asarray(next_obs["action_mask"], dtype=np.float32).reshape(-1),
            )
        )
    return transitions


def rollout_worker(
    worker_id: int,
    args_dict: dict[str, Any],
    input_dim: int,
    conn: mp.connection.Connection,
) -> None:
    torch.set_num_threads(1)
    args = argparse.Namespace(**args_dict)
    args.headless = True
    local_device = torch.device("cpu")
    env, cfg = make_env(args)
    model = DQN(input_dim, args.hidden_dim, len(Action)).to(local_device)
    epsilon = args.epsilon_start
    episode_index = 0

    try:
        init_msg = conn.recv()
        if init_msg["cmd"] != "weights":
            raise ValueError("worker expected initial weights message")
        model.load_state_dict(init_msg["state_dict"])
        model.eval()
        epsilon = float(init_msg["epsilon"])

        while True:
            if conn.poll():
                msg = conn.recv()
                if msg["cmd"] == "stop":
                    return
                if msg["cmd"] == "weights":
                    model.load_state_dict(msg["state_dict"])
                    model.eval()
                    epsilon = float(msg["epsilon"])

            episode_index += 1
            seed = None if args.seed is None else args.seed + worker_id * 1_000_000 + episode_index
            env.reset(seed=seed)
            pending: dict[str, PendingTransition] = {}
            action_histories: dict[str, deque[int]] = {
                agent_id: deque(maxlen=ACTION_HISTORY_LEN) for agent_id in env.possible_agents
            }
            internal_maps: dict[str, InternalMap] = {
                agent_id: InternalMap(int(cfg.env.grid_size)) for agent_id in env.possible_agents
            }
            transitions: list[Transition] = []
            episode_reward = 0.0
            rounds = 0

            for agent_id in env.agent_iter():
                obs, _reward, termination, truncation, _info = env.last()
                was_last = env.agent_selector.is_last()

                if termination or truncation:
                    action = None
                else:
                    internal_maps[agent_id].update(obs, cfg)
                    state = obs_to_vector(obs, cfg, action_histories.get(agent_id), internal_maps[agent_id])
                    action = select_action(model, state, obs["action_mask"], epsilon, local_device)
                    pending[agent_id] = PendingTransition(state=state, action=action)
                    action_histories[agent_id].append(int(action))

                env.step(action)

                if was_last:
                    round_transitions = collect_round_transitions(
                        env,
                        cfg,
                        pending,
                        action_histories,
                        internal_maps,
                        args.forward_backward_penalty,
                        args.left_right_penalty,
                        args.same_turn_penalty,
                    )
                    transitions.extend(round_transitions)
                    pending.clear()
                    episode_reward += sum(float(value) for value in env.rewards.values())
                    rounds += 1

                    if conn.poll():
                        msg = conn.recv()
                        if msg["cmd"] == "weights":
                            model.load_state_dict(msg["state_dict"])
                            model.eval()
                            epsilon = float(msg["epsilon"])

                if all(env.truncations.values()) or all(env.terminations.values()):
                    break

            conn.send(
                {
                    "type": "episode",
                    "worker_id": worker_id,
                    "episode_index": episode_index,
                    "episode_reward": episode_reward,
                    "rounds": rounds,
                    "transitions": transitions,
                }
            )
    finally:
        env.close()


def evaluate_policy(
    args: argparse.Namespace,
    model: DQN,
    input_dim: int,
    device: torch.device,
    base_seed: int | None,
) -> EvalResult:
    eval_args = argparse.Namespace(**vars(args))
    eval_args.headless = True
    env, cfg = make_env(eval_args)
    total_rewards: list[float] = []
    agent0_rewards: list[float] = []
    freeze_counts: list[int] = []

    try:
        for episode_idx in range(args.eval_episodes):
            seed = None if base_seed is None else base_seed + 100_000 + episode_idx
            env.reset(seed=seed)
            action_histories: dict[str, deque[int]] = {
                agent_id: deque(maxlen=ACTION_HISTORY_LEN) for agent_id in env.possible_agents
            }
            internal_maps: dict[str, InternalMap] = {
                agent_id: InternalMap(int(cfg.env.grid_size)) for agent_id in env.possible_agents
            }
            sample_agent = env.possible_agents[0]
            sample_obs = env.observe(sample_agent)
            internal_maps[sample_agent].update(sample_obs, cfg)
            if int(obs_to_vector(sample_obs, cfg, action_histories[sample_agent], internal_maps[sample_agent]).shape[0]) != input_dim:
                raise ValueError("evaluation observation size does not match training model")

            cumulative = {agent_id: 0.0 for agent_id in env.possible_agents}
            previous_frozen = {agent_id: 0 for agent_id in env.possible_agents}
            freezes = 0

            for agent_id in env.agent_iter():
                obs, _reward, termination, truncation, _info = env.last()
                was_last = env.agent_selector.is_last()

                if termination or truncation:
                    action = None
                else:
                    internal_maps[agent_id].update(obs, cfg)
                    state = obs_to_vector(obs, cfg, action_histories.get(agent_id), internal_maps[agent_id])
                    action = select_action(model, state, obs["action_mask"], 0.0, device)
                    action_histories[agent_id].append(int(action))

                env.step(action)

                if was_last:
                    for current_agent in env.possible_agents:
                        cumulative[current_agent] += float(env.rewards.get(current_agent, 0.0))
                        frozen = int(as_scalar(env.observations[current_agent]["frozen_ticks"]))
                        if previous_frozen[current_agent] == 0 and frozen > 0:
                            freezes += 1
                        previous_frozen[current_agent] = frozen

                if all(env.truncations.values()) or all(env.terminations.values()):
                    break

            total_rewards.append(sum(cumulative.values()))
            agent0_rewards.append(cumulative.get(env.possible_agents[0], 0.0))
            freeze_counts.append(freezes)
    finally:
        env.close()

    return EvalResult(
        avg_reward=float(np.mean(total_rewards)) if total_rewards else 0.0,
        avg_agent0_reward=float(np.mean(agent0_rewards)) if agent0_rewards else 0.0,
        avg_freezes=float(np.mean(freeze_counts)) if freeze_counts else 0.0,
    )


class AsyncEvalManager:
    """Runs evaluation episodes on a background thread using a CPU copy of the policy.

    Skips a new eval if the previous one is still running, so a slow eval cannot
    block the rollout loop indefinitely.
    """

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._best_reward = float("-inf")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def launch(
        self,
        args: argparse.Namespace,
        input_dim: int,
        weights: dict[str, torch.Tensor],
        checkpoint: dict[str, Any],
        base_seed: int | None,
        episode: int,
    ) -> bool:
        if self.is_running():
            return False
        self._thread = threading.Thread(
            target=self._run,
            args=(args, input_dim, weights, checkpoint, base_seed, episode),
            daemon=True,
        )
        self._thread.start()
        return True

    def _run(
        self,
        args: argparse.Namespace,
        input_dim: int,
        weights: dict[str, torch.Tensor],
        checkpoint: dict[str, Any],
        base_seed: int | None,
        episode: int,
    ) -> None:
        try:
            eval_model = DQN(input_dim, args.hidden_dim, len(Action))
            eval_model.load_state_dict(weights)
            eval_model.eval()
            result = evaluate_policy(args, eval_model, input_dim, torch.device("cpu"), base_seed)
            print(
                f"eval episode={episode} avg_reward={result.avg_reward:.2f} "
                f"agent0={result.avg_agent0_reward:.2f} freezes={result.avg_freezes:.2f}"
            )
            with self._lock:
                if result.avg_reward > self._best_reward:
                    self._best_reward = result.avg_reward
                    args.best_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(checkpoint, args.best_checkpoint)
                    print(f"new best checkpoint: {args.best_checkpoint}")
        except Exception as exc:
            print(f"eval episode={episode} failed: {exc}")

    def wait(self) -> None:
        if self._thread is not None:
            self._thread.join()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--novice", action="store_true", help="Use fixed novice map layout.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--headless", action="store_true", help="Train without the pygame viewer.")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--window-size", type=int, default=768)
    parser.add_argument(
        "--forward-backward-penalty",
        type=float,
        default=-20.0,
        help="Penalty when an agent alternates FORWARD and BACKWARD on consecutive actions.",
    )
    parser.add_argument(
        "--left-right-penalty",
        type=float,
        default=-20.0,
        help="Penalty when an agent alternates LEFT and RIGHT on consecutive actions.",
    )
    parser.add_argument(
        "--same-turn-penalty",
        type=float,
        default=-20.0,
        help="Penalty when an agent turns the same direction (LEFT/LEFT or RIGHT/RIGHT) consecutively.",
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel rollout workers for headless training.")
    parser.add_argument("--updates-per-round", type=int, default=1)
    parser.add_argument("--target-sync-rounds", type=int, default=50)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-rounds", type=int, default=5000)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "ae" / "models" / "dqn_viewer.pt")
    parser.add_argument(
        "--best-checkpoint",
        type=Path,
        default=ROOT / "ae" / "models" / "dqn_viewer_best.pt",
    )
    parser.add_argument("--save-every", type=int, default=25, help="Save every N episodes; 0 disables saves.")
    parser.add_argument("--eval-every", type=int, default=25, help="Evaluate every N episodes; 0 disables eval.")
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--resume", action="store_true", help="Load --checkpoint before training.")

    parser.add_argument("--env-agent-collide-wall", type=float, default=0.0)
    parser.add_argument("--env-agent-collide-agent", type=float, default=0.0)
    parser.add_argument("--env-collect-mission", type=float, default=5.0)
    parser.add_argument("--env-collect-recon", type=float, default=1.0)
    parser.add_argument("--env-collect-resource", type=float, default=2.0)
    parser.add_argument("--env-attack-damage", type=float, default=1.0)
    parser.add_argument("--env-attack-kill", type=float, default=30.0)
    parser.add_argument("--env-destroy-wall", type=float, default=0.0)
    parser.add_argument("--env-destroy-enemy-base", type=float, default=50.0)
    parser.add_argument("--env-own-base-destroyed", type=float, default=-50.0)
    parser.add_argument("--env-step-penalty", type=float, default=2)
    parser.add_argument("--env-stationary-penalty", type=float, default=-10.0)
    parser.add_argument("--env-invalid-action", type=float, default=-1.0)
    parser.add_argument("--env-truncation", type=float, default=0.0)
    return parser.parse_args()


def apply_env_reward_overrides(cfg: Any, args: argparse.Namespace) -> None:
    reward_overrides = {
        "agent_collide_wall": args.env_agent_collide_wall,
        "agent_collide_agent": args.env_agent_collide_agent,
        "collect_mission": args.env_collect_mission,
        "collect_recon": args.env_collect_recon,
        "collect_resource": args.env_collect_resource,
        "attack_damage": args.env_attack_damage,
        "attack_kill": args.env_attack_kill,
        "destroy_wall": args.env_destroy_wall,
        "destroy_enemy_base": args.env_destroy_enemy_base,
        "own_base_destroyed": args.env_own_base_destroyed,
        "step_penalty": args.env_step_penalty,
        "stationary_penalty": args.env_stationary_penalty,
        "invalid_action": args.env_invalid_action,
        "truncation": args.env_truncation,
    }
    changed = []
    for key, value in reward_overrides.items():
        if value is None:
            continue
        current = float(getattr(cfg.rewards, key))
        target = float(value)
        if target != current:
            setattr(cfg.rewards, key, target)
            changed.append(f"{key}={target}")
    if changed:
        print("env reward overrides: " + ", ".join(changed))


def make_env(args: argparse.Namespace) -> tuple[Bomberman, Any]:
    cfg = default_config()
    cfg.env.novice = bool(args.novice)
    cfg.env.render_mode = None if args.headless else "rgb_array"
    cfg.renderer.window_size = int(args.window_size)
    cfg.renderer.render_fps = int(args.fps)
    apply_env_reward_overrides(cfg, args)
    return Bomberman(cfg), cfg


def save_checkpoint(path: Path, model: DQN, optimiser: torch.optim.Optimizer, episode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "episode": episode,
            "model_state": model.state_dict(),
            "optimiser_state": optimiser.state_dict(),
        },
        path,
    )


def epsilon_for_round(args: argparse.Namespace, round_index: int) -> float:
    decay_rate = max(float(args.epsilon_decay_rounds), 1.0)
    return args.epsilon_end + (args.epsilon_start - args.epsilon_end) * math.exp(-round_index / decay_rate)


def train_parallel(
    args: argparse.Namespace,
    device: torch.device,
    input_dim: int,
    policy: DQN,
    target: DQN,
    optimiser: torch.optim.Optimizer,
) -> None:
    if args.num_workers < 2:
        raise ValueError("parallel trainer requires at least 2 workers")
    if not args.headless:
        raise ValueError("parallel rollout workers require --headless")

    replay = ReplayBuffer(args.buffer_size)
    latest_loss: float | None = None
    round_index = 0
    episodes_completed = 0
    next_target_sync = args.target_sync_rounds
    ctx = mp.get_context("spawn")
    workers: list[tuple[mp.Process, mp.connection.Connection]] = []
    args_dict = vars(args).copy()
    eval_manager = AsyncEvalManager()

    try:
        for worker_id in range(args.num_workers):
            parent_conn, child_conn = ctx.Pipe()
            process = ctx.Process(
                target=rollout_worker,
                args=(worker_id, args_dict, input_dim, child_conn),
            )
            process.start()
            child_conn.close()
            workers.append((process, parent_conn))

        initial_weights = cpu_state_dict(policy)
        initial_epsilon = epsilon_for_round(args, round_index)
        for _process, conn in workers:
            conn.send({"cmd": "weights", "state_dict": initial_weights, "epsilon": initial_epsilon})

        while episodes_completed < args.episodes:
            ready_conns = wait([conn for _process, conn in workers])
            arrived: list[tuple[mp.connection.Connection, dict[str, Any]]] = []
            for conn in ready_conns:
                try:
                    payload = conn.recv()
                except EOFError:
                    continue
                if payload.get("type") != "episode":
                    continue
                arrived.append((conn, payload))

            if not arrived:
                continue

            total_rounds = 0
            for _conn, payload in arrived:
                for transition in payload["transitions"]:
                    replay.add(*transition)
                total_rounds += int(payload["rounds"])

            round_index += total_rounds
            update_count = max(1, total_rounds) * args.updates_per_round
            for _ in range(update_count):
                loss = optimise(
                    policy,
                    target,
                    optimiser,
                    replay,
                    args.batch_size,
                    args.gamma,
                    device,
                )
                if loss is not None:
                    latest_loss = loss

            while round_index >= next_target_sync:
                target.load_state_dict(policy.state_dict())
                next_target_sync += args.target_sync_rounds

            weights_snapshot = cpu_state_dict(policy)
            epsilon_snapshot = epsilon_for_round(args, round_index)
            sync_msg = {
                "cmd": "weights",
                "state_dict": weights_snapshot,
                "epsilon": epsilon_snapshot,
            }
            for conn, _payload in arrived:
                conn.send(sync_msg)

            for _conn, payload in arrived:
                episodes_completed += 1
                episode_reward = float(payload["episode_reward"])
                env_score = episode_reward / 1000.0
                print(
                    f"episode={episodes_completed} worker={payload['worker_id']} "
                    f"env_reward={episode_reward:.2f} "
                    f"env_score={env_score:.4f} "
                    f"epsilon={epsilon_snapshot:.3f} "
                    f"replay={len(replay)} loss={latest_loss if latest_loss is not None else 'warmup'}"
                )

                if args.save_every and episodes_completed % args.save_every == 0:
                    save_checkpoint(args.checkpoint, policy, optimiser, episodes_completed)

                if args.eval_every and episodes_completed % args.eval_every == 0:
                    checkpoint_data = {
                        "episode": episodes_completed,
                        "model_state": weights_snapshot,
                    }
                    launched = eval_manager.launch(
                        args,
                        input_dim,
                        weights_snapshot,
                        checkpoint_data,
                        args.seed,
                        episodes_completed,
                    )
                    if not launched:
                        print(
                            f"eval episode={episodes_completed} skipped "
                            "(previous eval still running)"
                        )

                if episodes_completed >= args.episodes:
                    break
    finally:
        if args.save_every:
            save_checkpoint(args.checkpoint, policy, optimiser, episodes_completed)
        for _process, conn in workers:
            try:
                conn.send({"cmd": "stop"})
            except Exception:
                pass
        for process, conn in workers:
            try:
                process.join(timeout=10)
            finally:
                conn.close()
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        eval_manager.wait()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env, cfg = make_env(args)
    env.reset(seed=args.seed)
    sample_obs = env.observe(env.possible_agents[0])
    sample_map = InternalMap(int(cfg.env.grid_size))
    sample_map.update(sample_obs, cfg)
    input_dim = int(obs_to_vector(sample_obs, cfg, deque(maxlen=ACTION_HISTORY_LEN), sample_map).shape[0])

    policy = DQN(input_dim, args.hidden_dim, len(Action)).to(device)
    target = DQN(input_dim, args.hidden_dim, len(Action)).to(device)
    target.load_state_dict(policy.state_dict())
    optimiser = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    if args.resume and args.checkpoint.exists():
        checkpoint = torch.load(args.checkpoint, map_location=device)
        adjusted_state, adjusted = adapt_state_dict_input_dim(checkpoint["model_state"], input_dim)
        if adjusted:
            print("resume: adjusted first layer input size to match current feature vector")
        policy.load_state_dict(adjusted_state)
        target.load_state_dict(policy.state_dict())
        if "optimiser_state" in checkpoint:
            optimiser.load_state_dict(checkpoint["optimiser_state"])
        else:
            print("resume: no optimiser state found (best checkpoint?), starting optimiser fresh")
        print(f"resumed from {args.checkpoint} at episode {checkpoint.get('episode', '?')}")

    if args.num_workers > 1:
        env.close()
        train_parallel(args, device, input_dim, policy, target, optimiser)
        return

    replay = ReplayBuffer(args.buffer_size)
    viewer = None if args.headless else TrainingViewer(env, fps=args.fps)
    round_index = 0
    latest_loss: float | None = None
    eval_manager = AsyncEvalManager()

    try:
        for episode in range(1, args.episodes + 1):
            seed = None if args.seed is None else args.seed + episode
            env.reset(seed=seed)
            pending: dict[str, PendingTransition] = {}
            action_histories: dict[str, deque[int]] = {
                agent_id: deque(maxlen=ACTION_HISTORY_LEN) for agent_id in env.possible_agents
            }
            internal_maps: dict[str, InternalMap] = {
                agent_id: InternalMap(int(cfg.env.grid_size)) for agent_id in env.possible_agents
            }
            episode_reward = 0.0
            running = True

            for agent_id in env.agent_iter():
                obs, _reward, termination, truncation, _info = env.last()
                was_last = env.agent_selector.is_last()

                if termination or truncation:
                    action = None
                else:
                    internal_maps[agent_id].update(obs, cfg)
                    state = obs_to_vector(obs, cfg, action_histories.get(agent_id), internal_maps[agent_id])
                    epsilon = epsilon_for_round(args, round_index)
                    action = select_action(policy, state, obs["action_mask"], epsilon, device)
                    pending[agent_id] = PendingTransition(state=state, action=action)
                    action_histories[agent_id].append(int(action))

                env.step(action)

                if was_last:
                    finish_round_transitions(
                        env,
                        cfg,
                        replay,
                        pending,
                        action_histories,
                        internal_maps,
                        args.forward_backward_penalty,
                        args.left_right_penalty,
                        args.same_turn_penalty,
                    )
                    episode_reward += sum(float(v) for v in env.rewards.values())
                    round_index += 1

                    for _ in range(args.updates_per_round):
                        loss = optimise(
                            policy,
                            target,
                            optimiser,
                            replay,
                            args.batch_size,
                            args.gamma,
                            device,
                        )
                        if loss is not None:
                            latest_loss = loss

                    if round_index % args.target_sync_rounds == 0:
                        target.load_state_dict(policy.state_dict())

                    if viewer is not None:
                        running = viewer.draw(episode, epsilon, latest_loss, len(replay))
                    if not running:
                        break

                if all(env.truncations.values()) or all(env.terminations.values()):
                    break

            print(
                f"episode={episode} env_reward={episode_reward:.2f} "
                f"env_score={episode_reward / 1000.0:.4f} "
                f"epsilon={epsilon_for_round(args, round_index):.3f} "
                f"replay={len(replay)} loss={latest_loss if latest_loss is not None else 'warmup'}"
            )

            if args.save_every and episode % args.save_every == 0:
                save_checkpoint(args.checkpoint, policy, optimiser, episode)

            if args.eval_every and episode % args.eval_every == 0:
                weights_snapshot = cpu_state_dict(policy)
                checkpoint_data = {
                    "episode": episode,
                    "model_state": weights_snapshot,
                }
                launched = eval_manager.launch(
                    args,
                    input_dim,
                    weights_snapshot,
                    checkpoint_data,
                    args.seed,
                    episode,
                )
                if not launched:
                    print(
                        f"eval episode={episode} skipped (previous eval still running)"
                    )

            if not running:
                break

    finally:
        if args.save_every:
            save_checkpoint(args.checkpoint, policy, optimiser, episode if "episode" in locals() else 0)
        env.close()
        if viewer is not None:
            viewer.close()
        eval_manager.wait()


if __name__ == "__main__":
    main()
