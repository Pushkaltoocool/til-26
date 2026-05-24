"""Manages the AE model."""

from __future__ import annotations

import os
import heapq
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


NUM_ACTIONS = 6
ACTION_HISTORY_LEN = 3

ACTION_FORWARD = 0
ACTION_BACKWARD = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_STAY = 4
ACTION_PLACE_BOMB = 5
NAV_TARGET_COUNT = 3
NAV_FEATURE_DIM = NAV_TARGET_COUNT * (NUM_ACTIONS + 1)

GRID_SIZE = 16
AGENT_MAX_HEALTH = 60.0
AGENT_FREEZE_TURNS = 3.0
BASE_MAX_HEALTH = 100.0
MAX_TEAM_RESOURCES = 100.0
MAX_TEAM_BOMBS = 50.0
NUM_ITERS = 200.0
UNKNOWN_WALL = -1
NO_WALL = 0
HAS_WALL = 1

DIR_RIGHT = 0
DIR_DOWN = 1
DIR_LEFT = 2
DIR_UP = 3
DIRECTION_VECTORS = {
    DIR_RIGHT: (1, 0),
    DIR_DOWN: (0, 1),
    DIR_LEFT: (-1, 0),
    DIR_UP: (0, -1),
}
DIRECTION_FROM_DELTA = {delta: direction for direction, delta in DIRECTION_VECTORS.items()}

CH_VISIBLE = 0
CH_WALL_RIGHT = 1
CH_WALL_DOWN = 2
CH_WALL_LEFT = 3
CH_WALL_UP = 4
CH_ENEMY_AGENT = 10
CH_ENEMY_BASE = 12

AGENT_VIEWCONE = (2, 2, 2, 4)


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


def _as_array(value: Any, dtype: Any = np.float32) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def _as_scalar(value: Any, default: float = 0.0) -> float:
    arr = _as_array(value).reshape(-1)
    if arr.size == 0:
        return default
    return float(arr[0])


def _encode_action_history(action_history: deque[int]) -> np.ndarray:
    encoded = np.zeros((ACTION_HISTORY_LEN, NUM_ACTIONS), dtype=np.float32)
    history = list(action_history)[-ACTION_HISTORY_LEN:]
    offset = ACTION_HISTORY_LEN - len(history)
    for idx, action in enumerate(history):
        if 0 <= int(action) < NUM_ACTIONS:
            encoded[offset + idx, int(action)] = 1.0
    return encoded.reshape(-1)


def _as_int_xy(value: Any) -> tuple[int, int]:
    arr = _as_array(value, dtype=np.int64).reshape(-1)
    return int(arr[0]), int(arr[1])


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _idx_to_view(idx: tuple[int, int], viewcone: tuple[int, int, int, int]) -> tuple[int, int]:
    return int(idx[0] - viewcone[0]), int(idx[1] - viewcone[2])


def _view_to_world(
    agent_loc: tuple[int, int],
    agent_dir: int,
    view_coord: tuple[int, int],
) -> tuple[int, int]:
    vx, vy = view_coord
    ax, ay = agent_loc
    if agent_dir == DIR_RIGHT:
        return ax + vx, ay + vy
    if agent_dir == DIR_DOWN:
        return ax - vy, ay + vx
    if agent_dir == DIR_LEFT:
        return ax - vx, ay - vy
    return ax + vy, ay - vx


class InternalMap:
    def __init__(self, grid_size: int = GRID_SIZE) -> None:
        self.grid_size = int(grid_size)
        self.seen = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.walls = np.full((self.grid_size, self.grid_size, 4), UNKNOWN_WALL, dtype=np.int8)
        self.enemy_agents: set[tuple[int, int]] = set()
        self.enemy_bases: set[tuple[int, int]] = set()

    def reset(self) -> None:
        self.seen.fill(False)
        self.walls.fill(UNKNOWN_WALL)
        self.enemy_agents.clear()
        self.enemy_bases.clear()

    def update(self, observation: dict[str, Any]) -> None:
        agent_loc = _as_int_xy(observation.get("location", [0, 0]))
        base_loc = _as_int_xy(observation.get("base_location", [0, 0]))
        direction = int(observation.get("direction", 0))
        self._update_agent_view(_as_array(observation["agent_viewcone"]), agent_loc, direction)
        base_viewcone = _as_array(observation["base_viewcone"])
        radius = max((base_viewcone.shape[0] - 1) // 2, 0)
        self._update_radius_view(base_viewcone, base_loc, radius)

    def navigation_features(self, observation: dict[str, Any]) -> np.ndarray:
        start = _as_int_xy(observation.get("location", [0, 0]))
        base = _as_int_xy(observation.get("base_location", [0, 0]))
        direction = int(observation.get("direction", 0))
        grid_norm = max(float(self.grid_size - 1), 1.0)
        target_groups = [
            [base],
            sorted(self.enemy_agents, key=lambda target: _manhattan(start, target)),
            sorted(self.enemy_bases, key=lambda target: _manhattan(start, target)),
        ]

        features: list[float] = []
        for targets in target_groups:
            action, distance = self._next_action_to_nearest(start, direction, targets)
            action_one_hot = np.zeros(NUM_ACTIONS, dtype=np.float32)
            if action is not None:
                action_one_hot[int(action)] = 1.0
            features.extend(float(value) for value in action_one_hot)
            features.append(distance / grid_norm if distance is not None else 1.0)
        return np.asarray(features, dtype=np.float32)

    def _update_agent_view(self, view: np.ndarray, agent_loc: tuple[int, int], direction: int) -> None:
        for idx in np.ndindex(view.shape[:2]):
            if view[idx][CH_VISIBLE] < 0.5:
                continue
            world = _view_to_world(agent_loc, direction, _idx_to_view(idx, AGENT_VIEWCONE))
            self._update_cell(view[idx], world[0], world[1])

    def _update_radius_view(self, view: np.ndarray, center: tuple[int, int], radius: int) -> None:
        if view.size == 0:
            return
        for x, y in np.ndindex(view.shape[:2]):
            if view[x, y, CH_VISIBLE] < 0.5:
                continue
            self._update_cell(view[x, y], center[0] + x - radius, center[1] + y - radius)

    def _update_cell(self, tile: np.ndarray, x: int, y: int) -> None:
        if not self._in_bounds((x, y)):
            return
        self.seen[x, y] = True
        self._set_wall((x, y), DIR_RIGHT, tile[CH_WALL_RIGHT] > 0.5)
        self._set_wall((x, y), DIR_DOWN, tile[CH_WALL_DOWN] > 0.5)
        self._set_wall((x, y), DIR_LEFT, tile[CH_WALL_LEFT] > 0.5)
        self._set_wall((x, y), DIR_UP, tile[CH_WALL_UP] > 0.5)

        key = (x, y)
        if tile[CH_ENEMY_AGENT] > 0.5:
            self.enemy_agents.add(key)
        else:
            self.enemy_agents.discard(key)
        if tile[CH_ENEMY_BASE] > 0.5:
            self.enemy_bases.add(key)
        else:
            self.enemy_bases.discard(key)

    def _set_wall(self, cell: tuple[int, int], direction: int, has_wall: bool) -> None:
        if not self._in_bounds(cell):
            return
        x, y = cell
        self.walls[x, y, direction] = HAS_WALL if has_wall else NO_WALL
        dx, dy = DIRECTION_VECTORS[direction]
        other = (x + dx, y + dy)
        if self._in_bounds(other):
            self.walls[other[0], other[1], (direction + 2) % 4] = HAS_WALL if has_wall else NO_WALL

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
            return ACTION_STAY, distance
        next_cell = best_path[1]
        step_dir = DIRECTION_FROM_DELTA.get((next_cell[0] - start[0], next_cell[1] - start[1]))
        if step_dir is None:
            return ACTION_STAY, distance
        if step_dir == direction:
            return ACTION_FORWARD, distance
        if step_dir == (direction + 2) % 4:
            return ACTION_BACKWARD, distance
        if step_dir == (direction + 3) % 4:
            return ACTION_LEFT, distance
        return ACTION_RIGHT, distance

    def _a_star(self, start: tuple[int, int], target: tuple[int, int]) -> list[tuple[int, int]] | None:
        if not self._in_bounds(start) or not self._in_bounds(target):
            return None
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
                heapq.heappush(frontier, (new_cost + _manhattan(neighbor, target), order, neighbor))
                came_from[neighbor] = current

        if target not in came_from:
            return None
        path: list[tuple[int, int]] = []
        current: tuple[int, int] | None = target
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path

    def _neighbors(self, cell: tuple[int, int]) -> list[tuple[int, int]]:
        neighbors: list[tuple[int, int]] = []
        x, y = cell
        for direction, (dx, dy) in DIRECTION_VECTORS.items():
            neighbor = (x + dx, y + dy)
            if not self._in_bounds(neighbor):
                continue
            if self.walls[x, y, direction] == HAS_WALL:
                continue
            neighbors.append(neighbor)
        return neighbors

    def _in_bounds(self, cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] < self.grid_size and 0 <= cell[1] < self.grid_size


def _legal_actions(mask: Any) -> np.ndarray:
    mask_arr = _as_array(mask).reshape(-1)
    actions = np.flatnonzero(mask_arr > 0)
    if actions.size == 0:
        return np.array([ACTION_STAY], dtype=np.int64)
    return actions.astype(np.int64)


def _checkpoint_candidates() -> list[Path]:
    explicit = os.getenv("AE_MODEL_PATH")
    if explicit:
        return [Path(explicit)]

    here = Path(__file__).resolve().parent
    cwd = Path.cwd()
    candidates = [
        here / "models" / "dqn_viewer_best.pt",
        here / "models" / "dqn_viewer.pt",
        here.parent / "models" / "dqn_viewer_best.pt",
        here.parent / "models" / "dqn_viewer.pt",
        cwd / "models" / "dqn_viewer_best.pt",
        cwd / "models" / "dqn_viewer.pt",
    ]

    models_dirs = [here / "models", here.parent / "models", cwd / "models"]
    for models_dir in models_dirs:
        if models_dir.exists():
            candidates.extend(
                sorted(models_dir.glob("*.pt"), key=lambda item: item.stat().st_mtime, reverse=True)
            )
    return candidates


class AEManager:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_history: deque[int] = deque(maxlen=ACTION_HISTORY_LEN)
        self.internal_map = InternalMap(GRID_SIZE)
        self.last_step: int | None = None
        self.input_dim = self._observation_dim()
        self.model = self._load_model()

    def reset(self) -> None:
        self.action_history.clear()
        self.internal_map.reset()
        self.last_step = None

    def _observation_dim(self) -> int:
        agent_viewcone_dim = 7 * 5 * 25
        base_viewcone_dim = 7 * 7 * 25
        scalar_dim = 4 + 2 + 2 + 1 + 1 + 1 + 1 + 1 + 1 + NUM_ACTIONS
        action_history_dim = ACTION_HISTORY_LEN * NUM_ACTIONS
        return agent_viewcone_dim + base_viewcone_dim + scalar_dim + action_history_dim + NAV_FEATURE_DIM

    def _load_model(self) -> DQN | None:
        for path in _checkpoint_candidates():
            if not path.exists():
                continue
            try:
                checkpoint = torch.load(path, map_location=self.device)
                state_dict = checkpoint["model_state"]
                first_layer = state_dict.get("net.0.weight")
                if first_layer is None:
                    first_layer = next(iter(state_dict.values()))
                hidden_dim = int(first_layer.shape[0])
                model = DQN(self.input_dim, hidden_dim, NUM_ACTIONS).to(self.device)
                model.load_state_dict(self._adapt_state_dict(state_dict), strict=True)
                model.eval()
                print(f"AEManager loaded model: {path}")
                return model
            except Exception as exc:
                print(f"AEManager skipped checkpoint {path}: {exc}")
        print("AEManager did not find a usable checkpoint; using heuristic fallback.")
        return None

    def _adapt_state_dict(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        weight_key = "net.0.weight"
        if weight_key not in state_dict:
            return state_dict

        first_weight = state_dict[weight_key]
        old_input_dim = int(first_weight.shape[1])
        if old_input_dim == self.input_dim:
            return state_dict

        adjusted = {key: value.clone() for key, value in state_dict.items()}
        if old_input_dim < self.input_dim:
            pad = torch.zeros(
                (first_weight.shape[0], self.input_dim - old_input_dim),
                dtype=first_weight.dtype,
                device=first_weight.device,
            )
            adjusted[weight_key] = torch.cat([first_weight, pad], dim=1)
        else:
            adjusted[weight_key] = first_weight[:, : self.input_dim]
        return adjusted

    def _obs_to_vector(self, observation: dict[str, Any]) -> np.ndarray:
        direction = int(observation.get("direction", 0))
        direction_one_hot = np.zeros(4, dtype=np.float32)
        if 0 <= direction < 4:
            direction_one_hot[direction] = 1.0

        grid_norm = max(float(GRID_SIZE - 1), 1.0)
        scalars = np.array(
            [
                *direction_one_hot,
                *(_as_array(observation.get("location", [0, 0]))[:2] / grid_norm),
                *(_as_array(observation.get("base_location", [0, 0]))[:2] / grid_norm),
                _as_scalar(observation.get("health", [0.0])) / AGENT_MAX_HEALTH,
                _as_scalar(observation.get("frozen_ticks", 0.0)) / AGENT_FREEZE_TURNS,
                _as_scalar(observation.get("base_health", [0.0])) / BASE_MAX_HEALTH,
                _as_scalar(observation.get("team_resources", [0.0])) / MAX_TEAM_RESOURCES,
                _as_scalar(observation.get("team_bombs", 0.0)) / MAX_TEAM_BOMBS,
                _as_scalar(observation.get("step", 0.0)) / NUM_ITERS,
                *_as_array(observation.get("action_mask", [0, 0, 0, 0, 1, 0])).reshape(-1)[:NUM_ACTIONS],
                *_encode_action_history(self.action_history),
                *self.internal_map.navigation_features(observation),
            ],
            dtype=np.float32,
        )

        vector = np.concatenate(
            [
                _as_array(observation["agent_viewcone"]).reshape(-1),
                _as_array(observation["base_viewcone"]).reshape(-1),
                scalars,
            ]
        )
        if vector.shape[0] < self.input_dim:
            vector = np.pad(vector, (0, self.input_dim - vector.shape[0]))
        elif vector.shape[0] > self.input_dim:
            vector = vector[: self.input_dim]
        return vector.astype(np.float32, copy=False)

    def _fallback_action(self, action_mask: Any) -> int:
        legal = set(int(action) for action in _legal_actions(action_mask))
        for action in (ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT, ACTION_BACKWARD, ACTION_STAY):
            if action in legal:
                return action
        return ACTION_STAY

    def _select_action(self, observation: dict[str, Any]) -> int:
        action_mask = observation.get("action_mask", [0, 0, 0, 0, 1, 0])
        legal = _legal_actions(action_mask)
        if self.model is None:
            return self._fallback_action(action_mask)

        state = self._obs_to_vector(observation)
        with torch.no_grad():
            q_values = self.model(torch.from_numpy(state).float().unsqueeze(0).to(self.device))[0]

        mask = torch.zeros_like(q_values, dtype=torch.bool)
        mask[torch.as_tensor(legal, device=self.device)] = True
        q_values = q_values.masked_fill(~mask, -1e9)
        return int(torch.argmax(q_values).item())

    def ae(self, observation: dict[str, int | list[int]]) -> int:
        """Gets the next action for the agent, based on the observation."""
        step = int(_as_scalar(observation.get("step", 0)))
        if step == 0 or (self.last_step is not None and step < self.last_step):
            self.reset()
        self.last_step = step
        self.internal_map.update(observation)

        action = self._select_action(observation)
        action_mask = observation.get("action_mask", [0, 0, 0, 0, 1, 0])
        if int(action) not in set(int(item) for item in _legal_actions(action_mask)):
            action = self._fallback_action(action_mask)

        self.action_history.append(int(action))
        return int(action)
