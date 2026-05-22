from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import tomllib

import numpy as np

from hunter_gatherers import (
    BandMemberPatchEnv,
    HunterGathererPatchEnv,
    PatchEnvConfig,
)
from hunter_gatherers.envs.patch_env import Action, Terrain


try:
    import pygame
except ModuleNotFoundError as exc:
    raise SystemExit(
        "pygame is required for this viewer. "
        "Install it with: pip install pygame"
    ) from exc


TERRAIN_COLORS = {
    Terrain.GRASSLAND: (119, 158, 82),
    Terrain.WATER: (58, 126, 170),
}

TERRAIN_LABELS = {
    Terrain.GRASSLAND: "Grassland",
    Terrain.WATER: "Water",
}

PLANT_COLOR = (55, 174, 65)
PLANT_CELL_MIN_ENERGY = 0.01

RANDOM_POLICY_ACTIONS = (
    Action.STAY,
    Action.MOVE_NORTH,
    Action.MOVE_SOUTH,
    Action.MOVE_WEST,
    Action.MOVE_EAST,
)

BAND_COLORS = {
    0: (246, 198, 75),
    1: (91, 190, 230),
}

CONTROLLED_MEMBER_COLOR = (255, 238, 88)
CAMP_COLOR = (245, 236, 198)
CAMP_RADIUS_FILL = (245, 236, 198, 34)
CAMP_RADIUS_OUTLINE = (255, 247, 209)
CONFIG_PATH = Path(__file__).with_name("pygame_viewer_config.toml")


@dataclass(frozen=True)
class ViewerConfig:
    cell_size: int = 18
    sidebar_width: int = 280
    scale: float = 1.8
    fps: int = 12
    seed: int = 123
    auto_step_delay_ms: int = 180


def load_viewer_config(path: Path = CONFIG_PATH) -> ViewerConfig:
    raw_config = tomllib.loads(path.read_text(encoding="utf-8"))
    viewer_values = raw_config.get("viewer", raw_config)
    viewer_config = ViewerConfig(
        cell_size=int(
            viewer_values.get("cell_size", ViewerConfig.cell_size)
        ),
        sidebar_width=int(
            viewer_values.get("sidebar_width", ViewerConfig.sidebar_width)
        ),
        scale=float(viewer_values.get("scale", ViewerConfig.scale)),
        fps=int(viewer_values.get("fps", ViewerConfig.fps)),
        seed=int(viewer_values.get("seed", ViewerConfig.seed)),
        auto_step_delay_ms=int(
            viewer_values.get(
                "auto_step_delay_ms",
                ViewerConfig.auto_step_delay_ms,
            )
        ),
    )
    return viewer_config


def load_env_config(path: Path = CONFIG_PATH) -> PatchEnvConfig:
    raw_config = tomllib.loads(path.read_text(encoding="utf-8"))
    if "environment" not in raw_config:
        return PatchEnvConfig()
    return PatchEnvConfig.from_toml(path)


class PygamePatchViewer:
    def __init__(
        self,
        env: HunterGathererPatchEnv,
        config: ViewerConfig,
        *,
        autoplay: bool = False,
        random_policy_all_members: bool = False,
    ):
        self.env = env
        self.config = config
        self.autoplay = autoplay
        self.random_policy_all_members = random_policy_all_members
        self.random_policy_rng = np.random.default_rng(config.seed)
        self.multi_agent_env = (
            env if isinstance(env, BandMemberPatchEnv) else None
        )
        if self.random_policy_all_members and self.multi_agent_env is None:
            raise ValueError(
                "random_policy_all_members requires BandMemberPatchEnv."
            )
        self.last_auto_step_ms = 0
        self.last_reward = 0.0
        self.terminated = False
        self.truncated = False
        self.info = {}
        self.scale = config.scale
        if self.scale <= 0:
            raise ValueError("Viewer scale must be greater than 0.")
        self.cell_size = self._scale(config.cell_size)
        self.sidebar_width = self._scale(config.sidebar_width)

        width = env.grid_size * self.cell_size + self.sidebar_width
        height = env.grid_size * self.cell_size
        pygame.init()
        pygame.display.set_caption("HunterGathererPatchEnv")
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", self._scale(18))
        self.small_font = pygame.font.SysFont("arial", self._scale(17))
        self.energy_font = pygame.font.SysFont("arial", self._scale(13))
        self.badge_font = pygame.font.SysFont(
            "arial",
            self._scale(13),
            bold=True,
        )
        self.grid_font = pygame.font.SysFont(
            "arial",
            self._scale(11),
            bold=True,
        )
        self.large_font = pygame.font.SysFont(
            "arial",
            self._scale(22),
            bold=True,
        )

        _, self.info = self.env.reset(seed=config.seed)

    def _scale(self, value: int | float) -> int:
        return max(1, int(round(value * self.scale)))

    def run(self, max_frames: int | None = None) -> None:
        running = True
        frames = 0
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    else:
                        self._handle_key(event)

            if self.autoplay and not self.terminated and not self.truncated:
                now = pygame.time.get_ticks()
                elapsed = now - self.last_auto_step_ms
                if elapsed >= self.config.auto_step_delay_ms:
                    self._step_random_policy()
                    self.last_auto_step_ms = now

            self._draw()
            pygame.display.flip()
            self.clock.tick(self.config.fps)

            frames += 1
            if max_frames is not None and frames >= max_frames:
                running = False

        pygame.quit()

    def _handle_key(self, event: pygame.event.Event) -> None:
        if event.key == pygame.K_a:
            self.autoplay = not self.autoplay
            return
        if event.key == pygame.K_n:
            seed = (
                self.config.seed
                + self.env.camp_id
                + self.env.step_count
                + 1
            )
            self._reset(seed)
            return
        if self.terminated or self.truncated:
            return

        if event.key == pygame.K_r:
            self._trigger_recamping_test()
            return

        if event.key == pygame.K_UP:
            self._step(Action.MOVE_NORTH)
        elif event.key == pygame.K_DOWN:
            self._step(Action.MOVE_SOUTH)
        elif event.key == pygame.K_LEFT:
            self._step(Action.MOVE_WEST)
        elif event.key == pygame.K_RIGHT:
            self._step(Action.MOVE_EAST)
        elif event.key == pygame.K_SPACE:
            self._step(Action.STAY)

    def _reset(self, seed: int) -> None:
        self.last_reward = 0.0
        self.terminated = False
        self.truncated = False
        _, self.info = self.env.reset(seed=seed)

    def _step(self, action: Action | int) -> None:
        if self.random_policy_all_members:
            self._step_multi_agent({self._controlled_agent_id(): int(action)})
            return

        (
            _,
            reward,
            self.terminated,
            self.truncated,
            self.info,
        ) = self.env.step(int(action))
        self.last_reward = reward

    def _step_random_policy(self) -> None:
        if not self.random_policy_all_members:
            self._step(self._random_policy_action())
            return

        multi_env = self._multi_agent_env()
        action_dict = {
            agent_id: int(self._random_policy_action())
            for agent_id in multi_env.agents
        }
        self._step_multi_agent(action_dict)

    def _step_multi_agent(self, action_dict: dict[str, int]) -> None:
        multi_env = self._multi_agent_env()
        (
            _,
            rewards,
            terminations,
            truncations,
            infos,
        ) = multi_env.step(action_dict)
        self.last_reward = float(sum(rewards.values()))
        self.terminated = bool(terminations["__all__"])
        self.truncated = bool(truncations["__all__"])
        self.info = {
            "agent_infos": infos,
            "step_count": multi_env.step_count,
            "population": multi_env.population,
        }

    def _trigger_recamping_test(self) -> None:
        self.autoplay = False
        self.last_auto_step_ms = pygame.time.get_ticks()
        camp_radius = self.env._camp_radius_mask(self.env.camp_pos)
        resource_cells = camp_radius & (self.env.plant_capacity > 0.0)
        self.env.plant_energy[resource_cells] = 0.0
        self.env._update_depletion()
        self.env._maybe_relocate_depleted_camp()

    @staticmethod
    def _controlled_agent_id() -> str:
        return "band_0_member_0"

    def _random_policy_action(self) -> Action:
        return self.random_policy_rng.choice(RANDOM_POLICY_ACTIONS)

    def _multi_agent_env(self) -> BandMemberPatchEnv:
        if self.multi_agent_env is None:
            raise RuntimeError("Multi-agent viewer mode is not enabled.")
        return self.multi_agent_env

    def _draw(self) -> None:
        self.screen.fill((24, 27, 24))
        self._draw_grid()
        self._draw_sidebar()

    def _draw_grid(self) -> None:
        cell = self.cell_size
        for y in range(self.env.grid_size):
            for x in range(self.env.grid_size):
                rect = pygame.Rect(x * cell, y * cell, cell, cell)
                terrain = Terrain(int(self.env.terrain[y, x]))
                plant_energy = float(self.env.plant_energy[y, x])
                color = self._cell_color(terrain, plant_energy)
                pygame.draw.rect(self.screen, color, rect)

                pygame.draw.rect(
                    self.screen,
                    (29, 35, 29),
                    rect,
                    self._scale(1),
                )

        self._draw_camp_radii()
        self._draw_camps()
        self._draw_members()

    def _draw_camp_radii(self) -> None:
        radius = int(self.env.config.camp_depletion_radius)
        if radius <= 0:
            return

        cell = self.cell_size
        thickness = self._scale(2)
        overlay = pygame.Surface(
            (
                self.env.grid_size * cell,
                self.env.grid_size * cell,
            ),
            pygame.SRCALPHA,
        )

        for camp_y, camp_x in self.env.band_camp_positions:
            cy = int(camp_y)
            cx = int(camp_x)
            y_min = max(0, cy - radius)
            y_max = min(self.env.grid_size - 1, cy + radius)
            x_min = max(0, cx - radius)
            x_max = min(self.env.grid_size - 1, cx + radius)

            for y in range(y_min, y_max + 1):
                for x in range(x_min, x_max + 1):
                    if abs(y - cy) + abs(x - cx) > radius:
                        continue

                    rect = pygame.Rect(x * cell, y * cell, cell, cell)
                    pygame.draw.rect(overlay, CAMP_RADIUS_FILL, rect)
                    self._draw_camp_radius_edges(
                        overlay,
                        rect,
                        y,
                        x,
                        cy,
                        cx,
                        radius,
                        thickness,
                    )

        self.screen.blit(overlay, (0, 0))

    def _draw_camp_radius_edges(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        y: int,
        x: int,
        camp_y: int,
        camp_x: int,
        radius: int,
        thickness: int,
    ) -> None:
        edge_color = CAMP_RADIUS_OUTLINE
        neighbors = (
            (-1, 0, rect.topleft, rect.topright),
            (1, 0, rect.bottomleft, rect.bottomright),
            (0, -1, rect.topleft, rect.bottomleft),
            (0, 1, rect.topright, rect.bottomright),
        )
        for dy, dx, start, end in neighbors:
            ny = y + dy
            nx = x + dx
            outside_grid = (
                ny < 0
                or ny >= self.env.grid_size
                or nx < 0
                or nx >= self.env.grid_size
            )
            outside_radius = (
                abs(ny - camp_y) + abs(nx - camp_x) > radius
            )
            if outside_grid or outside_radius:
                pygame.draw.line(surface, edge_color, start, end, thickness)

    def _draw_camps(self) -> None:
        cell = self.cell_size
        for y, x in self.env.band_camp_positions:
            cx = int(x) * cell + cell // 2
            cy = int(y) * cell + cell // 2
            points = [
                (cx, cy - cell * 0.38),
                (cx - cell * 0.42, cy + cell * 0.35),
                (cx + cell * 0.42, cy + cell * 0.35),
            ]
            pygame.draw.polygon(self.screen, CAMP_COLOR, points)
            pygame.draw.polygon(
                self.screen,
                (56, 48, 39),
                points,
                self._scale(2),
            )

    def _draw_members(self) -> None:
        cell = self.cell_size
        for band_id, band_positions in enumerate(self.env.member_positions):
            color = BAND_COLORS.get(int(band_id), (220, 220, 220))
            for member_id, (y, x) in enumerate(band_positions):
                if not self.env.member_alive[band_id, member_id]:
                    continue
                center = (
                    int(x) * cell + cell // 2,
                    int(y) * cell + cell // 2,
                )
                controlled = band_id == 0 and member_id == 0
                radius = max(4, int(cell * (0.36 if controlled else 0.28)))
                fill = CONTROLLED_MEMBER_COLOR if controlled else color
                pygame.draw.circle(
                    self.screen,
                    (32, 33, 35),
                    center,
                    radius + self._scale(2),
                )
                pygame.draw.circle(self.screen, fill, center, radius)

    def _draw_sidebar(self) -> None:
        left = self.env.grid_size * self.cell_size
        panel = pygame.Rect(
            left,
            0,
            self.sidebar_width,
            self.screen.get_height(),
        )
        pygame.draw.rect(self.screen, (31, 34, 32), panel)
        pygame.draw.line(
            self.screen,
            (64, 70, 64),
            (left, 0),
            (left, panel.height),
            self._scale(2),
        )

        x = left + self._scale(18)
        y = self._scale(18)
        y = self._draw_camp_metrics(x, y)
        y += self._scale(16)
        y = self._draw_legend(x, y)
        y += self._scale(16)
        available_height = self.screen.get_height() - y - self._scale(12)
        chart_height = max(self._scale(180), available_height)
        self._draw_member_resource_chart(
            x,
            y,
            chart_height,
        )

    def _draw_legend(self, x: int, y: int) -> int:
        y = self._draw_section_title("Legend", x, y)
        y = self._draw_symbol_entry(
            "Leader",
            x,
            y,
            "circle",
            CONTROLLED_MEMBER_COLOR,
        )
        y = self._draw_symbol_entry(
            "Band 0 member",
            x,
            y,
            "circle",
            BAND_COLORS[0],
        )
        y = self._draw_symbol_entry("Camp", x, y, "triangle", CAMP_COLOR)
        y += self._scale(5)

        y = self._draw_section_title("Cells", x, y)
        y = self._draw_swatch_entry("Plants", PLANT_COLOR, x, y)
        for terrain, label in TERRAIN_LABELS.items():
            y = self._draw_swatch_entry(
                label,
                TERRAIN_COLORS[terrain],
                x,
                y,
            )
        return y

    def _draw_camp_metrics(self, x: int, y: int) -> int:
        y = self._draw_section_title("Camp", x, y)
        y = self._draw_metric_row(
            "Potential energy",
            f"{self.env.camp_potential_energy():.1f}",
            x,
            y,
        )
        y = self._draw_metric_row(
            "Camp food",
            f"{float(self.env.camp_stored_food[0]):.1f}",
            x,
            y,
        )
        y = self._draw_metric_row(
            "Camp water",
            f"{float(self.env.camp_stored_water[0]):.1f}",
            x,
            y,
        )
        y = self._draw_metric_row(
            "Carried food",
            f"{float(np.sum(self.env.member_carried_food[0])):.1f}",
            x,
            y,
        )
        y = self._draw_metric_row(
            "Carried water",
            f"{float(np.sum(self.env.member_carried_water[0])):.1f}",
            x,
            y,
        )
        y = self._draw_metric_row(
            "Depletion",
            f"{self.env.local_depletion_level:.0%}",
            x,
            y,
        )
        y = self._draw_metric_row(
            "Camp moves",
            str(self.env.num_camp_moves),
            x,
            y,
        )
        return self._draw_metric_row(
            "Relocated",
            "yes" if self.env._last_camp_relocated else "no",
            x,
            y,
        )

    def _draw_metric_row(
        self,
        label: str,
        value: str,
        x: int,
        y: int,
    ) -> int:
        label_surface = self.small_font.render(label, True, (211, 214, 201))
        value_surface = self.small_font.render(value, True, (238, 237, 225))
        self.screen.blit(label_surface, (x, y))
        value_rect = value_surface.get_rect(
            topright=(self.screen.get_width() - self._scale(18), y)
        )
        self.screen.blit(value_surface, value_rect)
        return y + self._scale(22)

    def _draw_member_resource_chart(
        self,
        x: int,
        y: int,
        chart_height: int,
    ) -> int:
        chart_width = self.sidebar_width - self._scale(36)
        band_id = 0
        member_count = self.env.config.members_per_band
        header_height = self._scale(20)
        row_height = max(
            self._scale(12),
            (chart_height - header_height - self._scale(6)) // member_count,
        )
        chart_height = (
            header_height
            + row_height * member_count
            + self._scale(6)
        )
        chart_rect = pygame.Rect(x, y, chart_width, chart_height)
        id_width = self._scale(36)
        gap = self._scale(8)
        metric_width = max(
            self._scale(40),
            (chart_rect.width - id_width - gap) // 2,
        )
        energy_x = chart_rect.x + id_width
        water_x = energy_x + metric_width + gap
        energy_max = max(1.0, float(self.env.config.max_energy))
        water_max = max(1.0, float(self.env.config.max_hydration))
        bar_height = max(self._scale(6), row_height - self._scale(8))

        pygame.draw.rect(self.screen, (26, 29, 27), chart_rect)
        pygame.draw.rect(
            self.screen,
            (66, 72, 66),
            chart_rect,
            self._scale(1),
        )
        self._draw_chart_header("Energy", energy_x, y, metric_width)
        self._draw_chart_header("Water", water_x, y, metric_width)

        for member_id in range(member_count):
            row_y = (
                chart_rect.y
                + header_height
                + self._scale(3)
                + member_id * row_height
            )
            alive = bool(self.env.member_alive[band_id, member_id])
            member_identity = int(self.env.member_ids[band_id, member_id])
            id_surface = self.energy_font.render(
                f"#{member_identity}",
                True,
                (211, 214, 201),
            )
            id_rect = id_surface.get_rect(
                midleft=(
                    chart_rect.x + self._scale(6),
                    row_y + row_height // 2,
                )
            )
            self.screen.blit(id_surface, id_rect)

            energy_color = (
                CONTROLLED_MEMBER_COLOR
                if member_id == 0
                else BAND_COLORS[0]
            )
            water_color = TERRAIN_COLORS[Terrain.WATER]
            if not alive:
                energy_color = (85, 88, 82)
                water_color = (85, 88, 82)

            self._draw_member_resource_bar(
                energy_x,
                row_y,
                metric_width,
                bar_height,
                row_height,
                max(0.0, float(self.env.member_energy[band_id, member_id])),
                energy_max,
                energy_color,
            )
            self._draw_member_resource_bar(
                water_x,
                row_y,
                metric_width,
                bar_height,
                row_height,
                max(
                    0.0,
                    float(self.env.member_hydration[band_id, member_id]),
                ),
                water_max,
                water_color,
            )

        pygame.draw.rect(
            self.screen,
            (31, 34, 32),
            chart_rect,
            self._scale(1),
        )
        return chart_rect.bottom

    def _draw_chart_header(
        self,
        label: str,
        x: int,
        y: int,
        width: int,
    ) -> None:
        surface = self.energy_font.render(label, True, (211, 214, 201))
        rect = surface.get_rect(
            midtop=(x + width // 2, y + self._scale(3))
        )
        self.screen.blit(surface, rect)

    def _draw_member_resource_bar(
        self,
        x: int,
        row_y: int,
        width: int,
        bar_height: int,
        row_height: int,
        value: float,
        max_value: float,
        fill_color: tuple[int, int, int],
    ) -> None:
        value_ratio = min(1.0, value / max_value)
        fill_width = int(round(width * value_ratio))
        bar_bg_rect = pygame.Rect(
            x,
            row_y + (row_height - bar_height) // 2,
            width,
            bar_height,
        )
        bar_rect = pygame.Rect(
            x,
            bar_bg_rect.y,
            fill_width,
            bar_height,
        )
        pygame.draw.rect(self.screen, (45, 50, 45), bar_bg_rect)
        pygame.draw.rect(self.screen, fill_color, bar_rect)
        pygame.draw.rect(
            self.screen,
            (31, 34, 32),
            bar_bg_rect,
            self._scale(1),
        )

        value_surface = self.energy_font.render(
            f"{value:.0f}",
            True,
            (238, 237, 225),
        )
        value_rect = value_surface.get_rect(
            midright=(
                bar_bg_rect.right - self._scale(4),
                bar_bg_rect.centery,
            )
        )
        self.screen.blit(value_surface, value_rect)

    def _draw_section_title(self, text: str, x: int, y: int) -> int:
        return self._draw_text(
            text,
            x,
            y,
            self.font,
            (238, 237, 225),
            self._scale(25),
        )

    def _draw_swatch_entry(
        self,
        label: str,
        color: tuple[int, int, int],
        x: int,
        y: int,
    ) -> int:
        rect = pygame.Rect(
            x,
            y + self._scale(2),
            self._scale(20),
            self._scale(20),
        )
        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(
            self.screen,
            (88, 93, 87),
            rect,
            self._scale(1),
        )
        return self._draw_text(
            label,
            x + self._scale(28),
            y,
            self.small_font,
            (211, 214, 201),
            self._scale(22),
        )

    def _draw_symbol_entry(
        self,
        label: str,
        x: int,
        y: int,
        symbol: str,
        color: tuple[int, int, int],
    ) -> int:
        center = (x + self._scale(8), y + self._scale(12))
        if symbol == "circle":
            pygame.draw.circle(self.screen, color, center, self._scale(7))
            pygame.draw.circle(
                self.screen,
                (35, 35, 38),
                center,
                self._scale(8),
                self._scale(1),
            )
        else:
            points = [
                (x + self._scale(8), y + self._scale(3)),
                (x, y + self._scale(17)),
                (x + self._scale(16), y + self._scale(17)),
            ]
            pygame.draw.polygon(self.screen, color, points)
            pygame.draw.polygon(
                self.screen,
                (56, 48, 39),
                points,
                self._scale(1),
            )
        return self._draw_text(
            label,
            x + self._scale(23),
            y,
            self.small_font,
            (211, 214, 201),
            self._scale(22),
        )

    def _cell_color(
        self,
        terrain: Terrain,
        plant_energy: float,
    ) -> tuple[int, int, int]:
        if self._is_plant_cell(terrain, plant_energy):
            return PLANT_COLOR
        return TERRAIN_COLORS[terrain]

    @staticmethod
    def _is_plant_cell(terrain: Terrain, plant_energy: float) -> bool:
        return (
            terrain != Terrain.WATER
            and plant_energy >= PLANT_CELL_MIN_ENERGY
        )

    @staticmethod
    def _contrast_color(
        color: tuple[int, int, int],
    ) -> tuple[int, int, int]:
        red, green, blue = color
        luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        return (24, 24, 24) if luminance > 145 else (245, 245, 238)

    def _draw_text(
        self,
        text: str,
        x: int,
        y: int,
        font: pygame.font.Font,
        color: tuple[int, int, int] = (211, 214, 201),
        line_height: int | None = None,
    ) -> int:
        surface = font.render(text, True, color)
        self.screen.blit(surface, (x, y))
        if line_height is None:
            line_height = self._scale(22)
        return y + line_height

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--multi-agent-random",
        action="store_true",
        help="Control all band 0 members with random actions during autoplay.",
    )
    parser.add_argument(
        "--autoplay",
        action="store_true",
        help="Start stepping immediately instead of waiting for the 'a' key.",
    )
    args = parser.parse_args()

    config = load_viewer_config()
    env_config = load_env_config()
    env = (
        BandMemberPatchEnv(env_config)
        if args.multi_agent_random
        else HunterGathererPatchEnv(env_config)
    )
    viewer = PygamePatchViewer(
        env,
        config,
        autoplay=args.autoplay,
        random_policy_all_members=args.multi_agent_random,
    )
    viewer.run()


if __name__ == "__main__":
    main()
