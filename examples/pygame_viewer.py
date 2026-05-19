from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib

import numpy as np

from hunter_gatherers import HunterGathererPatchEnv, PatchEnvConfig
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
    Terrain.WOODLAND: (80, 132, 76),
    Terrain.DENSE_FOREST: (37, 92, 67),
    Terrain.WATER: (58, 126, 170),
    Terrain.HILL: (133, 126, 106),
    Terrain.MARSH: (91, 123, 101),
}

TERRAIN_LABELS = {
    Terrain.GRASSLAND: "Grassland",
    Terrain.WOODLAND: "Woodland",
    Terrain.DENSE_FOREST: "Dense forest",
    Terrain.WATER: "Water",
    Terrain.HILL: "Hill",
    Terrain.MARSH: "Marsh",
}

OVERLAY_COLORS = {
    "plants": (92, 217, 82),
    "animals": (229, 178, 73),
    "danger": (220, 68, 59),
    "depletion": (76, 63, 50),
}

OVERLAY_KEYS = {
    "plants": "1",
    "animals": "2",
    "danger": "3",
    "depletion": "4",
}

BAND_COLORS = {
    0: (246, 198, 75),
    1: (91, 190, 230),
}

CONTROLLED_MEMBER_COLOR = (255, 238, 88)
CAMP_COLOR = (245, 236, 198)
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
    viewer_config = ViewerConfig(
        cell_size=int(raw_config.get("cell_size", ViewerConfig.cell_size)),
        sidebar_width=int(
            raw_config.get("sidebar_width", ViewerConfig.sidebar_width)
        ),
        scale=float(raw_config.get("scale", ViewerConfig.scale)),
        fps=int(raw_config.get("fps", ViewerConfig.fps)),
        seed=int(raw_config.get("seed", ViewerConfig.seed)),
        auto_step_delay_ms=int(
            raw_config.get(
                "auto_step_delay_ms",
                ViewerConfig.auto_step_delay_ms,
            )
        ),
    )
    return viewer_config


class PygamePatchViewer:
    def __init__(self, env: HunterGathererPatchEnv, config: ViewerConfig):
        self.env = env
        self.config = config
        self.overlay = "plants"
        self.autoplay = False
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
                    self._step(self.env.action_space.sample())
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
        if event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
            self.overlay = {
                pygame.K_1: "plants",
                pygame.K_2: "animals",
                pygame.K_3: "danger",
                pygame.K_4: "depletion",
            }[event.key]
            return

        if self.terminated or self.truncated:
            return

        shifted = bool(event.mod & pygame.KMOD_SHIFT)
        if event.key == pygame.K_UP:
            action = Action.MOVE_CAMP_NORTH if shifted else Action.MOVE_NORTH
            self._step(action)
        elif event.key == pygame.K_DOWN:
            action = Action.MOVE_CAMP_SOUTH if shifted else Action.MOVE_SOUTH
            self._step(action)
        elif event.key == pygame.K_LEFT:
            self._step(Action.MOVE_CAMP_WEST if shifted else Action.MOVE_WEST)
        elif event.key == pygame.K_RIGHT:
            self._step(Action.MOVE_CAMP_EAST if shifted else Action.MOVE_EAST)
        elif event.key == pygame.K_SPACE:
            self._step(Action.STAY)
        elif event.key == pygame.K_g:
            self._step(Action.GATHER)
        elif event.key == pygame.K_h:
            self._step(Action.HUNT)
        elif event.key == pygame.K_r:
            self._step(Action.REST)

    def _reset(self, seed: int) -> None:
        self.last_reward = 0.0
        self.terminated = False
        self.truncated = False
        _, self.info = self.env.reset(seed=seed)

    def _step(self, action: Action | int) -> None:
        (
            _,
            reward,
            self.terminated,
            self.truncated,
            self.info,
        ) = self.env.step(int(action))
        self.last_reward = reward

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
                color = TERRAIN_COLORS[terrain]
                pygame.draw.rect(self.screen, color, rect)
                display_color = color

                value = self._overlay_value(y, x)
                if value > 0.01:
                    overlay = pygame.Surface((cell, cell), pygame.SRCALPHA)
                    alpha = int(np.clip(value, 0.0, 1.0) * 145)
                    overlay.fill((*OVERLAY_COLORS[self.overlay], alpha))
                    self.screen.blit(overlay, rect)
                    display_color = self._blend_color(
                        color,
                        OVERLAY_COLORS[self.overlay],
                        alpha / 255.0,
                    )

                pygame.draw.rect(
                    self.screen,
                    (29, 35, 29),
                    rect,
                    self._scale(1),
                )
                self._draw_cell_number(rect, terrain, display_color)

        self._draw_camps()
        self._draw_members()

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
        self._draw_legend(x, y)

    def _draw_legend(self, x: int, y: int) -> None:
        y = self._draw_section_title("Legend", x, y)
        y = self._draw_symbol_entry(
            "Controlled member",
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
        y = self._draw_symbol_entry(
            "Band 1 member",
            x,
            y,
            "circle",
            BAND_COLORS[1],
        )
        y = self._draw_symbol_entry("Camp", x, y, "triangle", CAMP_COLOR)
        y += self._scale(5)

        y = self._draw_section_title("Terrain", x, y)
        for terrain, label in TERRAIN_LABELS.items():
            y = self._draw_swatch_entry(
                label,
                TERRAIN_COLORS[terrain],
                x,
                y,
                str(int(terrain)),
            )
        y += self._scale(5)

        y = self._draw_section_title("Overlays", x, y)
        for key, color in OVERLAY_COLORS.items():
            prefix = "*" if key == self.overlay else " "
            y = self._draw_swatch_entry(
                f"{prefix} {key}",
                color,
                x,
                y,
                OVERLAY_KEYS[key],
            )

        y += self._scale(5)
        self._draw_text(
            "Shift+arrows: move camp",
            x,
            y,
            self.small_font,
            (174, 178, 166),
            self._scale(22),
        )

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
        marker: str,
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
        marker_surface = self.badge_font.render(
            marker,
            True,
            self._contrast_color(color),
        )
        marker_rect = marker_surface.get_rect(center=rect.center)
        self.screen.blit(marker_surface, marker_rect)
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

    def _draw_cell_number(
        self,
        rect: pygame.Rect,
        terrain: Terrain,
        display_color: tuple[int, int, int],
    ) -> None:
        text_color = self._contrast_color(display_color)
        shadow_color = self._contrast_color(text_color)
        marker = self.grid_font.render(str(int(terrain)), True, shadow_color)
        offset = self._scale(1)
        shadow_center = (rect.centerx + offset, rect.centery + offset)
        marker_rect = marker.get_rect(center=shadow_center)
        self.screen.blit(marker, marker_rect)

        marker = self.grid_font.render(str(int(terrain)), True, text_color)
        marker_rect = marker.get_rect(center=rect.center)
        self.screen.blit(marker, marker_rect)

    @staticmethod
    def _contrast_color(
        color: tuple[int, int, int],
    ) -> tuple[int, int, int]:
        red, green, blue = color
        luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        return (24, 24, 24) if luminance > 145 else (245, 245, 238)

    @staticmethod
    def _blend_color(
        base: tuple[int, int, int],
        overlay: tuple[int, int, int],
        alpha: float,
    ) -> tuple[int, int, int]:
        return tuple(
            int((1.0 - alpha) * base_channel + alpha * overlay_channel)
            for base_channel, overlay_channel in zip(base, overlay)
        )  # type: ignore[return-value]

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

    def _overlay_value(self, y: int, x: int) -> float:
        if self.overlay == "plants":
            return float(self.env.plant_food[y, x])
        if self.overlay == "animals":
            return float(self.env.animal_density[y, x])
        if self.overlay == "danger":
            return float(self.env.danger[y, x])
        if self.overlay == "depletion":
            return float(self.env.depletion[y, x])
        return 0.0


def main() -> None:
    config = load_viewer_config()
    env = HunterGathererPatchEnv(PatchEnvConfig())
    viewer = PygamePatchViewer(env, config)
    viewer.run()


if __name__ == "__main__":
    main()
