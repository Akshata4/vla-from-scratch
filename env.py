import numpy as np

GRID_SIZE = 8
CELL_PX = 4
IMG_SIZE = GRID_SIZE * CELL_PX  # 32x32 rendered image

COLORS = {
    "red": (220, 60, 60),
    "green": (60, 180, 90),
    "blue": (60, 100, 220),
    "yellow": (220, 200, 60),
}
BG_COLOR = (30, 30, 35)
AGENT_COLOR = (240, 240, 240)

ACTION_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


class GridWorld:
    """Agent must reach the object matching the color named in the instruction."""

    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng()

    def reset(self):
        colors = list(COLORS.keys())
        n_objects = int(self.rng.integers(2, len(colors) + 1))
        chosen = [str(c) for c in self.rng.choice(colors, size=n_objects, replace=False)]

        idx = self.rng.permutation(GRID_SIZE * GRID_SIZE)
        cells = [(int(i // GRID_SIZE), int(i % GRID_SIZE)) for i in idx]

        self.agent_pos = cells[0]
        self.objects = {color: cells[i + 1] for i, color in enumerate(chosen)}
        self.target_color = str(self.rng.choice(chosen))
        self.instruction = f"go to the {self.target_color}"
        return self.render(), self.instruction

    def target_pos(self):
        return self.objects[self.target_color]

    def expert_action(self):
        """Scripted policy used to generate behavior-cloning labels."""
        ar, ac = self.agent_pos
        tr, tc = self.target_pos()
        dr, dc = tr - ar, tc - ac
        if dr == 0 and dc == 0:
            return None
        if abs(dr) >= abs(dc) and dr != 0:
            return "down" if dr > 0 else "up"
        return "right" if dc > 0 else "left"

    def step(self, action):
        dr, dc = ACTION_DELTA[action]
        r, c = self.agent_pos
        self.agent_pos = (
            max(0, min(GRID_SIZE - 1, r + dr)),
            max(0, min(GRID_SIZE - 1, c + dc)),
        )
        success = self.agent_pos == self.target_pos()
        return self.render(), success

    def render(self):
        img = np.full((IMG_SIZE, IMG_SIZE, 3), BG_COLOR, dtype=np.uint8)
        for color, (r, c) in self.objects.items():
            self._paint_cell(img, r, c, COLORS[color])
        self._paint_cell(img, *self.agent_pos, AGENT_COLOR)
        return img

    @staticmethod
    def _paint_cell(img, r, c, color):
        img[r * CELL_PX:(r + 1) * CELL_PX, c * CELL_PX:(c + 1) * CELL_PX] = color
