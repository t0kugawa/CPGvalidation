import numpy as np
import pandas as pd
from loader.loader_cfg import SAMPLE_DT_MS, FRAME_POINTS


class MCLoader:
    def __init__(self, path: str):
        self.df = pd.read_csv(path)
        self.frame = 0
        self.time = 0.0
        self.dt_ms = SAMPLE_DT_MS

    def __len__(self) -> int:
        return len(self.df)

    def is_done(self) -> bool:
        return self.frame >= len(self.df)

    def reset(self) -> None:
        self.frame = 0
        self.time = 0.0

    def load_frame(self) -> np.ndarray | None:
        if self.is_done():
            return None
        frame = self.df.iloc[self.frame][FRAME_POINTS].to_numpy(dtype=np.float64)
        self.frame += 1
        self.time += self.dt_ms
        return frame

    def iter_frames(self):
        self.reset()
        while not self.is_done():
            yield self.load_frame()
