import numpy as np
import mujoco

try:
    from PIL import Image, ImageDraw
    _HAS_PIL = True
except ImportError:  # pragma: no cover
    _HAS_PIL = False


class LegRenderer:
    """
    Offscreen MuJoCo renderer with a text HUD overlay.
    Use MUJOCO_GL=osmesa for headless rendering.
    """
    def __init__(self, model, data, width: int = 1920, height: int = 1080):
        self.model = model
        self.data = data
        self.width = int(width)
        self.height = int(height)
        self.renderer = mujoco.Renderer(model, self.height, self.width)

    def frame(self, camera: str, hud_lines=None) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera)
        img = self.renderer.render()
        if hud_lines and _HAS_PIL:
            im = Image.fromarray(img)
            draw = ImageDraw.Draw(im)
            y = 8
            for line in hud_lines:
                draw.text((10, y), line, fill=(255, 255, 0))
                y += 20
            img = np.asarray(im)
        return img

    def close(self):
        try:
            self.renderer.close()
        except Exception:
            pass


def hud_lines(cpg_in, cpg_out, t_ms: float, stage: str = "") -> list:
    """Builds the HUD text lines from the current sensory/motor state."""
    return [
        f"t = {t_ms/1000.0:5.2f} s   {stage}",
        f"CUT   L={cpg_in.cut_L:6.1f}  R={cpg_in.cut_R:6.1f} Hz",
        f"Ia    E={cpg_in.ia_rate_E_L:6.1f}  F={cpg_in.ia_rate_F_L:6.1f} Hz",
        f"Ib    E={cpg_in.ib_rate_E_L:6.1f}  F={cpg_in.ib_rate_F_L:6.1f} Hz",
        f"Ankle L model={cpg_in.ankle_angle_m_L:+.3f} ref={cpg_in.ankle_angle_ref_L:+.3f}",
        f"      R model={cpg_in.ankle_angle_m_R:+.3f} ref={cpg_in.ankle_angle_ref_R:+.3f}",
        f"Knee  L={cpg_in.knee_angle_m_L:+.3f}  Hip L={cpg_in.hip_angle_m_L:+.3f}",
        f"Muscle  SOL_L={cpg_out.mus_E_L:.2f} TA_L={cpg_out.mus_F_L:.2f}",
        f"        SOL_R={cpg_out.mus_E_R:.2f} TA_R={cpg_out.mus_F_R:.2f}",
    ]