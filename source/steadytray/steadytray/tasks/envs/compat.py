from __future__ import annotations

from isaaclab.managers import TerminationTermCfg
from isaaclab.managers.termination_manager import TerminationManager


def DoneTerm(*args, track_only: bool = False, track_only_delay: float = 0.0, **kwargs):
    """Create a termination term while preserving legacy track-only behavior."""
    _patch_track_only_termination_manager()

    cfg = TerminationTermCfg(*args, **kwargs)
    cfg.track_only = track_only
    cfg.track_only_delay = track_only_delay
    return cfg


def _patch_track_only_termination_manager():
    """Teach Isaac Lab 5.x TerminationManager about legacy track-only terms."""
    if getattr(TerminationManager, "_steadytray_track_only_patch", False):
        return

    def compute(self) -> object:
        self._truncated_buf[:] = False
        self._terminated_buf[:] = False

        for i, term_cfg in enumerate(self._term_cfgs):
            value = term_cfg.func(self._env, **term_cfg.params)

            track_only_delay = getattr(term_cfg, "track_only_delay", 0.0)
            if track_only_delay > 0.0:
                episode_time = self._env.episode_length_buf * self._env.step_dt
                value = value & (episode_time >= track_only_delay)

            self._term_dones[:, i] = value

            if getattr(term_cfg, "track_only", False):
                continue
            if term_cfg.time_out:
                self._truncated_buf |= value
            else:
                self._terminated_buf |= value

        rows = self._term_dones.any(dim=1).nonzero(as_tuple=True)[0]
        if rows.numel() > 0:
            self._last_episode_dones[rows] = self._term_dones[rows]

        return self._truncated_buf | self._terminated_buf

    TerminationManager.compute = compute
    TerminationManager._steadytray_track_only_patch = True
