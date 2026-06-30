import torch
from legged_gym.envs.h1.h1 import H1Robot
from legged_gym.envs.h1_2.h1_2_config import H1_2Cfg


class H1_2Robot(H1Robot):
    """H1-2 variant of HugWBC.

    Differences from H1:
    - 27 DOF (vs 19): split ankle (pitch+roll), 3 wrist joints per arm
    - Root body is 'pelvis' (H1 uses 'torso' as root)
    - Foot contact detected at 'ankle_roll' links
    """

    def __init__(self, cfg: H1_2Cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _create_envs(self):
        super()._create_envs()
        # Compute extra DOF index groups specific to H1-2
        dof_names_lower = [n.lower() for n in self.dof_names]
        self.wrist_inds = [i for i, n in enumerate(dof_names_lower) if 'wrist' in n]
        self.knee_inds  = [i for i, n in enumerate(dof_names_lower) if 'knee'  in n]

    # -----------------------------------------------------------------------
    # Override rewards that used hardcoded H1 joint indices
    # -----------------------------------------------------------------------

    def _reward_standing_joint_deviation(self):
        # H1 used fixed indices [11..18] for arm joints. H1-2 has more arm DOF
        # (shoulder + elbow + wrist), so we build the index list dynamically.
        arm_inds = torch.tensor(
            self.shoulder_inds + self.elbow_inds + self.wrist_inds,
            dtype=torch.long, device=self.device)
        reward = torch.square(self.dof_pos - self.default_dof_pos)[:, arm_inds]
        reward[~self.standing_envs_mask] = 0
        return torch.sum(reward, dim=-1)

    def _reward_joint_power_distribution(self):
        # H1 hardcoded shank (knee) indices [3, 8]. H1-2 knees are at different
        # positions in the DOF array because of the extra ankle DOF.
        knee_inds = torch.tensor(self.knee_inds, dtype=torch.long, device=self.device)
        penalize = (torch.abs(self.torques) * torch.abs(self.dof_vel))[:, knee_inds].var(dim=-1)
        return (torch.square(penalize) * 1.e-8).clip(max=1)

