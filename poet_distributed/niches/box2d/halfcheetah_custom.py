import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from collections import namedtuple
from scipy import ndimage

Env_config = namedtuple('Env_config', [
    'name',
    'ground_roughness',
    'pit_gap',
    'stump_width',  'stump_height', 'stump_float',
    'stair_height', 'stair_width', 'stair_steps'
])

DEFAULT_CAMERA_CONFIG = {
    'distance': 4.0,
}

FPS = 50
SCALE = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

VIEWPORT_W = 600
VIEWPORT_H = 400

TERRAIN_STEP = 14 / SCALE
TERRAIN_LENGTH = 40000     # in steps
TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4
TERRAIN_GRASS = 10    # low long are grass spots, in steps
TERRAIN_STARTPAD = 20    # in steps

_HEIGHTFIELD_ID = 0
_TERRAIN_SMOOTHNESS = 0.15  # 0.0: maximally bumpy; 1.0: completely smooth.
_TERRAIN_BUMP_SCALE = 2  # Spatial scale of terrain bumps (in meters).
_DEFAULT_VALUE_AT_MARGIN = 0.1

class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='half_cheetah.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self.env_config = None
        self.env_params = None
        self.terrian = None
        self.game_over = False

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

        orientation = np.random.randn(2)
        orientation /= np.linalg.norm(orientation)
        self._find_non_contacting_height(orientation)
        self._generate_terrain()

        # print(self.sim.model.hfield_data)
        # self.reset_model()

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def set_env_config(self, env_config):
        self.config = env_config

    def augment(self, params):
        self.env_params = params

    def step(self, action):
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]
        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)
        ctrl_cost = self.control_cost(action)
        forward_reward = self._forward_reward_weight * x_velocity
        reward = forward_reward - ctrl_cost

        # ctrl_cost = - 0.1 * np.square(action).sum()
        # reward = ctrl_cost + x_velocity

        observation = self._get_obs()
        done = False

        finish = False
        # if self.game_over or self.sim.data.qpos[0] < 0:
        #     done = True
        if self.sim.data.qpos[0] > (TERRAIN_LENGTH - TERRAIN_GRASS) * TERRAIN_STEP:
            done = True
            finish = True

        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,

            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'finish': finish
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        self.scroll = position[0] - VIEWPORT_W / SCALE / 5

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset(self):
        return self.reset_model()

    def reset_model(self):
        self.game_over = False
        self.scroll = 0.0

        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation
    #
    # def viewer_setup(self):
    #     self.viewer.cam.distance = self.model.stat.extent * 0.5

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def _find_non_contacting_height(self, orientation, x_pos=0.0, y_pos=0.0):
        z_pos = 0.0  # Start embedded in the floor.
        num_contacts = 1
        num_attempts = 0
        # Move up in 1cm increments until no contacts.
        while num_contacts > 0:
            try:
                self.reset_model()
                # self.data.qpos['bfoot'][:3] = x_pos, y_pos, z_pos
                # self.data.qpos['torso'][3:] = orientation
                # self.data.qpos['rootx'] = x_pos
                # self.data.qpos['rooty'] = y_pos
                # self.data.qpos['rootz'] = z_pos
            except NotImplementedError:
                pass
            num_contacts = self.data.ncon
            z_pos += 0.01
            num_attempts += 1
            if num_attempts > 10000:
                raise RuntimeError('Failed to find a non-contacting configuration.')

    def _generate_terrain(self):
        velocity = 0.0
        z = TERRAIN_HEIGHT
        self.terrain = []
        nrows = self.model.hfield_nrow[_HEIGHTFIELD_ID]
        ncols = self.model.hfield_ncol[_HEIGHTFIELD_ID]
        mid = nrows * TERRAIN_STEP / 2.
        for i in range(nrows):
            x = i * TERRAIN_STEP
            for j in range(ncols):
                y = j * TERRAIN_STEP
                velocity = 0.8 * velocity + 0.01 * np.sign(TERRAIN_HEIGHT - z)
                if self.env_params is not None and self.env_params.altitude_fn is not None:
                    z += velocity
                    if i > TERRAIN_STARTPAD and j > TERRAIN_STARTPAD:
                        x_ = (x - mid) * np.pi / mid
                        y_ = (y - mid) * np.pi / mid
                        z = TERRAIN_HEIGHT + self.env_params.altitude_fn((x_, y_))[0]
                        if i == TERRAIN_STARTPAD+1:
                            z_norm = self.env_params.altitude_fn((x_, y_))[0]
                        z -= z_norm
                self.terrain.append(z)
        norm = np.linalg.norm(self.terrain)
        self.terrain = self.terrain / norm
        self.terrain = np.array(self.terrain).reshape(nrows,ncols)
        # print(self.terrain)

        # res = self.model.hfield_nrow[_HEIGHTFIELD_ID]
        # assert res == self.model.hfield_ncol[_HEIGHTFIELD_ID]
        # Sinusoidal bowl shape.
        row_grid, col_grid = np.ogrid[-1:1:nrows * 1j, -1:1:ncols * 1j]
        radius = np.clip(np.sqrt(col_grid ** 2 + row_grid ** 2), .04, 1)
        bowl_shape = 1.5 - np.cos(2 * np.pi * radius) / 2
        # print(bowl_shape.shape)
        # Random smooth bumps.
        terrain_size = 2 * self.model.hfield_size[_HEIGHTFIELD_ID, 0]
        bump_res = int(terrain_size / _TERRAIN_BUMP_SCALE)
        bumps = np.random.uniform(_TERRAIN_SMOOTHNESS, 1, (bump_res, bump_res))

        smooth_bumps = ndimage.zoom(bumps, nrows / float(bump_res))

        terrain = bowl_shape * smooth_bumps* self.terrain
        # print(terrain)
        start_idx = self.model.hfield_adr[_HEIGHTFIELD_ID]
        self.model.hfield_data[start_idx:start_idx + nrows *ncols] = terrain.ravel()

        # print(self.data.ncon)

        # np.savetxt("foo.csv", terrain, delimiter=",")
        # super()._generate_terrain()

        # self.model.hfield_data[0:] = self.terrain

        orientation = np.random.randn(2)
        orientation /= np.linalg.norm(orientation)
        self._find_non_contacting_height(orientation)