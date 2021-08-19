import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from collections import namedtuple
from scipy import ndimage
import mujoco_py
from math import acos

DEFAULT_CAMERA_CONFIG = {
    'distance': 4.0,
}

Env_config = namedtuple('Env_config', [
    'name',
    'ground_roughness',
    'pit_gap',
    'stump_width',  'stump_height', 'stump_float',
    'stair_height', 'stair_width', 'stair_steps'
])

_HEIGHTFIELD_ID = 0
_TERRAIN_SMOOTHNESS = 0.15  # 0.0: maximally bumpy; 1.0: completely smooth.
_TERRAIN_BUMP_SCALE = 2  # Spatial scale of terrain bumps (in meters).
_DEFAULT_VALUE_AT_MARGIN = 0.1

SCALE = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

VIEWPORT_W = 600
VIEWPORT_H = 400

TERRAIN_STEP = 14 / SCALE
TERRAIN_LENGTH = 100000     # in steps
TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4
# TERRAIN_HEIGHT = 1.0
TERRAIN_GRASS = 10    # low long are grass spots, in steps
TERRAIN_STARTPAD = 120    # in steps

class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='ant.xml',
                 ctrl_cost_weight=0.05,
                 contact_cost_weight=5e-4,
                 healthy_reward=1.0,
                 terminate_when_unhealthy=True,
                 healthy_z_range=(0.2, 1.0),
                 contact_force_range=(-0.5, 0.5),
                 reset_noise_scale=0.01,
                 exclude_current_positions_from_observation=True):
        utils.EzPickle.__init__(**locals())

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self.env_config = None
        self.env_params = None
        self.sum_reward = 0.0
        self.reward_threshold = 3000.0
        self.max_steps = 5000
        self.over = 0
        self.step_ctr = 0
        self.viewer = None
        self.status = 'init'
        self.hf_offset_x = 17
        self.hf_offset_y = 20
        self.finish = False

        self.frame_skip = 2

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        mujoco_env.MujocoEnv.__init__(self, xml_file, self.frame_skip)

        self.q_dim = self.sim.get_state().qpos.shape[0]
        self.qvel_dim = self.sim.get_state().qvel.shape[0]
        self.act_dim = self.sim.data.actuator_length.shape[0]
        self.reset_model()

    def set_env_config(self, env_config):
        self.config = env_config

    def augment(self, params):
        self.env_params = params

    @property
    def healthy_reward(self):
        return float(
            self.is_healthy
            or self._terminate_when_unhealthy
        ) * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.sim.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces))
        return contact_cost

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        max_z = max(max_z, self.model.hfield_size[0,2]+0.5)
        is_healthy = (np.isfinite(state).all() and min_z <= state[2] <= max_z)
        return is_healthy

    @property
    def done(self):
        done = (not self.is_healthy
                if self._terminate_when_unhealthy
                else False)
        return done

    def step(self, action):
        obs_p = self._get_obs()
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()
        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity
        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost
        forward_reward = x_velocity
        healthy_reward = self.healthy_reward
        rewards = forward_reward + healthy_reward
        costs = ctrl_cost + contact_cost
        reward = forward_reward - costs

        obs_c = self._get_obs()
        self.sum_reward += reward
        if np.sum(obs_p[-4:]) == -4 and np.sum(obs_c[-4:]) == -4:
            self.over += 1
        else:
            self.over = 0
        self.step_ctr += 1

        # Angle deviation
        x, y, z, qw, qx, qy, qz = obs_c[:7]
        xd, yd, zd, thd, phid, psid = self.sim.get_state().qvel.tolist()[:6]
        # xa, ya, za, tha, phia, psia = self.sim.data.qacc.tolist()[:6]
        target_vel = 0.4
        velocity_rew = 1. / (abs(xd - target_vel) + 1.) - 1. / (target_vel + 1.)
        # velocity_rew = velocity_rew * (1 / (1 + 30 * np.square(yd)))
        q_yaw = 2 * acos(qw)

        r = velocity_rew * 10 - \
            np.square(q_yaw) * .05 - \
            np.square(costs) * .01- \
            np.square(zd) * 0.05

        done = self.sim.get_state().qpos.tolist()[2] < 0 or self.step_ctr > self.max_steps or self.over > 100

        info = {
            'reward_forward': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'reward_contact': -contact_cost,
            'reward_survive': healthy_reward,

            'x_position': xy_position_after[0],
            'y_position': xy_position_after[1],
            'distance_from_origin': np.linalg.norm(xy_position_after, ord=2),

            'x_velocity': x_velocity,
            'y_velocity': y_velocity,
            'forward_reward': forward_reward,
        }

        return obs_c, r, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        contact_force = self.contact_forces.flat.copy()
        if self._exclude_current_positions_from_observation:
            position = position[2:]
        contacts = np.array(self.sim.data.sensordata[0:4], dtype=np.float32)
        contacts[contacts > 0.05] = 1
        contacts[contacts <= 0.05] = -1
        # if self.status != 'init':
        #     local_terrains = np.array(self.get_local_hf(position[0], position[1]))
        # else:
        #     local_terrains = np.zeros((10,10))
        # observations = np.concatenate((position, velocity, contact_force, local_terrains.ravel(), contacts))
        observations = np.concatenate((position, velocity, contact_force, contacts))
        return observations

    def reset_model(self):
        # if self.sum_reward > self.reward_threshold or self.step_ctr <= 1:
        if self.status != 'init':
            self.distroy_terrains()
            self.gen_terrains()

        # self.viewer = None
        self.status = 'normal'
        self.over = 0
        self.step_ctr = 0
        self.sum_reward = 0.0
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        init_q = np.zeros(self.q_dim, dtype=np.float32)
        init_q[0] = self.init_qpos[0] + self.np_random.uniform(low=noise_low, high=noise_high, size=1)[0]
        init_q[1] = self.init_qpos[1] + self.np_random.uniform(low=noise_low, high=noise_high, size=1)[0]
        init_q[2] = 0.6 + self.np_random.uniform(low=noise_low, high=noise_high, size=1)[0]
        # init_q[2] = min(np.mean(self.get_local_hf(init_q[0], init_q[1])) + np.random.randn() * 0.1 + 0.2, self.model.hfield_size[0, 2]/2 + np.random.rand() * 0.1)
        init_qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(self.model.nv)
        self.set_state(init_q, init_qvel)
        obs,_,_,_ = self.step(np.zeros(self.act_dim))
        return obs

    def render(self, human=True):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
        if not human:
            return self.sim.render(camera_name=None,
                                   width=224,
                                   height=224,
                                   depth=False)
        self.viewer.render()

    # def viewer_setup(self):
    #     for key, value in DEFAULT_CAMERA_CONFIG.items():
    #         if isinstance(value, np.ndarray):
    #             getattr(self.viewer.cam, key)[:] = value
    #         else:
    #             setattr(self.viewer.cam, key, value)

    def get_local_hf(self, x, y):
        # print(x,y)
        x_coord = int((x + self.hf_offset_x) * 5)
        y_coord = int((y + self.hf_offset_y) * 5)
        # print('after', x_coord, y_coord)
        # print(self.hf_res)
        return self.hf_grid_aug[y_coord - self.hf_res+2: y_coord + self.hf_res+2,
               x_coord - self.hf_res+2: x_coord + self.hf_res+2]


    def distroy_terrains(self):
        import glfw
        if self.viewer is not None and self.viewer.window is not None:
            # print(glfw.get_current_context())
            # glfw.terminate()
            # glfw.init()
            # glfw.swap_buffers(self.viewer.window)
            # glfw.restore_window(self.viewer.window)
            glfw.destroy_window(self.viewer.window)
            # # glfw.poll_events()
            self.viewer = None
        # res = self.model.hfield_nrow[_HEIGHTFIELD_ID]
        # assert res == self.model.hfield_ncol[_HEIGHTFIELD_ID]
        # start_idx = self.model.hfield_adr[_HEIGHTFIELD_ID]
        # self.model.hfield_data[start_idx:start_idx+res*self.model.hfield_ncol[_HEIGHTFIELD_ID]] = np.zeros((res, res)).ravel()

    def gen_terrains(self):
        if self.model.hfield_size[0, 2] <= 2.2:
            self.model.hfield_size[0, 2] += 0.001
        level = np.random.uniform(0.5, self.model.hfield_size[0, 2]+0.5, 1)
        res = self.model.hfield_nrow[_HEIGHTFIELD_ID]
        assert res == self.model.hfield_ncol[_HEIGHTFIELD_ID]
        # Sinusoidal bowl shape.
        row_grid, col_grid = np.ogrid[-1:1:res*1j, -1:1:res*1j]
        radius = np.clip(np.sqrt(col_grid**2 + row_grid**2), .04, 1)
        bowl_shape = level - np.cos(2*np.pi*radius)/2
        # Random smooth bumps.
        terrain_size = 2 * self.model.hfield_size[_HEIGHTFIELD_ID, 0]
        bump_res = int(terrain_size / _TERRAIN_BUMP_SCALE)
        bumps = np.random.uniform(_TERRAIN_SMOOTHNESS, 1, (bump_res, bump_res))
        smooth_bumps = ndimage.zoom(bumps, res / float(bump_res))
        terrain_bowl = bowl_shape * smooth_bumps
        if self.env_params is not None:
            # Terrain is elementwise product.
            terrain_bowl = (terrain_bowl - np.min(terrain_bowl)) / (np.max(terrain_bowl) - np.min(terrain_bowl)) * \
                           (self.model.hfield_size[_HEIGHTFIELD_ID, 2] - 0.0) + 0.0
            terrain = terrain_bowl * 0.3 + self._generate_terrain() * 0.7
            # terrain = self._generate_terrain()
        else:
            terrain = bowl_shape * smooth_bumps
        # terrain = self._generate_terrain()
        terrain = (terrain - np.min(terrain)) / (np.max(terrain) - np.min(terrain)) * (self.model.hfield_size[_HEIGHTFIELD_ID, 2] - 0.0) + 0.0

        # self.difficulty_meassure(terrain)

        start_idx = self.model.hfield_adr[_HEIGHTFIELD_ID]
        self.model.hfield_data[start_idx:start_idx+res*self.model.hfield_ncol[_HEIGHTFIELD_ID]] = terrain.ravel()
        # print('variance: ', np.var(terrain.ravel()))

        self.hf_data = self.model.hfield_data
        self.hf_ncol = self.model.hfield_ncol[0]
        self.hf_nrow = self.model.hfield_nrow[0]
        self.hf_size = self.model.hfield_size[0]
        self.hf_grid = self.hf_data.reshape((self.hf_nrow, self.hf_ncol))
        self.hf_grid_aug = np.zeros((self.hf_nrow * 2, self.hf_ncol * 2))
        self.hf_grid_aug[:self.hf_nrow, :self.hf_ncol] = self.hf_grid
        self.hf_m_per_cell = float(self.hf_size[1]) / self.hf_nrow
        self.rob_dim = 0.5
        self.hf_res = int(self.rob_dim / self.hf_m_per_cell)
        self._healthy_z_range = (0.2, 0.5 + np.max(self.model.hfield_data))

        # self.hf_column_meters = self.model.hfield_size[0][0] * 2
        # self.hf_row_meters = self.model.hfield_size[0][1] * 2
        # self.hf_height_meters = self.model.hfield_size[0][2]
        # self.pixels_per_column = self.hf_ncol / float(self.hf_column_meters)
        # self.pixels_per_row = self.hf_nrow / float(self.hf_row_meters)
        # self.hf_grid = self.hf_data.reshape((self.hf_nrow, self.hf_ncol))
        # local_grid = self.hf_grid[45:55, 5:15]
        # max_height = np.max(local_grid) * self.hf_height_meters

    def _generate_terrain(self):
        # self.genstairs()
        velocity = 0.0
        z_norm = 0.0
        terrain_z = []
        nrows = self.model.hfield_nrow[_HEIGHTFIELD_ID]
        ncols = self.model.hfield_ncol[_HEIGHTFIELD_ID]
        TERRAIN_HEIGHT = self.model.hfield_size[_HEIGHTFIELD_ID, 2] / 2
        z = TERRAIN_HEIGHT
        for y in range(nrows):
            for x in range(ncols):
                # nx = x
                # ny = y
                nx = x/5 - self.hf_offset_x
                ny = y/5 - self.hf_offset_y
                # nx = x * TERRAIN_STEP
                # ny = y * TERRAIN_STEP
                # nx = x / nrows - 0.5
                # ny = y / ncols - 0.5
                velocity = 0.8 * velocity + 0.08 * np.sign(TERRAIN_HEIGHT - z)
                if self.env_params is not None and self.env_params.altitude_fn is not None:
                    z += velocity
                    if x < TERRAIN_STARTPAD:
                        # y_ = (ny + self.hf_offset_y)/20 -1
                        # x_ = (nx + self.hf_offset_x)/20 -1
                        mid = 12
                        y_ = (ny - mid) * np.pi / mid
                        x_ = (nx - mid) * np.pi / mid
                        # y_ = ny/(200*TERRAIN_STEP) * 2 - 1
                        # x_ = nx/(200*TERRAIN_STEP) * 2 - 1
                        z = TERRAIN_HEIGHT + self.env_params.altitude_fn((y_, x_))[0]
                        if y == TERRAIN_STARTPAD - 10 and x == TERRAIN_STARTPAD - 10:
                            z_norm = self.env_params.altitude_fn((y_, x_))[0]
                        z -= z_norm
                else:
                    if x < TERRAIN_STARTPAD:
                        velocity += np.random.uniform(-1, 1) / SCALE
                    z += _TERRAIN_SMOOTHNESS * velocity
                terrain_z.append(z)
        terrain_z = np.array(terrain_z).reshape(nrows,ncols)
        # terrain = terrain_z
        # print(np.var(terrain))
        terrain = (terrain_z - np.min(terrain_z)) / (np.max(terrain_z) - np.min(terrain_z)) * (self.model.hfield_size[_HEIGHTFIELD_ID,2] - 0.2) + 0.2
        return terrain

