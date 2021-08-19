import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from collections import namedtuple
from scipy import ndimage

from scipy.stats import multivariate_normal
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import os
from gym.spaces import Box
import collections
from . import rewards, enums

STEP = 0.1
TERRAIN_CMAP = 'Greens'

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
TERRAIN_LENGTH = 100000     # in steps
TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4
# TERRAIN_HEIGHT = 1.0
TERRAIN_GRASS = 10    # low long are grass spots, in steps
TERRAIN_STARTPAD = 20    # in steps

_HEIGHTFIELD_ID = 0
_TERRAIN_SMOOTHNESS = 0.15  # 0.0: maximally bumpy; 1.0: completely smooth.
_TERRAIN_BUMP_SCALE = 2  # Spatial scale of terrain bumps (in meters).
_DEFAULT_VALUE_AT_MARGIN = 0.1

_TOES = ['toe_front_left', 'toe_back_left', 'toe_back_right', 'toe_front_right']

DEFAULT_PATH = '/tmp/mujoco_terrains'

class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='ant.xml',
                 ctrl_cost_weight=0.5,
                 contact_cost_weight=5e-4,
                 healthy_reward=1.0,
                 terminate_when_unhealthy=True,
                 healthy_z_range=(0.2, 2.0),
                 # healthy_z_range=(0.2, 10.0),
                 contact_force_range=(-1.0, 1.0),
                 reset_noise_scale=0.1,
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
        # self._sensor_types_to_names = {}
        # self._hinge_names = []

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

        orientation = np.random.randn(4)
        orientation /= np.linalg.norm(orientation)
        # self._find_non_contacting_height(orientation)
        self._generate_terrain()

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
        # print('contact_forces 1, ', contact_forces)
        # # print('contact force: ', contact_forces)
        # # print('torso: ', self.torso_velocity())
        # # print('contact: ', self.data.contact)
        # # print('origin_distance', self.origin_distance())
        # contact_forces = self.force_torque()
        # print('contact_forces 2, ', contact_forces.shape)
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
        is_healthy = (np.isfinite(state).all() and min_z <= state[2] <= max_z)
        return is_healthy

    @property
    def done(self):
        done = (not self.is_healthy
                if self._terminate_when_unhealthy
                else False)
        return done

    def set_env_config(self, env_config):
        self.config = env_config

    def augment(self, params):
        self.env_params = params

    def step(self, action):
        # print('self.get_body_com("torso"): ', self.model.body_subtreemass)
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        forward_reward = x_velocity
        healthy_reward = self.healthy_reward

        # print('get_reward', self.get_reward())
        # x_ind = np.int(xy_position_after[0])
        # y_ind = np.int(xy_position_after[1])
        # z_position_after = self.get_body_com("torso")[2].copy()
        # print('nhfielddata', self.model.hfield_data[x_ind*201 + y_ind])
        # print('z_position_after: ', z_position_after)

        # rewards = forward_reward * 0.5 + healthy_reward *0.2 + self.get_reward()[0] * 0.3
        costs = ctrl_cost + contact_cost

        comvel = self.get_body_comvel('torso')
        forward_reward2 = comvel[0]
        # print('forward_reward2', forward_reward2)
        # print('forward_reward', forward_reward)

        survive_reward = 0.05
        reward = forward_reward2 - costs + survive_reward
        # reward = rewards - costs
        done = self.done
        observation = self.get_current_obs()
        # observation = self._get_obs()
        # print('reward: ', reward)
        finish = False
        if self.sim.data.qpos[0] > (TERRAIN_LENGTH - TERRAIN_GRASS) * TERRAIN_STEP:
            done = True
            finish = True

        info = {
            'reward_forward': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'reward_contact': -contact_cost,
            'reward_survive': healthy_reward,

            # 'x_position': xy_position_after[0],
            # 'y_position': xy_position_after[1],
            'distance_from_origin': np.linalg.norm(xy_position_after, ord=2),

            'x_velocity': x_velocity,
            'y_velocity': y_velocity,
            # 'forward_reward': forward_reward,
            'finish': finish,
        }

        return observation, reward, done, info

    def get_body_comvel(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.data.cvel[idx]

    def get_body_com(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.data.subtree_com[idx]

    def get_body_xmat(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.data.body_xmat[idx].reshape((3, 3))

    def get_current_obs(self):
        return np.concatenate([
            self.data.qpos.flat,
            self.data.qvel.flat,
            np.clip(self.data.cfrc_ext, -1, 1).flat,
            self.get_body_xmat("torso").flat,
            self.get_body_com("torso"),
        ]).reshape(-1)

    # def _common_observations(self):
    #     """Returns the observations common to all tasks."""
    #     obs = collections.OrderedDict()
    #     obs['egocentric_state'] = self.egocentric_state()
    #     obs['torso_velocity'] = self.torso_velocity()
    #     obs['torso_upright'] = self.torso_upright()
    #     obs['force_torque'] = self.force_torque()
    #     obs['origin'] = self.origin()
    #     obs = np.concatenate((self.origin(), self.torso_velocity(), self.force_torque()))
    #     return obs

    # def _get_obs(self):
    #     position = self.sim.data.qpos.flat.copy()
    #     velocity = self.sim.data.qvel.flat.copy()
    #     contact_force = self.contact_forces.flat.copy()
    #
    #     if self._exclude_current_positions_from_observation:
    #         position = position[2:]
    #     observations = np.concatenate((position, velocity, contact_force))
    #     # print('force_torque', self.force_torque())
    #     # observations = np.concatenate((position, velocity, self.force_torque()))
    #
    #     # observations = self._common_observations()
    #
    #     return observations

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        self._generate_terrain()

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)
        self.set_state(qpos, qvel)

        # observation = self._get_obs()
        observation = self.get_current_obs()
        return observation

    def _reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)


    # def get_reward(self):
    #     terrain_size = self.model.hfield_size[_HEIGHTFIELD_ID, 0]
    #     escape_reward = rewards.tolerance(
    #         self.origin_distance(),
    #         bounds=(terrain_size, float('inf')),
    #         margin=terrain_size,
    #         value_at_margin=0,
    #         sigmoid='linear')
    #     return self._upright_reward(deviation_angle=20) * escape_reward
    #
    # def origin_distance(self):
    #     """Returns the distance from the origin to the workspace."""
    #     assert  'workspace' == self.model.site_names[0]
    #     return np.asarray(np.linalg.norm(self.data.site_xpos[0]))
    #
    # def origin(self):
    #     """Returns origin position in the torso frame."""
    #     idx = self.model.body_names.index('torso')
    #     torso_frame = self.data.body_xmat[idx].reshape((3, 3))
    #     torso_pos = self.data.body_xpos[idx]
    #     return -torso_pos.dot(torso_frame)
    #
    # def torso_upright(self):
    #     """Returns the dot-product of the torso z-axis and the global z-axis."""
    #     idx = self.model.body_names.index('torso')
    #     return np.asarray(self.data.body_xmat[idx])
    #
    # # def torso_velocity(self):
    # #     """Returns the velocity of the torso, in the local frame."""
    # #     return self.data.get_body_xvelp('torso')
    #
    # def torso_velocity(self):
    #     """Returns the velocity of the torso, in the local frame."""
    #     idx = self.model.sensor_names.index('velocimeter')
    #     return self.data.sensordata[idx].copy()
    #
    # def egocentric_state(self):
    #     """Returns the state without global orientation or position."""
    #     if not self._hinge_names:
    #         [hinge_ids] = np.nonzero(self.model.jnt_type ==
    #                                  enums.mjtJoint.mjJNT_HINGE)
    #         self._hinge_names = [self.model.joint_id2name(j_id)
    #                              for j_id in hinge_ids]
    #     return np.hstack((self.data.qpos[hinge_ids],
    #                       self.data.qvel[hinge_ids],
    #                       self.data.act))
    #
    # def toe_positions(self):
    #     """Returns toe positions in egocentric frame."""
    #     torso_frame = self.data.xmat['torso'].reshape(3, 3)
    #     torso_pos = self.data.xpos['torso']
    #     torso_to_toe = self.data.xpos[_TOES] - torso_pos
    #     return torso_to_toe.dot(torso_frame)
    #
    # def force_torque(self):
    #     """Returns scaled force/torque sensor readings at the toes."""
    #     # print(self.model.sensor_names)
    #     return np.arcsinh(self.data.sensordata[3:11])
    #
    # def _upright_reward(self, deviation_angle=0):
    #     deviation = np.cos(np.deg2rad(deviation_angle))
    #     return rewards.tolerance(
    #         self.torso_upright(),
    #         bounds=(deviation, float('inf')),
    #         sigmoid='linear',
    #         margin=1 + deviation,
    #         value_at_margin=0)





    def _find_non_contacting_height(self, orientation, x_pos=0.0, y_pos=0.0):
        z_pos = 0.0  # Start embedded in the floor.
        num_contacts = 1
        num_attempts = 0
        # print(self.model.joint_name2id('root'))
        print(self.data.qpos)
        # Move up in 1cm increments until no contacts.
        while num_contacts > 0:
            try:
                self._reset_model()
                # idx = self.model.
                self.data.qpos[:3] = x_pos, y_pos, z_pos
                self.data.qpos[3:7] = orientation
            except NotImplementedError:
                pass
            num_contacts = self.data.ncon
            print(self.data.ncon)
            z_pos += 0.01
            num_attempts += 1
            if num_attempts > 10000:
                raise RuntimeError('Failed to find a non-contacting configuration.')

    def _generate_terrain(self):
        # print('env_params: ', self.env_params)
        velocity = 0.0
        z = TERRAIN_HEIGHT
        z_norm = 0.0
        terrain_z = []
        nrows = self.model.hfield_nrow[_HEIGHTFIELD_ID]
        ncols = self.model.hfield_ncol[_HEIGHTFIELD_ID]
        nrows_size = np.int(self.model.hfield_size[_HEIGHTFIELD_ID, 0])
        ncols_size = np.int(self.model.hfield_size[_HEIGHTFIELD_ID, 0])

        for y in range(nrows_size):
            for x in range(ncols_size):
                nx = x / nrows - 0.5
                ny = y / ncols - 0.5
                velocity = 0.8 * velocity + 0.01 * np.sign(TERRAIN_HEIGHT - z)
                if self.env_params is not None and self.env_params.altitude_fn is not None:
                    z += velocity
                    if y > TERRAIN_STARTPAD:
                        mid = nrows_size * _TERRAIN_BUMP_SCALE / 2.
                        y_ = (ny - mid) * np.pi / mid
                        x_ = (nx - mid) * np.pi / mid
                        z = TERRAIN_HEIGHT + self.env_params.altitude_fn((x_, y_))[0]
                        if y == TERRAIN_STARTPAD+1:
                            z_norm = self.env_params.altitude_fn((x_, y_))[0]
                        z -= z_norm
                    # z += velocity
                    # z = (1.00 * (self.env_params.altitude_fn((1 * nx, 1 * ny))[0] / 2 + 0.5)
                    #      + 0.50 * self.env_params.altitude_fn((2 * nx, 2 * ny))[0]
                    #      + 0.25 * self.env_params.altitude_fn((4 * nx, 4 * ny))[0]
                    #      + 0.13 * self.env_params.altitude_fn((8 * nx, 8 * ny))[0]
                    #      + 0.06 * self.env_params.altitude_fn((16 * nx, 16 * ny))[0]
                    #      + 0.03 * self.env_params.altitude_fn((32 * nx, 32 * ny))[0])
                    # z = z / (1.00 + 0.50 + 0.25 + 0.13 + 0.06 + 0.03)
                    # if y == TERRAIN_STARTPAD + 1 or x == TERRAIN_STARTPAD + 1:
                    #     z_norm = self.env_params.altitude_fn((nx, ny))[0]
                else:
                    if y > TERRAIN_STARTPAD:
                        velocity += np.random.uniform(-1, 1) / SCALE
                    z += _TERRAIN_SMOOTHNESS * velocity

                # z -= z_norm
                terrain_z.append(z)

        terrain_z = np.array(terrain_z).reshape(nrows_size,nrows_size)

        row_grid, col_grid = np.ogrid[-1:1:nrows * 1j, -1:1:ncols * 1j]
        radius = np.clip(np.sqrt(col_grid ** 2 + row_grid ** 2), .04, 1)
        bowl_shape = 100.5 - np.cos(2 * np.pi * radius) / 2
        # bowl_shape = terrain_z - np.cos(2 * np.pi * radius) / 2

        # Random smooth bumps.
        terrain_size = 2 * self.model.hfield_size[_HEIGHTFIELD_ID, 0]
        bump_res = int(terrain_size / _TERRAIN_BUMP_SCALE)

        bumps = np.random.uniform(_TERRAIN_SMOOTHNESS, 1, (bump_res, bump_res))
        # smooth_bumps = ndimage.zoom(bumps, nrows / float(bump_res))
        smooth_bumps = ndimage.zoom(terrain_z, nrows / float(bump_res))
        # terrain = bowl_shape * smooth_bumps * _TERRAIN_SMOOTHNESS
        terrain = bowl_shape * smooth_bumps
        terrain = (terrain - np.min(terrain)) / (np.max(terrain) - np.min(terrain)) * (1.0 - 0.2) + 0.2
        # print('terrain', terrain)
        # print('terrain median', np.median(terrain))
        self.model.hfield_data[0:] = terrain.ravel()

        # ## another way of generating terrains
        # x, y, hfield = self.generate_hills(nrows/10, ncols/10, 1000)
        # h_center = int(0.5 * hfield.shape[0])
        # w_center = int(0.5 * hfield.shape[1])
        # fromrow, torow = w_center + int(-2.0 / STEP), w_center + int(-2.0 / STEP)
        # fromcol, tocol = h_center + int(0.0 / STEP), h_center + int(0.0 / STEP)
        # hfield[fromrow:torow, fromcol:tocol] = 0.0
        # K = np.ones((10, 10)) / 100.0
        # s = convolve2d(hfield[fromrow - 9:torow + 9, fromcol - 9:tocol + 9], K, mode='same', boundary='symm')
        # hfield[fromrow - 9:torow + 9, fromcol - 9:tocol + 9] = s
        # print(hfield)
        # self.model.hfield_data[0:] = hfield.ravel()

        orientation = np.random.randn(4)
        orientation /= np.linalg.norm(orientation)
        # self._find_non_contacting_height(orientation)

    # def _generate_terrain(self):
    #     # print('env_params: ', self.env_params)
    #     velocity = 0.0
    #     z = TERRAIN_HEIGHT
    #     z_norm = 0.0
    #     terrain_z = []
    #     nrows = self.model.hfield_nrow[_HEIGHTFIELD_ID]
    #     ncols = self.model.hfield_ncol[_HEIGHTFIELD_ID]
    #     mid = nrows * _TERRAIN_BUMP_SCALE / 2.
    #     for i in range(nrows):
    #         x = i * _TERRAIN_BUMP_SCALE
    #         for j in range(ncols):
    #             y = j * _TERRAIN_BUMP_SCALE * _TERRAIN_SMOOTHNESS
    #             velocity = 0.8 * velocity + 0.01 * np.sign(TERRAIN_HEIGHT - z)
    #             if self.env_params is not None and self.env_params.altitude_fn is not None:
    #                 z += velocity
    #                 # if i > TERRAIN_STARTPAD and j > TERRAIN_STARTPAD:
    #                 x_ = (x - mid) * np.pi / mid
    #                 y_ = (y - mid) * np.pi / mid
    #                 z = TERRAIN_HEIGHT + self.env_params.altitude_fn((x_, y_))[0]
    #                 # if i == TERRAIN_STARTPAD+1 or j == TERRAIN_STARTPAD+1:
    #                 #     z_norm = self.env_params.altitude_fn((x_, y_))[0]
    #                 # z -= z_norm
    #             terrain_z.append(z)
    #
    #     terrain_z = np.array(terrain_z).reshape(nrows,ncols)
    #
    #     row_grid, col_grid = np.ogrid[-1:1:nrows * 1j, -1:1:ncols * 1j]
    #     radius = np.clip(np.sqrt(col_grid ** 2 + row_grid ** 2), .04, 1)
    #     bowl_shape = 5.5 - np.cos(2 * np.pi * radius) / 2
    #     # bowl_shape = terrain_z - np.cos(2 * np.pi * radius) / 2
    #
    #     # Random smooth bumps.
    #     terrain_size = 2 * self.model.hfield_size[_HEIGHTFIELD_ID, 0]
    #     bump_res = int(terrain_size / _TERRAIN_BUMP_SCALE)
    #
    #     # bumps = np.random.uniform(_TERRAIN_SMOOTHNESS, 1, (bump_res, bump_res))
    #
    #     smooth_bumps = ndimage.zoom(terrain_z, nrows / float(bump_res))
    #
    #     terrain = bowl_shape * smooth_bumps * _TERRAIN_SMOOTHNESS
    #     terrain = (terrain - np.min(terrain)) / (np.max(terrain) - np.min(terrain)) * (1.0 - 0.2) + 0.2
    #     print('terrain', terrain)
    #     # print(terrain)
    #     start_idx = self.model.hfield_adr[_HEIGHTFIELD_ID]
    #     self.model.hfield_data[start_idx:start_idx + nrows *ncols] = terrain.ravel()
    #
    #     orientation = np.random.randn(4)
    #     orientation /= np.linalg.norm(orientation)
    #     self._find_non_contacting_height(orientation)



    def generate_hills(self, width, height, nhills):
        '''
        @param width float, terrain width
        @param height float, terrain height
        @param nhills int, #hills to gen. #hills actually generted is sqrt(nhills)^2
        '''
        # setup coordinate grid
        xmin, xmax = -width / 2.0, width / 2.0
        ymin, ymax = -height / 2.0, height / 2.0
        x, y = np.mgrid[xmin:xmax:STEP, ymin:ymax:STEP]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x;
        pos[:, :, 1] = y

        # generate hilltops
        xm, ym = np.mgrid[xmin:xmax:width / np.sqrt(nhills), ymin:ymax:height / np.sqrt(nhills)]
        mu = np.c_[xm.flat, ym.flat]
        sigma = float(width * height) / (nhills * 8)
        for i in range(mu.shape[0]):
            mu[i] = multivariate_normal.rvs(mean=mu[i], cov=sigma)

        # generate hills
        sigma = sigma + sigma * np.random.rand(mu.shape[0])
        rvs = [multivariate_normal(mu[i, :], cov=sigma[i]) for i in range(mu.shape[0])]
        hfield = np.max([rv.pdf(pos) for rv in rvs], axis=0)

        self.save_heightfield(x, y, hfield, fname='hill.png', path='/tmp/mujoco_terrains')
        self.save_texture(x, y, hfield, fname='ant_textures', path=None)

        return x, y, hfield

    # def save_heightfield(self, x, y, hfield, path=None):
    #     '''
    #     @param path, str (optional). If not provided, DEFAULT_PATH is used. Make sure the path + fname match the <file> attribute
    #         of the <asset> element in the env XML where the height field is defined
    #     '''
    #     fig = plt.figure()
    #     plt.contourf(x, y, -hfield, 100,
    #                  cmap=TERRAIN_CMAP)  # terrain_cmap is necessary to make sure tops get light color
    #     ax = fig.add_subplot(111, projection="3d")
    #     ax.plot_surface(x, y, hfield, cmap='terrain')
    #     # plt.imshow(hfield, cmap='terrain')
    #     plt.show()
    #     # plt.savefig(os.path.join(path, fname), bbox_inches='tight')
    #     # plt.close()

    def _checkpath(self, path_):
        if path_ is None:
            path_ = DEFAULT_PATH
        if not os.path.exists(path_):
            os.makedirs(path_)
        return path_

    def save_heightfield(self, x, y, hfield, fname, path=None):
        '''
        @param path, str (optional). If not provided, DEFAULT_PATH is used. Make sure the path + fname match the <file> attribute
            of the <asset> element in the env XML where the height field is defined
        '''
        path = self._checkpath(path)
        fig = plt.figure()
        plt.contourf(x, y, -hfield, 100,
                     cmap=TERRAIN_CMAP)  # terrain_cmap is necessary to make sure tops get light color
        plt.savefig(os.path.join(path, fname), bbox_inches='tight')
        plt.close()

    def save_texture(self, x, y, hfield, fname, path=None):
        '''
        @param path, str (optional). If not provided, DEFAULT_PATH is used. Make sure this matches the <texturedir> of the
            <compiler> element in the env XML
        '''
        path = self._checkpath(path)
        plt.figure()
        plt.contourf(x, y, -hfield, 100, cmap=TERRAIN_CMAP)
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        # for some reason plt.grid does not work here, so generate gridlines manually
        for i in np.arange(xmin, xmax, 0.5):
            plt.plot([i, i], [ymin, ymax], 'k', linewidth=0.1)
        for i in np.arange(ymin, ymax, 0.5):
            plt.plot([xmin, xmax], [i, i], 'k', linewidth=0.1)
        plt.savefig(os.path.join(path, fname), bbox_inches='tight')
        plt.close()
