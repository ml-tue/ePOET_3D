import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import mujoco_py
from scipy import ndimage
from collections import namedtuple
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

class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='ant.xml',
                 ctrl_cost_weight=0.5,
                 contact_cost_weight=5e-4,
                 healthy_reward=1.0,
                 terminate_when_unhealthy=True,
                 healthy_z_range=(0.2, 1.0),
                 contact_force_range=(-1.0, 1.0),
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=False):
        utils.EzPickle.__init__(**locals())

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range
        self._reset_noise_scale = reset_noise_scale
        self.frame_skip = 3

        self.env_config = None
        self.env_params = None
        self.target_vel = 0.4  # Target velocity with which we want agent to move
        self.max_steps = 50
        self.step_ctr = 0
        self.stand_ctr = 0
        self.HF = False

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)
        self.reset_model()

    def set_env_config(self, env_config):
        self.config = env_config

    def augment(self, params):
        self.env_params = params

    def get_state(self):
        return self.sim.get_state()

    def set_state(self, qpos, qvel=None):
        qvel = np.zeros(self.q_dim) if qvel is None else qvel
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    def get_local_hf(self, x, y):
        x_coord = int(x*2 + self.hf_offset_x)
        y_coord = int(y*2 + self.hf_offset_y)
        # print('x_', x)
        # print('y_', y)
        # print('x_coord', x_coord)
        # print('y_coord', y_coord)
        # print('self.hf_res', self.hf_grid[y_coord - self.hf_res: y_coord + self.hf_res,
        #        x_coord - self.hf_res: x_coord + self.hf_res])
        return self.hf_grid[y_coord - self.hf_res: y_coord + self.hf_res,
               x_coord - self.hf_res: x_coord + self.hf_res]

    def get_cont_force(self):
        c_forces = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if ((self.model.geom_id2name(contact.geom1) == 'left_ankle_geom'
                 or self.model.geom_id2name(contact.geom1) == 'right_ankle_geom'
                 or self.model.geom_id2name(contact.geom1) == 'third_ankle_geom'
                 or self.model.geom_id2name(contact.geom1) == 'fourth_ankle_geom') and self.model.geom_id2name(contact.geom2) == 'floor') or \
                ((self.model.geom_id2name(contact.geom2) == 'left_ankle_geom'
                  or self.model.geom_id2name(contact.geom2) == 'right_ankle_geom'
                  or self.model.geom_id2name(contact.geom2) == 'third_ankle_geom'
                  or self.model.geom_id2name(contact.geom2) == 'fourth_ankle_geom') and self.model.geom_id2name(contact.geom1) == 'floor'):

                c_array = np.zeros(6, dtype=np.float64)
                mujoco_py.functions.mj_contactForce(self.model, self.data, i, c_array)
                # Convert the contact force from contact frame to world frame
                ref = np.reshape(contact.frame, (3, 3))
                c_force = np.dot(np.linalg.inv(ref), c_array[0:3])
                # print('contact force in world frame:', c_force)
                c_forces.append(c_force)
        return c_forces

    def get_obs_dict(self):
        od = {}
        for j in self.sim.model.joint_names:
            od[j + "_pos"] = self.sim.data.get_joint_qpos(j)
            od[j + "_vel"] = self.sim.data.get_joint_qvel(j)
        if self.HF:
            # print(*od["root_pos"][0:2])
            od["hf"] = self.get_local_hf(*od["root_pos"][0:2])
        else:
            od["hf"] = [[0,0],[0,0]]
        contacts = np.array(self.sim.data.sensordata[0:5], dtype=np.float32)
        min_value, max_value = self._contact_force_range
        contacts = np.clip(contacts, min_value, max_value)
        # contacts[4] = -contacts[4]
        # print(contacts)
        # con_for = self.get_cont_force()
        od['contacts'] = contacts
        # od['contacts'] = np.clip(np.square(np.array(self.sim.data.cfrc_ext[[4, 7, 10, 13]])).sum(axis=1), 0, 1)
        return od

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
        is_healthy = (np.isfinite(state).all() and min_z <= state[2] <= max_z)
        return is_healthy

    @property
    def done(self):
        done = (not self.is_healthy
                if self._terminate_when_unhealthy
                else False)
        return done

    def step(self, action):
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()
        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity
        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost
        forward_reward = x_velocity
        healthy_reward = self.healthy_reward

        if self.data.site_xpos[self.model.site_name2id('torso_back')][2] < self.data.site_xpos[self.model.site_name2id('l0')][2] \
                and self.data.site_xpos[self.model.site_name2id('torso_back')][2] < self.data.site_xpos[self.model.site_name2id('l1')][2] \
            and self.data.site_xpos[self.model.site_name2id('torso_back')][2] < self.data.site_xpos[self.model.site_name2id('l2')][2] \
            and self.data.site_xpos[self.model.site_name2id('torso_back')][2] < self.data.site_xpos[self.model.site_name2id('l3')][2]:
            stand_reward = -1.0
            self.stand_ctr += 1
        else:
            stand_reward = 0.0

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost + contact_cost
        reward = rewards - costs
        observation = self._get_obs()

        obs_dict = self.get_obs_dict()
        pen_r = 0.0
        if self.HF:
            if self.sim.get_state().qpos[6] < np.min(obs_dict['hf']) + 0.5:
                pen_r = -1.0
            else:
                pen_r = 0.0
        obs = np.concatenate((observation.astype(np.float32)[2:], obs_dict["contacts"], np.array(obs_dict['hf']).reshape(-1)))
        contact_reward = obs_dict["contacts"]
        # print(contact_reward)

        # ctrl_pen = np.square(action).mean()
        self.step_ctr += 1
        # _, _, _, qw, qx, qy, qz = self.sim.get_state().qpos.tolist()[:7]
        # xd, yd, zd, thd, phid, psid = self.sim.get_state().qvel.tolist()[:6]
        # # Reward conditions
        # velocity_rew = 1. / (abs(xd - self.target_vel) + 1.) - 1. / (self.target_vel + 1.)
        # q_yaw = 2 * acos(qw)
        # # r = velocity_rew * 10 - \
        # #     np.square(q_yaw) * 0.5 - \
        # #     np.square(ctrl_pen) * 0.01 - \
        # #     np.square(zd) * 0.5

        r = reward + stand_reward + pen_r
        done = self.stand_ctr > self.max_steps or self.done
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

        return obs, r, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()  # shape:15
        velocity = self.sim.data.qvel.flat.copy()  # shape: 14
        contact_force = self.contact_forces.flat.copy()  #shape: 84
        # print(self.sim.data.sensordata)
        # contacts = np.array(self.sim.data.sensordata[0:5], dtype=np.float32)
        # min_value, max_value = self._contact_force_range
        # contacts_f = np.clip(contacts, min_value, max_value)
        # contacts[contacts > 0.05] = 1
        # contacts[contacts <= 0.05] = 0

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        observations = np.concatenate((position, velocity, contact_force))
        return observations

    # def get_agent_obs(self):
    #     # Joints, joint velocities, quaternion, pose velocities (xd,yd,zd,thd,phd,psid), foot contacts
    #     return np.concatenate((self.scale_joints(qpos[7:]), qvel[6:], qpos[3:7], qvel[:6], contacts))
    # def reset(self):
    #     self.obs_dim = 12 + 12 + 4 + 6 + 4 # j, jd, quat, pose_velocity, contacts

    def reset_model(self):
        self.HF = True
        self.gen_terrains()
        self.q_dim = self.sim.get_state().qpos.shape[0]
        self.qvel_dim = self.sim.get_state().qvel.shape[0]

        if self.HF:
            self.hf_data = self.model.hfield_data
            self.hf_ncol = self.model.hfield_ncol[0]
            self.hf_nrow = self.model.hfield_nrow[0]
            self.hf_size = self.model.hfield_size[0]
            self.hf_grid = self.hf_data.reshape((self.hf_nrow, self.hf_ncol))
            # self.hf_grid_aug = np.zeros((self.hf_nrow * 2, self.hf_ncol * 2))
            # self.hf_grid_aug[:self.hf_nrow, :self.hf_ncol] = self.hf_grid
            self.hf_m_per_cell = float(self.hf_size[1]) / self.hf_nrow
            self.rob_dim = 0.5
            self.hf_res = int(self.rob_dim / self.hf_m_per_cell)
            self.hf_offset_x = self.model.hfield_size[0][0]
            self.hf_offset_y = self.model.hfield_size[0][1]

            self.hf_column_meters = self.model.hfield_size[0][0] * 2
            self.hf_row_meters = self.model.hfield_size[0][1] * 2
            self.hf_height_meters = self.model.hfield_size[0][2]
            self.pixels_per_column = self.hf_ncol / float(self.hf_column_meters)
            self.pixels_per_row = self.hf_nrow / float(self.hf_row_meters)
            self.local_grid = self.hf_grid[np.int(self.hf_nrow/2 * self.pixels_per_row)-8:np.int(self.hf_nrow/2 * self.pixels_per_row)+8, \
                         np.int(self.hf_nrow/2 * self.pixels_per_row)-8:np.int(self.hf_nrow/2 * self.pixels_per_row)+8]
            # local_grid = self.hf_grid[45:55, 5:15]
            max_height = np.max(self.local_grid) * self.hf_height_meters

            init_q = np.zeros(self.q_dim, dtype=np.float32)
            init_q[0] = np.random.randn() * 0.1
            init_q[1] = np.random.randn() * 0.1
            # init_q[2] = np.max(self.local_grid) + np.random.randn() * 0.1 + 0.5
            # init_q[2] = np.random.randn() * 0.1
            # z = self.get_local_hf(0, 0)
            # print(self.hf_grid.shape)
            # print(init_q[2])
            init_q[2] = max_height + np.random.randn() * 0.1 + 0.5
            self._healthy_z_range = (0.2, 1.0 + max_height)
            init_qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
                self.model.nv)
        else:
            noise_low = -self._reset_noise_scale
            noise_high = self._reset_noise_scale
            init_q = self.init_qpos + self.np_random.uniform(
                low=noise_low, high=noise_high, size=self.model.nq)
            init_qvel = np.random.randn(self.qvel_dim).astype(np.float32) * 0.1

        self.set_state(init_q, init_qvel)

        # orientation = np.random.randn(4)
        # orientation /= np.linalg.norm(orientation)
        # self._find_non_contacting_height(orientation)
        # self.sim.forward()

        self.step_ctr = 0
        self.stand_ctr = 0
        self.model.opt.timestep = 0.02
        observation = self._get_obs()

        obs_dict = self.get_obs_dict()
        obs = np.concatenate((observation.astype(np.float32)[2:], obs_dict["contacts"], np.array(obs_dict['hf']).reshape(-1)))

        return obs

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def gen_terrains(self):
        level = np.random.uniform(0.5, 1.5, 1)
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
        # Terrain is elementwise product.
        terrain = bowl_shape * smooth_bumps
        start_idx = self.model.hfield_adr[_HEIGHTFIELD_ID]
        self.model.hfield_data[start_idx:start_idx+res**2] = terrain.ravel()

    def _find_non_contacting_height(self, orientation, x_pos=0.0, y_pos=0.0):
      z_pos = np.min(self.local_grid)
      gap = np.max(self.local_grid) - np.min(self.local_grid)
      # z_pos = np.mean(self.local_grid)  # Start embedded in the floor.
      num_contacts = 1
      num_attempts = 0
      # Move up in 1cm increments until no contacts.
      while num_contacts > 0:
        print(num_contacts)
        try:
          self.data.qpos[:3] = x_pos, y_pos, z_pos
          # self.data.qpos[3:7] = orientation
          self.sim.forward()
        except NotImplementedError:
          pass
        num_contacts = self.data.ncon
        # z_pos += 0.01
        z_pos += gap/num_contacts
        num_attempts += 1
        if num_attempts > 100000:
          raise RuntimeError('Failed to find a non-contacting configuration.')

    def reset_m(self):
        self.q_dim = self.sim.get_state().qpos.shape[0]
        self.qvel_dim = self.sim.get_state().qvel.shape[0]
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        init_q = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        init_qvel = np.random.randn(self.qvel_dim).astype(np.float32) * 0.1
        self.set_state(init_q, init_qvel)
        return
        # observation = self._get_obs()
        # return observation.astype(np.float32)[2:]
