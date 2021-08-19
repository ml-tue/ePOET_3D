import numpy as np
import mujoco_py
# from nexabots.src import my_utils as my_utils
import os
import cv2
from gym.utils import seeding
from collections import namedtuple
from scipy import ndimage
from math import acos

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
TERRAIN_STARTPAD = 20    # in steps

class AntTerrainMjc:
    def __init__(self, animate=False, sim=None, camera=False, heightfield=True):
        if camera:
            import cv2
            self.prev_img = np.zeros((24,24))

        if sim is not None:
            self.sim = sim
            self.model = self.sim.model
        else:
            self.modelpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/ant_terrain_mjc.xml")
            self.model = mujoco_py.load_model_from_path(self.modelpath)
            self.sim = mujoco_py.MjSim(self.model)

        self.env_config = None
        self.env_params = None
        self.env_seed = None
        self._seed()

        self._healthy_z_range = (0.2, 1.0)
        self._z_reward_weight = 0.05
        self._healthy_reward = 1.0 * self._z_reward_weight
        self._terminate_when_unhealthy = False
        self._contact_force_range = (-1.0, 1.0)
        self._ctrl_cost_weight = 0.3 #0.5
        self._contact_cost_weight = 5e-4
        self.forward_weight = 70
        self.max_reward = 0.0
        self.reward_threshold = 2000.0

        # External parameters
        self.joints_rads_low = np.array([-70, -30] * 4)
        self.joints_rads_high = np.array([30, 70] * 4)
        self.joints_rads_diff = self.joints_rads_high - self.joints_rads_low
        self.target_vel = 0.4  # Target velocity with which we want agent to move
        self.max_steps = 1000

        self.camera = camera
        self.animate = animate
        self.HF = heightfield
        self.HF_div = 5

        # if self.HF:
        #     self.hf_data = self.model.hfield_data
        #     self.hf_ncol = self.model.hfield_ncol[0]
        #     self.hf_nrow = self.model.hfield_nrow[0]
        #     self.hf_size = self.model.hfield_size[0]
        #     self.hf_grid = self.hf_data.reshape((self.hf_nrow, self.hf_ncol))
        #     self.hf_grid_aug = np.zeros((self.hf_nrow * 2, self.hf_ncol * 2))
        #     self.hf_grid_aug[:self.hf_nrow, :self.hf_ncol] = self.hf_grid
        #     self.hf_m_per_cell = float(self.hf_size[1]) / self.hf_nrow
        #     self.rob_dim = 0.5
        #     self.hf_res = int(self.rob_dim / self.hf_m_per_cell)
        #     self.hf_offset_x = 4
        #     self.hf_offset_y = 3
        #     self._healthy_z_range = (0.2, 0.5+np.max(self.model.hfield_data))

        self.model.opt.timestep = 0.02

        # Environment dimensions
        self.q_dim = self.sim.get_state().qpos.shape[0]
        self.qvel_dim = self.sim.get_state().qvel.shape[0]

        self.obs_dim = self.q_dim + self.qvel_dim - 2 + 4 + (24**2) * 2 # x,y not present, + 4contacts
        self.act_dim = self.sim.data.actuator_length.shape[0]

        # Environent inner parameters
        self.viewer = None
        self.step_ctr = 0

        if camera:
            self.cam_viewer = mujoco_py.MjRenderContextOffscreen(self.sim, 0)

        self.frame_list = []

        # Initial methods
        if animate:
            self.setupcam()

        self.reset()

    def seed(self, seed=None):
        return self._seed(seed)

    def _seed(self, seed=None):
        self.env_seed = seed
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_env_config(self, env_config):
        self.config = env_config

    def augment(self, params):
        self.env_params = params

    def state_vector(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = (np.isfinite(state).all() and min_z <= state[2] <= max_z)
        return is_healthy

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


    def setupcam(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 1.3
        self.viewer.cam.lookat[0] = -0.1
        self.viewer.cam.lookat[1] = 0
        self.viewer.cam.lookat[2] = 0.5
        self.viewer.cam.elevation = -20

    def get_no_cont_penlty(self):
        c_forces = []
        # print(self.model.geom_names)
        penetration = 0
        touch = 0
        no_touch_penlty = 0.0
        penetration_penlty = 0.0
        body_penlty = 0.0
        # print(self.model.geom_contype)
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            if ((self.model.geom_id2name(contact.geom1) == 'left_ankle_geom'
                 or self.model.geom_id2name(contact.geom1) == 'right_ankle_geom'
                 or self.model.geom_id2name(contact.geom1) == 'third_ankle_geom'
                 or self.model.geom_id2name(contact.geom1) == 'fourth_ankle_geom') and self.model.geom_id2name(contact.geom2) == 'floor') or \
                ((self.model.geom_id2name(contact.geom2) == 'left_ankle_geom'
                  or self.model.geom_id2name(contact.geom2) == 'right_ankle_geom'
                  or self.model.geom_id2name(contact.geom2) == 'third_ankle_geom'
                  or self.model.geom_id2name(contact.geom2) == 'fourth_ankle_geom') and self.model.geom_id2name(contact.geom1) == 'floor'):

                if contact.dist < 0:
                    penetration += 1
                else:
                    touch += 1
            if (self.model.geom_id2name(contact.geom1) == 'floor' and self.model.geom_id2name(contact.geom2) == 'torso_geom') or \
                    (self.model.geom_id2name(contact.geom2) == 'floor' and self.model.geom_id2name(
                        contact.geom1) == 'torso_geom'):
                body_penlty = 1.0
                # c_array = np.zeros(6, dtype=np.float64)
                # mujoco_py.functions.mj_contactForce(self.model, self.sim.data, i, c_array)
                # # Convert the contact force from contact frame to world frame
                # ref = np.reshape(contact.frame, (3, 3))
                # c_force = np.dot(np.linalg.inv(ref), c_array[0:3])
                # # print('contact force in world frame:', c_force)
                # c_forces.append(c_force)
        if touch == 0:
            no_touch_penlty = 1.0
        elif penetration > 0:
            penetration_penlty = 1.0 * penetration
        return no_touch_penlty + penetration_penlty + body_penlty

    def get_obs(self):
        qpos = self.sim.get_state().qpos.tolist()
        qvel = self.sim.get_state().qvel.tolist()
        a = qpos + qvel
        return np.asarray(a, dtype=np.float32)

    def get_obs_dict(self):
        od = {}

        od['rangefinder'] = self.rangefinder()
        od['contact_sensors'] = self.contact_sensor()

        # Intrinsic parameters
        for j in self.sim.model.joint_names:
            od[j + "_pos"] = self.sim.data.get_joint_qpos(j)
            od[j + "_vel"] = self.sim.data.get_joint_qvel(j)

        # Height field
        if self.HF:
            od["hf"] = self.get_local_hf(*od["root_pos"][0:2])

        if self.camera:
            # On board camera input
            cam_array = self.sim.render(camera_name="frontal", width=24, height=24)
            img = cv2.cvtColor(np.flipud(cam_array), cv2.COLOR_BGR2GRAY)
            od['cam'] = img

        # Contacts:
        od['contacts'] = np.clip(np.square(np.array(self.sim.data.cfrc_ext[[4, 7, 10, 13]])).sum(axis=1), 0, 1)

        return od

    def get_the_hf(self, x, y):
        x_coord = int((x + self.hf_offset_x) * 5)
        y_coord = int((y + self.hf_offset_y) * 5)
        return self.hf_grid_aug[y_coord-1:y_coord+1, x_coord-1:x_coord+1]

    def get_local_hf(self, x, y):
        x_coord = int((x + self.hf_offset_x) * 5)
        y_coord = int((y + self.hf_offset_y) * 5)
        return self.hf_grid_aug[y_coord - self.hf_res: y_coord + self.hf_res,
               x_coord - self.hf_res: x_coord + self.hf_res]

    def get_state(self):
        return self.sim.get_state()

    def set_state(self, qpos, qvel=None):
        qvel = np.zeros(self.qvel_dim) if qvel is None else qvel
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    def render(self, human=True):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
        if not human:
            return self.sim.render(camera_name=None,
                                   width=224,
                                   height=224,
                                   depth=False)
        self.viewer.render()

    def get_body_com(self, body_name):
        return self.sim.data.get_body_xpos(body_name)

    def get_sensor(self):
        return self.sim.data.sensordata

    def rangefinder(self):
        rf_readings = self.get_sensor()[4:]
        no_intersection = -1.0
        return np.where(rf_readings == no_intersection, 1.0, np.tanh(rf_readings))
    def contact_sensor(self):
        contacts = self.get_sensor()[:4]
        min_value, max_value = self._contact_force_range
        contacts = np.clip(contacts, min_value, max_value)
        return contacts

    def do_simulation(self, action, n_frames=1):
        self.sim.data.ctrl[:] = action
        self.sim.forward()
        for _ in range(n_frames):
            self.sim.step()

    def scale_action(self, action):
        # if (np.max(action) - np.min(action)) != 0:
        #     return (action - np.min(action)) / (np.max(action) - np.min(action))
        # else:
        #     return action
        return action
        # return (np.array(action) * 0.5 + 0.5) * self.joints_rads_diff + self.joints_rads_low
    def scale_joints(self, joints):
        # if (np.max(joints) - np.min(joints)) != 0:
        #     return (joints - np.min(joints)) / (np.max(joints) - np.min(joints))
        # else:
        #     return joints
        return joints
        # return ((np.array(joints) - self.joints_rads_low) / self.joints_rads_diff) * 2 - 1
    def get_agent_obs(self):
        qpos = self.sim.get_state().qpos.tolist()
        qvel = self.sim.get_state().qvel.tolist()
        contacts = np.array(self.sim.data.sensordata[0:4], dtype=np.float32)
        contacts[contacts > 0.05] = 1
        contacts[contacts <= 0.05] = -1
        if self.camera:
            cam_array = self.sim.render(camera_name="frontal", width=64, height=64)
            img = cv2.cvtColor(np.flipud(cam_array), cv2.COLOR_BGR2GRAY)
        return np.concatenate((self.scale_joints(qpos[7:]), qvel[6:], qpos[3:7], qvel[:6], contacts))

    @property
    def dt(self):
        return self.model.opt.timestep

    def step(self, ctrl):
        obs_p = self.get_agent_obs()
        # print('self.get_body_com("torso"): ', self.model.body_subtreemass)
        xy_position_before = self.get_body_com("torso")[:2].copy()
        ctrl = np.clip(ctrl, -1, 1)
        ctrl_pen = np.square(ctrl).mean()
        ctrl = self.scale_action(ctrl)

        # Step the simulator
        self.sim.data.ctrl[:] = ctrl
        self.sim.forward()
        self.sim.step()
        self.step_ctr += 1

        xy_position_after = self.get_body_com("torso")[:2].copy()
        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity
        forward_reward = x_velocity

        # ctrl_cost = self.control_cost(action)
        # contact_cost = self.contact_cost
        # healthy_reward = self.healthy_reward

        bx, by, bz, qw, qx, qy, qz = self.sim.get_state().qpos.tolist()[:7]
        xd, yd, zd, thd, phid, psid = self.sim.get_state().qvel.tolist()[:6]

        # print(xd )
        # print('xy_velocity', xy_velocity)

        # Reward conditions
        velocity_rew = 1. / (abs(xd - self.target_vel) + 1.) - 1. / (self.target_vel + 1.)
        obs_c = self.get_agent_obs()
        # velocity_rew = obs_c[0] - obs_p[0]

        q_yaw = 2 * acos(qw)

        r = forward_reward * 0.5 - \
            np.square(q_yaw) * 0.5 - \
            np.square(ctrl_pen) * 0.01 - \
            np.square(zd) * 0.5

        # r = velocity_rew * 10 - \
        #     np.square(q_yaw) * 0.5 - \
        #     np.square(ctrl_pen) * 0.01 - \
        #     np.square(zd) * 0.5
        # print('velocity ', velocity_rew)
        # print('np.square(q_yaw) ', np.square(q_yaw))
        # print('np.square(ctrl_pen) ', np.square(ctrl_pen))
        # print('np.square(zd) * 0.5 ', np.square(zd) * 0.5)
        # print('obs_c[0] - obs_p[0] ', obs_c[0] - obs_p[0])

        self.max_reward = max(self.max_reward, r)

        obs_dict = self.get_obs_dict()
        # obs = np.concatenate(
        #     (obs_c.astype(np.float32)[2:], obs_dict["contacts"], obs_dict['rangefinder'], obs_dict['contact_sensors']))

        # Reevaluate termination condition
        # done = self.step_ctr > self.max_steps # or abs(roll) > 0.8 or abs(pitch) > 0.8
        done = bz < 0 or self.step_ctr > self.max_steps

        return obs_c, r, done, obs_dict
    #
    # def step(self, action):
    #
    #     obs_p = self.get_obs()
    #     self.do_simulation(action, n_frames=2)
    #     self.step_ctr += 1
    #
    #     #print(self.sim.data.ncon) # Prints amount of current contacts
    #     obs_c = self.get_obs()
    #     x,y,z = obs_c[0:3]
    #
    #     cost = self.control_cost(action) + self.contact_cost
    #     forward_reward = (obs_c[0] - obs_p[0]) * self.forward_weight
    #
    #     obs_dict = self.get_obs_dict()
    #     obs = np.concatenate((obs_c.astype(np.float32)[2:], obs_dict["contacts"], obs_dict['rangefinder'], obs_dict['contact_sensors']))
    #
    #     rewards = forward_reward - cost + self.healthy_reward
    #
    #     if self.camera:
    #         obs = np.concatenate((obs, obs_dict["cam"].flatten(), self.prev_img.flatten()))
    #         self.prev_img = obs_dict["cam"]
    #
    #     # Reevaluate termination condition
    #     done = self.step_ctr > 1000 or z < 0.1
    #
    #     return obs, rewards, done, obs_dict

    def reset(self):
        if self.max_reward > self.reward_threshold or self.step_ctr == 0:
            self.gen_terrains()

        self.step_ctr = 0
        init_q = np.zeros(self.q_dim, dtype=np.float32)
        init_q[0] = np.random.randn() * 0.1
        init_q[1] = np.random.randn() * 0.1
        # init_q[2] = 0.80 + np.random.rand() * 0.1
        init_q[2] = min(np.max(self.get_local_hf(init_q[0], init_q[1])) + np.random.randn() * 0.1 + 0.2, .80 + np.random.rand() * 0.1)
        init_qvel = np.random.randn(self.qvel_dim).astype(np.float32) * 0.1

        self.set_state(init_q, init_qvel)
        obs, _, _, _ = self.step(np.zeros(self.act_dim))

        # obs = np.concatenate((init_q, init_qvel)).astype(np.float32)
        obs_dict = self.get_obs_dict()
        # obs = np.concatenate((obs[2:], obs_dict["contacts"], obs_dict['rangefinder'], obs_dict['contact_sensors']))

        if self.camera:
            obs = np.concatenate((obs, obs_dict["cam"].flatten(), self.prev_img.flatten()))
            self.prev_img = obs_dict["cam"]

        return obs

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

        if self.env_params is not None:
            # Terrain is elementwise product.
            terrain = bowl_shape * smooth_bumps * self._generate_terrain()
        else:
            terrain = bowl_shape * smooth_bumps

        terrain = (terrain - np.min(terrain)) / (np.max(terrain) - np.min(terrain)) * (self.model.hfield_size[_HEIGHTFIELD_ID, 2] - 0.0) + 0.0
        start_idx = self.model.hfield_adr[_HEIGHTFIELD_ID]
        self.model.hfield_data[start_idx:start_idx+res*self.model.hfield_ncol[_HEIGHTFIELD_ID]] = terrain.ravel()

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
        self.hf_offset_x = 10
        self.hf_offset_y = 20
        self._healthy_z_range = (0.2, 0.5 + np.max(self.model.hfield_data))

    def _generate_terrain(self):
        velocity = 0.0
        z = TERRAIN_HEIGHT
        z_norm = 0.0
        terrain_z = []
        nrows = self.model.hfield_nrow[_HEIGHTFIELD_ID]
        ncols = self.model.hfield_ncol[_HEIGHTFIELD_ID]

        for y in range(nrows):
            for x in range(ncols):
                nx = x / nrows - 0.5
                ny = y / ncols - 0.5
                velocity = 0.8 * velocity + 0.01 * np.sign(TERRAIN_HEIGHT - z)
                if self.env_params is not None and self.env_params.altitude_fn is not None:
                    z += velocity
                    if y > TERRAIN_STARTPAD:
                        mid = nrows * _TERRAIN_BUMP_SCALE / 2.
                        y_ = (ny - mid) * np.pi / mid
                        x_ = (nx - mid) * np.pi / mid
                        z = TERRAIN_HEIGHT + self.env_params.altitude_fn((x_, y_))[0]
                        if y == TERRAIN_STARTPAD+1:
                            z_norm = self.env_params.altitude_fn((x_, y_))[0]
                        z -= z_norm
                else:
                    if y > TERRAIN_STARTPAD:
                        velocity += np.random.uniform(-1, 1) / SCALE
                    z += _TERRAIN_SMOOTHNESS * velocity
                terrain_z.append(z)
        terrain_z = np.array(terrain_z).reshape(nrows,ncols)
        terrain = (terrain_z - np.min(terrain_z)) / (np.max(terrain_z) - np.min(terrain_z)) * (self.model.hfield_size[_HEIGHTFIELD_ID,2] - 0.2) + 0.2
        return terrain

    def demo(self):
        self.reset()
        if self.HF:
            cv2.namedWindow("HF")
        if self.camera:
            cv2.namedWindow("cam")
        cv2.namedWindow("con")

        for i in range(1000):
            _, _, _, od = self.step(np.random.randn(self.act_dim))

            # LED IDS: 4,7,10,13
            cv2.imshow("con", np.array(self.sim.data.cfrc_ext[[4, 7, 10, 13]]))
            cv2.waitKey(1)

            if self.animate:
                self.render()

            if self.HF:
                hf = od['hf']
                cv2.imshow("HF", np.flipud(hf))
                cv2.waitKey(1)

            if self.camera:
                cv2.imshow("cam", cv2.resize(od['cam'], (24, 24)))
                cv2.waitKey(1)

if __name__ == "__main__":
    ant = AntTerrainMjc(animate=True, camera=False)
    ant.demo()

