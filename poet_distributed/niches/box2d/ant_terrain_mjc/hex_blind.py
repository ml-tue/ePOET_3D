import numpy as np
import mujoco_py
# import src.my_utils as my_utils
import time
import os
import cv2
# from src.envs.locom_benchmarks import hf_gen
import gym
from gym import spaces
from math import acos
from collections import namedtuple
from scipy import ndimage
from gym.utils import seeding
from .hex_all import quat_to_rpy,rpy_to_quat,to_tensor

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

class Hexapod(gym.Env):
    # MODELPATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
    #                          "assets/hex.xml")

    def __init__(self):
        # self.hm_fun_list = hm_fun_list
        # self.hm_args = hm_args

        # External parameters
        # self.joints_rads_low = np.array([-0.6, -1.2, -0.6] * 6)
        # self.joints_rads_high = np.array([0.6, 0.2, 0.6] * 6)
        self.joints_rads_low = np.array([-0.6, -1.4, -0.8] * 6)
        self.joints_rads_high = np.array([0.6, 0.4, 0.8] * 6)
        self.joints_rads_diff = self.joints_rads_high - self.joints_rads_low

        self.target_vel = 0.4 # Target velocity with which we want agent to move
        self.max_steps = 5000

        self.env_config = None
        self.env_params = None
        self.env_seed = None
        self._seed()

        self.viewer = None
        self.step_ctr = 0
        self.sum_reward = 0.0
        self.reward_threshold = 3000.0

        self.hf_offset_x = 16.5
        self.hf_offset_y = 20

        self.modelpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/hex.xml")
        self.model = mujoco_py.load_model_from_path(self.modelpath)
        self.sim = mujoco_py.MjSim(self.model)
        self.model.opt.timestep = 0.02

        self.camera = False
        self.reset()

        if self.camera:
            self.cam_viewer = mujoco_py.MjRenderContextOffscreen(self.sim, 0)


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

    # def setupcam(self):
    #     self.viewer = mujoco_py.MjViewer(self.sim)
    #     self.viewer.cam.trackbodyid = -1
    #     return
    #     self.viewer.cam.distance = self.model.stat.extent * .3
    #     self.viewer.cam.lookat[0] = 2.
    #     self.viewer.cam.lookat[1] = 0.3
    #     self.viewer.cam.lookat[2] = 0.9
    #     self.viewer.cam.elevation = -30
    #     self.viewer.cam.azimuth = -10

    def setupcam(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 1.3
        self.viewer.cam.lookat[0] = -0.1
        self.viewer.cam.lookat[1] = 0
        self.viewer.cam.lookat[2] = 0.5
        self.viewer.cam.elevation = -20

    def get_state(self):
        return self.sim.get_state()

    def set_state(self, qpos, qvel=None):
        qvel = np.zeros(self.q_dim) if qvel is None else qvel
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    def scale_action(self, action):
        return (np.array(action) * 0.5 + 0.5) * self.joints_rads_diff + self.joints_rads_low

    def scale_joints(self, joints):
        return ((np.array(joints) - self.joints_rads_low) / self.joints_rads_diff) * 2 - 1


    def get_agent_obs(self):
        qpos = self.sim.get_state().qpos.tolist()
        qvel = self.sim.get_state().qvel.tolist()
        contacts = np.array(self.sim.data.sensordata[0:6], dtype=np.float32)
        contacts[contacts > 0.05] = 1
        contacts[contacts <= 0.05] = -1
        if self.camera:
            # On board camera input
            cam_array = self.sim.render(camera_name="frontal", width=64, height=64)
            img = cv2.cvtColor(np.flipud(cam_array), cv2.COLOR_BGR2GRAY)

            # Rangefinder data
        #r0 = self.sim.data.get_sensor('r0')
        #r1 = self.sim.data.get_sensor('r1')

        # Joints, joint velocities, quaternion, pose velocities (xd,yd,zd,thd,phd,psid), foot contacts
        return np.concatenate((self.scale_joints(qpos[7:]), qvel[6:], qpos[3:7], qvel[:6], contacts))


    def step(self, ctrl):
        obs_p = self.get_agent_obs()

        # Clip control signal
        ctrl = np.clip(ctrl, -1, 1)

        # Control penalty
        ctrl_pen = np.square(ctrl).mean()

        # Scale control according to joint ranges
        ctrl = self.scale_action(ctrl)

        # Step the simulator
        self.sim.data.ctrl[:] = ctrl
        self.sim.forward()
        self.sim.step()
        self.step_ctr += 1

        # Get agent telemetry data
        bx, by, bz, qw, qx, qy, qz = self.sim.get_state().qpos.tolist()[:7]
        xd, yd, zd, thd, phid, psid = self.sim.get_state().qvel.tolist()[:6]
        self.get_local_hf(bx,by)
        # Reward conditions
        velocity_rew = 1. / (abs(xd - self.target_vel) + 1.) - 1. / (self.target_vel + 1.)
        q_yaw = 2 * acos(qw)

        r = velocity_rew * 10 - \
            np.square(q_yaw) * .5 - \
            np.square(ctrl_pen) * 0.01 - \
            np.square(zd) * 0.5

        # yaw_deviation = np.min((abs((q_yaw % 6.183) - (0 % 6.183)), abs(q_yaw - 0)))
        # y_deviation = by
        # roll, pitch, _ = quat_to_rpy([qw, qx, qy, qz])
        # r_neg = np.square(by) * 0.1 + \
        #         np.square(q_yaw) * 0.1 + \
        #         np.square(pitch) * 0.5 + \
        #         np.square(roll) * 0.5 + \
        #         ctrl_pen * 0.0001 + \
        #         np.square(zd) * 0.7
        # r_pos = velocity_rew * 6 + (abs(self.prev_deviation) - abs(yaw_deviation)) * 10 + (
        #             abs(self.prev_y_deviation) - abs(y_deviation)) * 10
        # self.prev_deviation = yaw_deviation
        # self.prev_y_deviation = y_deviation

        # ctrl_effort = np.square(ctrl).sum()
        # height_pen = np.square(zd)
        # r = (velocity_rew * 3.0,
        #       - ctrl_effort * 0.003,
        #       - np.square(q_yaw) * 0.1,
        #       - height_pen * 0.1 * int(self.step_ctr > 30))

        obs_c = self.get_agent_obs()
        if np.sum(obs_p[-6:]) == -6 and np.sum(obs_c[-6:]) == -6:
            self.over += 1
        else:
            self.over = 0

        self.sum_reward += r

        # Reevaluate termination condition
        # done = self.step_ctr > self.max_steps  # or abs(y) > 0.3 or abs(yaw) > 0.6 or abs(roll) > 0.8 or abs(pitch) > 0.8
        done = bz < -(self.model.hfield_size[0,2]+self.model.hfield_size[0,3]) or self.step_ctr > self.max_steps or self.over > 100
        # done = self.step_ctr > self.max_steps or (abs(q_yaw) > 2.4 and self.step_ctr > 30) or abs(obs_c[1]) > 2 or obs_c[0] < -1.0 or \
        #     bz < -(self.model.hfield_size[0, 2] + self.model.hfield_size[0, 3]) or self.over > 100


        return self.get_agent_obs(), r, done, None


    def reset(self):
        # Generate environment
        # hm_fun = np.random.choice(self.hm_fun_list)
        # hm, info = hm_fun(*self.hm_args)
        # cv2.imwrite(os.path.join(os.path.dirname(os.path.realpath(__file__)), "hm.png"), hm)
        # if self.sum_reward > self.reward_threshold or self.step_ctr <= 1:
        self.distroy_terrains()
        self.gen_terrains()
        # self.genstairs()
        # self.perlin()

        # # Load simulator
        # while True:
        #     try:
        #         self.model = mujoco_py.load_model_from_path(Hexapod.MODELPATH)
        #         break
        #     except Exception:
        #         pass

        # Set appropriate height according to height map
        # self.model.hfield_size[0][2] = info["height"]

        # self.sim = mujoco_py.MjSim(self.model)

        self.viewer = None
        self.over = 0
        self.step_ctr = 0
        self.sum_reward = 0.0

        # Environment dimensions
        self.q_dim = self.sim.get_state().qpos.shape[0]
        self.qvel_dim = self.sim.get_state().qvel.shape[0]

        self.obs_dim = 18 + 18 + 4 + 6 + 6 # j, jd, quat, pose_velocity, contacts
        self.act_dim = self.sim.data.actuator_length.shape[0]

        # Set initial position
        init_q = np.zeros(self.q_dim, dtype=np.float32)
        init_q[0] = 0.0
        init_q[1] = 0.0
        # init_q[2] = max_height + 0.05
        init_q[2] = min(np.max(self.get_local_hf(init_q[0], init_q[1])) + np.random.randn() * 0.1 + 0.2,
                        self.model.hfield_size[0, 2] / 2 + np.random.rand() * 0.1)
        init_qvel = np.random.randn(self.qvel_dim).astype(np.float32) * 0.1

        # Set environment state
        self.set_state(init_q, init_qvel)
        self.step_ctr = 0

        obs, _, _, _ = self.step(np.zeros(self.act_dim))

        return obs

    def get_local_hf(self, x, y):
        # print('x,y', x, y)
        x_coord = int((x + self.hf_offset_x) * 5)
        y_coord = int((y + self.hf_offset_y) * 5)
        # print('x_c, y_c: ', x_coord, y_coord)
        return self.hf_grid_aug[y_coord - self.hf_res: y_coord + self.hf_res,
               x_coord - self.hf_res: x_coord + self.hf_res]

    # def render(self, camera=None):
    #     if self.viewer is None:
    #         self.setupcam()
    #     self.viewer.render()

    def render(self, human=True):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
        if not human:
            return self.sim.render(camera_name=None,
                                   width=224,
                                   height=224,
                                   depth=False)
        self.viewer.render()

    def distroy_terrains(self):
        import glfw
        if self.viewer is not None:
            glfw.destroy_window(self.viewer.window)
            self.viewer = None
        res = self.model.hfield_nrow[_HEIGHTFIELD_ID]
        assert res == self.model.hfield_ncol[_HEIGHTFIELD_ID]
        start_idx = self.model.hfield_adr[_HEIGHTFIELD_ID]
        self.model.hfield_data[start_idx:start_idx+res*self.model.hfield_ncol[_HEIGHTFIELD_ID]] = np.zeros((res, res)).ravel()

    def gen_terrains(self):
        if self.model.hfield_size[0, 2] <= 2.2:
            self.model.hfield_size[0, 2] += 0.001
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

        # # Height field
        # self.hf_column_meters = self.model.hfield_size[0][0] * 2
        # self.hf_row_meters = self.model.hfield_size[0][1] * 2
        # self.hf_height_meters = self.model.hfield_size[0][2]
        # self.pixels_per_column = self.hf_ncol / float(self.hf_column_meters)
        # self.pixels_per_row = self.hf_nrow / float(self.hf_row_meters)
        # self.hf_grid = self.hf_data.reshape((self.hf_nrow, self.hf_ncol))
        # local_grid = self.hf_grid[45:55, 5:15]
        # max_height = np.max(local_grid) * self.hf_height_meters

    def _generate_terrain(self):
        self.genstairs()
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
                        z = TERRAIN_HEIGHT + self.env_params.altitude_fn((x_, y_))[0]
                        if y == TERRAIN_STARTPAD - 10 and x == TERRAIN_STARTPAD - 10:
                            z_norm = self.env_params.altitude_fn((x_, y_))[0]
                        z -= z_norm

                        # print(self.env_params.altitude_fn((x_, y_))[0])
                        # # print('dd: ', .5 - np.cos(2 * np.pi * (np.sqrt(x_ ** 2 + y_ ** 2))) / 2)

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
                    if x < TERRAIN_STARTPAD:
                        velocity += np.random.uniform(-1, 1) / SCALE
                    z += _TERRAIN_SMOOTHNESS * velocity
                terrain_z.append(z)
        terrain_z = np.array(terrain_z).reshape(nrows,ncols)
        # terrain = terrain_z
        # print(np.var(terrain))
        terrain = (terrain_z - np.min(terrain_z)) / (np.max(terrain_z) - np.min(terrain_z)) * (self.model.hfield_size[_HEIGHTFIELD_ID,2] - 0.2) + 0.2
        return terrain

    def difficulty_meassure(self, terrain):
        data = terrain[99,60:]
        dist = np.abs(np.diff(data))
        with open("level.txt", "a+") as file:
        # with open("/home/fang/project/thesis/ant_poet/poet/poet_distributed/niches/box2d/ant_terrain_mjc/level.txt", "a+") as file:
            file.seek(0)
            tmp = file.read(100)
            if len(tmp) > 0:
                file.write("\n")
            file.write(str(np.max(dist)))
            file.close()

    def genstairs(self):
        N = 200
        M = 200
        # filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
        #                         "assets/stairs.png")
        # Generate stairs
        mat = np.zeros((M, N))

        stair_height = 20
        stair_width = 3
        current_height = 0

        for i in range(6):
            mat[:, 10 + i * stair_width: 10 + i * stair_width + stair_width] = current_height
            current_height += stair_height

        for i in range(3):
            mat[:, 28 + i * stair_width:  28 + i * stair_width + stair_width] = current_height

        for i in range(4):
            mat[:, 37 + i * stair_width: 37 + i * stair_width + stair_width] = current_height
            current_height -= stair_height

        for i in range(2):
            mat[:, 49 + i * stair_width:  49 + i * stair_width + stair_width] = current_height

        for i in range(3):
            mat[:, 55 + i * stair_width: 55 + i * stair_width + stair_width] = current_height
            current_height -= stair_height

        # ---
        for i in range(12):
            mat[:, 55 + 10 + i * stair_width: 55 + 10 + i * stair_width + stair_width] = current_height
            current_height += stair_height

        for i in range(15):
            mat[:, 70 + 28 + i * stair_width: 70 + 28 + i * stair_width + stair_width] = current_height

        mat[0, :] = 255
        mat[:, 0] = 255
        mat[-1, :] = 255
        mat[:, -1] = 255
        mat = (mat - np.min(mat)) / (np.max(mat) - np.min(mat)) * (
                    self.model.hfield_size[_HEIGHTFIELD_ID, 2] - 0.2) + 0.2
        start_idx = self.model.hfield_adr[_HEIGHTFIELD_ID]
        self.model.hfield_data[start_idx:start_idx + M * self.model.hfield_ncol[_HEIGHTFIELD_ID]] = mat.ravel()

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
        self.hf_offset_x = 16.5
        self.hf_offset_y = 20
        self._healthy_z_range = (0.2, 0.5 + np.max(self.model.hfield_data))
        # import matplotlib.pyplot as plt
        # plt.imshow(mat, cmap='terrain')
        # plt.show()
        # cv2.imwrite(filename, mat)

    def perlin(self):
        from opensimplex import OpenSimplex
        oSim = OpenSimplex(seed=int(time.time()))

        height = 200

        M = self.model.hfield_ncol[0]
        N = self.model.hfield_nrow[0]
        mat = np.zeros((M, N))

        scale_x = np.random.randint(30, 100)
        scale_y = np.random.randint(30, 100)
        octaves = 4  # np.random.randint(1, 5)
        persistence = np.random.rand() * 0.3 + 0.3
        lacunarity = np.random.rand() + 1.5

        for i in range(M):
            for j in range(N):
                for o in range(octaves):
                    sx = scale_x * (1 / (lacunarity ** o))
                    sy = scale_y * (1 / (lacunarity ** o))
                    amp = persistence ** o
                    mat[i][j] += oSim.noise2d(i / sx, j / sy) * amp

        wmin, wmax = mat.min(), mat.max()
        mat = (mat - wmin) / (wmax - wmin) * height

        if np.random.rand() < 0.3:
            num = np.random.randint(50, 120)
            mat = np.clip(mat, num, 200)
        if np.random.rand() < 0.3:
            num = np.random.randint(120, 200)
            mat = np.clip(mat, 0, num)

        # Walls
        mat[0, 0] = 255.
        mat = (mat - np.min(mat))/(np.max(mat) - np.min(mat)) * (self.model.hfield_size[_HEIGHTFIELD_ID,2] - 0.2) + 0.2
        start_idx = self.model.hfield_adr[_HEIGHTFIELD_ID]
        self.model.hfield_data[start_idx:start_idx + M * self.model.hfield_ncol[_HEIGHTFIELD_ID]] = mat.ravel()

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
        self.hf_offset_x = 16.5
        self.hf_offset_y = 20
        self._healthy_z_range = (0.2, 0.5 + np.max(self.model.hfield_data))

    # def test(self, policy, render=True):
    #     N = 30
    #     rew = 0
    #     for i in range(N):
    #         obs = self.reset()
    #         cr = 0
    #         for j in range(int(self.max_steps)):
    #             action = policy(my_utils.to_tensor(obs, True)).detach()
    #             obs, r, done, od, = self.step(action[0].numpy())
    #             cr += r
    #             rew += r
    #             time.sleep(0.000)
    #             if render:
    #                 self.render()
    #         print("Total episode reward: {}".format(cr))
    #     print("Total average reward = {}".format(rew / N))


    def demo(self):
        for i in range(100000):
            self.sim.forward()
            self.sim.step()
            self.render()


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description='Enter input arguments')
#     parser.add_argument("--terrain", type=str,
#                         help="Terrain type, choose out of: perlin,"
#                              " flat, corridor, corridor_holes, tiles,"
#                              " triangles, domes, stairs, pipe, slant, corridor_various_width, "
#                              "pipe_variable_rad, corridor_turns, pillars_random, pillars_pseudorandom")
#     args = parser.parse_args()
#
#     print("CHOSEN : {}".format(args.terrain))
#
#     terrains = {'perlin' : [hf_gen.perlin],
#                 'flat' : [hf_gen.flat],
#                 'corridor' : [hf_gen.corridor],
#                 'corridor_holes' : [hf_gen.corridor_holes],
#                 'tiles' : [hf_gen.tiles],
#                 'triangles' : [hf_gen.triangles],
#                 'domes' : [hf_gen.domes],
#                 'stairs' : [hf_gen.stairs],
#                 'pipe' : [hf_gen.pipe],
#                 'slant' : [hf_gen.slant],
#                 'corridor_various_width' : [hf_gen.corridor_various_width],
#                 'pipe_variable_rad' : [hf_gen.pipe_variable_rad],
#                 'corridor_turns' : [hf_gen.corridor_turns],
#                 'pillars_random' : [hf_gen.pillars_random],
#                 'pillars_pseudorandom' : [hf_gen.pillars_pseudorandom]}
#
#     t = terrains[args.terrain]
#
#     hex = Hexapod(t, 1)
#     hex.demo()