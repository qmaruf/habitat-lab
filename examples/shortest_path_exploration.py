#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import habitat_sim
from PIL import Image
from habitat_sim.utils.common import d3_40_colors_rgb
import torchvision
import os
import shutil
import numpy as np
import habitat
from habitat.core.utils import try_cv2_import
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import colors
from myconfigs import category_map
from glob import glob
import argparse
import joblib
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-save_segmentation_data','--save_segmentation_data', help='save data for segmentaion training', action='store_true')
parser.add_argument('-save_video','--save_video', help='save video for segmentaion training', action='store_true')
args = vars(parser.parse_args())

cv2 = try_cv2_import()

device = "cuda"
IMAGE_DIR = os.path.join("examples", "images")
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

class SimpleRLEnv(habitat.RLEnv):
    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        return 0

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


from PIL import Image, ImageDraw
import cv2

font_scale = 1
font = cv2.FONT_HERSHEY_DUPLEX


def draw_top_down_map(info, heading, output_size):
    top_down_map = maps.colorize_topdown_map(
        info["top_down_map"]["map"], info["top_down_map"]["fog_of_war_mask"]
    )
    original_map_size = top_down_map.shape[:2]
    # print (original_map_size)
    
    map_scale = np.array(
        (1, original_map_size[1] * 1.0 / original_map_size[0])
    )
    new_map_size = np.round(output_size * map_scale).astype(np.int32)
    #OpenCV expects w, h but map size is in h, w
    top_down_map = cv2.resize(top_down_map, (new_map_size[1], new_map_size[0]))

    map_agent_pos = info["top_down_map"]["agent_map_coord"]
    #print (map_agent_pos)
    map_agent_pos = np.round(
        map_agent_pos * new_map_size / original_map_size
    ).astype(np.int32)
    #print (map_agent_pos)
    
    top_down_map = maps.draw_agent(
        top_down_map,
        map_agent_pos,
        heading - np.pi / 2,
        agent_radius_px=top_down_map.shape[0] / 40,
    )
    return top_down_map


def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene.id = settings["scene"]

    # Note: all sensors must have the same resolution
    sensors = {
        "color_sensor": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
        },
        "depth_sensor": {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
        },
        "semantic_sensor": {
            "sensor_type": habitat_sim.SensorType.SEMANTIC,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
        },
    }

    sensor_specs = []
    for sensor_uuid, sensor_params in sensors.items():
        print (sensor_uuid)
        if settings[sensor_uuid]:
            sensor_spec = habitat_sim.SensorSpec()
            sensor_spec.uuid = sensor_uuid
            sensor_spec.sensor_type = sensor_params["sensor_type"]
            sensor_spec.resolution = sensor_params["resolution"]
            sensor_spec.position = sensor_params["position"]

            sensor_specs.append(sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def get_scene(scene_path):
    sim_settings = {
        "width": 512*4,  # Spatial resolution of the observations
        "height": 512*4,
        "scene": scene_path,
        "default_agent": 0,
        "sensor_height": 1,  # Height of sensors in meters
        "color_sensor": True,  # RGB sensor
        "semantic_sensor": True,  # Semantic sensor
        "depth_sensor": False,  # Depth sensor
        "seed": 1,
    }

    cfg = make_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)
    scene = sim.semantic_scene    
    sim.close()    
    return scene

def semantic_to_rgba(semantic_obs):
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    return semantic_img


def get_floor_seg(im_rgb, im_sem, label_dict):
    im_seg = im_rgb.copy()
    im_seg[:] = 0      
    
    for label_id in label_dict:            
        label = label_dict[label_id]
        if label in ['floor', 'rug', 'mat']:
            im_seg[im_sem==label_id, :] = [153, 255, 255]
    return im_seg


def get_scene_key(dataset, config):
    if dataset == 'replica':
        skey = Path(config.SIMULATOR.SCENE).parts[-3]
    elif dataset == 'mp3d':
        skey = Path(config.SIMULATOR.SCENE).parts[-2]
    elif dataset == 'gibson':
        skey = Path(config.SIMULATOR.SCENE).stem
    else:
        raise Exception()
    return skey

def shortest_path_example(scene, data_path, dataset):

    sensor_size = 512
    images = []
    images_rgb, images_sem, images_map = [], [], []
    config = habitat.get_config(config_paths="configs/tasks/pointnav.yaml")
    config.defrost()    
    config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    config.TASK.SENSORS.append("HEADING_SENSOR")
    config.SIMULATOR.SCENE = scene
    config.DATASET.DATA_PATH = data_path
    config.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE=False
    config.SIMULATOR.RGB_SENSOR.HEIGHT = sensor_size
    config.SIMULATOR.RGB_SENSOR.WIDTH = sensor_size
    config.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = sensor_size
    config.SIMULATOR.SEMANTIC_SENSOR.WIDTH = sensor_size
    config.freeze()

    scene_obj = get_scene(scene)
    instance_id_to_label_name = dict()
    for obj in scene_obj.objects:
        if obj is not None:
            if obj.category is not None:
                instance_id_to_label_name[int(obj.id.split("_")[-1])] = obj.category.name()  

    # for key in instance_id_to_label_name:
    #     print (key, instance_id_to_label_name[key])  

    scene_key = get_scene_key(dataset, config)
    n_img = 0
    with SimpleRLEnv(config=config) as env:                   
        goal_radius = config.SIMULATOR.FORWARD_STEP_SIZE
        follower = ShortestPathFollower(env.habitat_env.sim, goal_radius, False)

        n = 10 # len(env.episodes)
        output_im_shape = None
        for iii in tqdm(range(n)):
            env.reset()
            dirname = os.path.join(IMAGE_DIR, "shortest_path_example", scene_key)
            # print("Agent stepping around inside environment.")            
            
            while not env.habitat_env.episode_over:
                best_action = follower.get_next_action(
                    env.habitat_env.current_episode.goals[0].position
                )
                if best_action is None:
                    break

                observations, reward, done, info = env.step(best_action)
                
                im_rgb = observations["rgb"]
                im_sem = semantic_to_rgba(observations["semantic"])
                
                top_down_map = draw_top_down_map(
                    info, observations["heading"][0], im_rgb.shape[0]
                )

                im_seg = get_floor_seg(observations["rgb"].copy(), observations["semantic"].copy(), instance_id_to_label_name)

                
                output_im = np.concatenate((im_rgb, im_sem, top_down_map, im_seg), axis=1)
                
                if output_im_shape is None:
                    output_im_shape = output_im.shape
                else:
                    if output_im.shape != output_im_shape:
                        output_im = cv2.resize(output_im, (output_im_shape[1], output_im_shape[0]))
                

                # print (output_im.shape)
                images.append(output_im)
                # print (im_seg.shape)
                # exit()
                if args['save_segmentation_data']:
                    seg_data = [observations["rgb"].copy(), im_seg[:, :, 0]]         
                    Path('/data/data/segmentation_data/%s'%scene_key).mkdir(parents=True, exist_ok=True)           
                    joblib.dump(seg_data, '/data/data/segmentation_data/%s/%03d.jlib'%(scene_key, n_img))
                    n_img += 1
        
        print (dirname)

        if args['save_video']:
            images_to_video(images, dirname, scene_key)

def get_scenes_data_paths(dataset):
    if dataset == 'replica':
        scenes = glob("/data/data/Replica-Dataset/dataset/*/habitat/mesh_semantic.ply")
        data_paths = ['/habitat-api/pointnavs/replica/%s.json.gz'%Path(s).parts[-3] for s in scenes]
    elif dataset == 'mp3d':
        scenes = glob('/data/data/matterport3d/v1/tasks/mp3d/*/*.glb')
        data_paths = ['/data/data/pointnavs/mp3d/%s.json.gz'%Path(s).stem for s in scenes]
    elif dataset == 'gibson':
        scenes = glob('/data/data/gibson/*.glb')
        data_paths = ['/habitat-api/pointnavs/gibson/%s.json.gz'%Path(s).stem for s in scenes]
    return scenes, data_paths

def main(dataset):    
    done = 0
    scenes, data_paths = get_scenes_data_paths(dataset)   
    for scene, data_path in zip(scenes, data_paths):
        if '2t7WUuJeko7' not in scene: continue
        print (scene, data_path)  
        if Path(scene).is_file() and Path(data_path).is_file():
            shortest_path_example(scene, data_path, dataset)
            done += 1
    
    print (done)
        
if __name__ == "__main__":
    main('mp3d')   