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

cv2 = try_cv2_import()

device = "cuda"
od_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
od_model.eval()

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

def draw_rect(img, box, img_src, outline="red", label=None):
    x1, y1, x2, y2 = map(int, box)   
    # if label 
    if img_src == 'pil':
        draw = ImageDraw.Draw(img)
        draw.rectangle(((x1, y1), (x2, y2)), width=5, outline=outline)
        del draw
    elif img_src == 'cv2':  
        # print ('label =============== ', label)
        if label is not None:
            img = cv2.putText(img, label, (x1, y1-10), font, fontScale=font_scale, color=(0, 0, 255), thickness=1)      
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)        
    return img

def detect_object(img):    
    image_tensor = torchvision.transforms.functional.to_tensor(img)
    prediction = od_model([image_tensor.to(device)])[0]
    img = np.array(img)
    for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
        x1, y1, x2, y2 = box
        if score >= 0.5:
            img = draw_rect(img, [x1, y1, x2, y2], 'cv2', 'red', category_map[label.item()])
    # img = Image.fromarray(img)
    return img
    


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
        "width": 512,  # Spatial resolution of the observations
        "height": 512,
        "scene": scene_path,
        "default_agent": 0,
        "sensor_height": 1,  # Height of sensors in meters
        "color_sensor": True,  # RGB sensor
        "semantic_sensor": True,  # Semantic sensor
        "depth_sensor": True,  # Depth sensor
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


def get_bbox(im_rgb, im_sem, label_dict):
    def get_bbox(sobs, iid):
        # print (sobs.shape, iid)
        # exit()
        X, Y = np.where(sobs==iid)
        if len(X) and len(Y):
            return min(Y), min(X), max(Y), max(X)
        else:
            return -1, -1, -1, -1

    im_bbox = im_rgb.copy()
    im_sem = np.asarray(im_sem)
    for label_id in label_dict:
        x1, y1, x2, y2 = get_bbox(im_sem, label_id)
        if x1 != -1:
            label = label_dict[label_id]
            if label == 'floor':
                im_bbox[im_sem==label_id, :] = [50,205,50]
            #     print (im_bbox.shape, im_sem.shape)
            #     exit()
            # if label in ['wall', 'ceiling', 'handrail', 'stair', 'floor']:
            #     print (label)
            #     label = None
            # im_bbox = draw_rect(im_bbox, [x1, y1, x2, y2], 'cv2', "green", label)
    
    return im_bbox



def shortest_path_example(scene, data_path, dataset):
    images = []
    images_rgb, images_sem, images_map = [], [], []
    config = habitat.get_config(config_paths="configs/tasks/pointnav.yaml")
    config.defrost()    
    config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    config.TASK.SENSORS.append("HEADING_SENSOR")
    # config.SIMULATOR.SCENE='/data/data/gibson_semantic/Allensville.glb'
    # config.DATASET.DATA_PATH='/habitat-api/pointnavs/Allensville.json.gz'

    config.SIMULATOR.SCENE = scene # '/data/data/matterport3d/v1/tasks/mp3d/gTV8FGcVJC9/gTV8FGcVJC9.glb'
    config.DATASET.DATA_PATH = data_path # '/habitat-api/pointnavs/gTV8FGcVJC9.json.gz'
    config.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE=False
    config.freeze()

    scene_obj = get_scene(scene)


    instance_id_to_label_name = dict()
    for obj in scene_obj.objects:
        if obj is not None:
            if obj.category is not None:
                instance_id_to_label_name[int(obj.id.split("_")[-1])] = obj.category.name()    

    

    if dataset == 'replica':
        out_vid = Path(config.SIMULATOR.SCENE).parts[-3]
    elif dataset == 'mp3d':
        out_vid = Path(config.SIMULATOR.SCENE).parts[-2]
    elif dataset == 'gibson':
        out_vid = Path(config.SIMULATOR.SCENE).stem
    else:
        raise Exception()
    
    with SimpleRLEnv(config=config) as env:                   
        goal_radius = config.SIMULATOR.FORWARD_STEP_SIZE
        follower = ShortestPathFollower(env.habitat_env.sim, goal_radius, False)

        
        
        n = 10 # len(env.episodes)
        for iii in range(n):
            env.reset()

            print (dir(env.habitat_env.sim.sensor_suite))
            # exit()
        

            print ('-------------'*10)
            print ('-------------'*10)
            print ('-------------'*10)
            print (env.habitat_env.current_episode)
            print ('-------------'*10)
            print ('-------------'*10)
            print ('-------------'*10)
        
            
            dirname = os.path.join(IMAGE_DIR, "shortest_path_example", out_vid)
        
            
            print("Agent stepping around inside environment.")
            
            
            while not env.habitat_env.episode_over:
                best_action = follower.get_next_action(
                    env.habitat_env.current_episode.goals[0].position
                )
                if best_action is None:
                    break

                observations, reward, done, info = env.step(best_action)
                
                im_rgb = observations["rgb"]
                im_rgb = detect_object(im_rgb)

                im_sem = semantic_to_rgba(observations["semantic"])
                
                top_down_map = draw_top_down_map(
                    info, observations["heading"][0], im_rgb.shape[0]
                )

                im_bbox = get_bbox(observations["rgb"].copy(), observations["semantic"].copy(), instance_id_to_label_name)
                output_im = np.concatenate((im_rgb, im_sem, top_down_map, im_bbox), axis=1)
                # plt.figure()
                # plt.imshow(output_im)
                # plt.savefig('tmp.jpg')
                # exit()
                # images_rgb.append(im_rgb)
                # images_sem.append(im_sem)
                # images_map.append(top_down_map)

                
        
                images.append(output_im)
        
            
        
        print (dirname)

        
        images_to_video(images, dirname, out_vid)

        




from glob import glob


def get_scenes_data_paths(dataset):
    if dataset == 'replica':
        scenes = glob("/data/data/Replica-Dataset/dataset/*/habitat/mesh_semantic.ply")
        data_paths = ['/habitat-api/pointnavs/replica/%s.json.gz'%Path(s).parts[-3] for s in scenes]
    elif dataset == 'mp3d':
        scenes = glob('/data/data/matterport3d/v1/tasks/mp3d/*/*.glb')
        data_paths = ['/habitat-api/pointnavs/%s.json.gz'%Path(s).stem for s in scenes]
    elif dataset == 'gibson':
        scenes = glob('/data/data/gibson/*.glb')
        data_paths = ['/habitat-api/pointnavs/gibson/%s.json.gz'%Path(s).stem for s in scenes]
    return scenes, data_paths

def main(dataset):    
    scenes, data_paths = get_scenes_data_paths(dataset)   
    for scene, data_path in zip(scenes, data_paths):
        print (scene, data_path)  
        if Path(scene).is_file() and Path(data_path).is_file():
            shortest_path_example(scene, data_path, dataset)
            # break
        
if __name__ == "__main__":
    # main('gibson')
    main('mp3d')
    main('replica')