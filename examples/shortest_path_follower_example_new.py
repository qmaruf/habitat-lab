#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from PIL import Image
from habitat_sim.utils.common import d3_40_colors_rgb

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
cv2 = try_cv2_import()

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




def semantic_to_rgba(semantic_obs):
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    return semantic_img

images = []

def shortest_path_example(episode_id):
    config = habitat.get_config(config_paths="configs/tasks/pointnav.yaml")
    config.defrost()    
    config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    config.TASK.SENSORS.append("HEADING_SENSOR")
    config.SIMULATOR.SCENE='/data/data/gibson_semantic/Allensville.glb'
    config.DATASET.DATA_PATH='/habitat-api/pointnavs/Allensville.json.gz'
    config.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE=False
    config.freeze()
    
    with SimpleRLEnv(config=config) as env:                   
        goal_radius = config.SIMULATOR.FORWARD_STEP_SIZE
        follower = ShortestPathFollower(env.habitat_env.sim, goal_radius, False)
        
        for iii in range(len(env.episodes)):
            env.reset()
            print (env.current_episode)
            # continue
            
            # exit()

            print ('-------------'*10)
            print ('-------------'*10)
            print ('-------------'*10)
            print (env.habitat_env.current_episode)
            print ('-------------'*10)
            print ('-------------'*10)
            print ('-------------'*10)
            # raise Exception()
            
            dirname = os.path.join(IMAGE_DIR, "shortest_path_example", "dir_%d"%iii)
            
            if os.path.exists(dirname):
                shutil.rmtree(dirname)
            os.makedirs(dirname)
            
            print("Agent stepping around inside environment.")
            
            
            while not env.habitat_env.episode_over:
                best_action = follower.get_next_action(
                    env.habitat_env.current_episode.goals[0].position
                )
                if best_action is None:
                    break

                observations, reward, done, info = env.step(best_action)
                # print (observations.keys())
                # exit()
                im_rgb = observations["rgb"]
                im_sem = semantic_to_rgba(observations["semantic"])
                
                top_down_map = draw_top_down_map(
                    info, observations["heading"][0], im_rgb.shape[0]
                )

                # print (im_rgb.shape, im_sem.shape)
                # exit()
                output_im = np.concatenate((im_rgb, im_sem, top_down_map), axis=1)
                # output_im = np.concatenate((output_im, top_down_map), axis=1)
                images.append(output_im)
            
            print('*********************************************************************************************', iii)
        print (dirname)
        images_to_video(images, dirname, "trajectory_%d"%(iii))
        print("Episode finished")

















# def shortest_path_example(episode_id):
#     config = habitat.get_config(config_paths="configs/tasks/pointnav.yaml")
#     config.defrost()    
#     config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
#     config.TASK.SENSORS.append("HEADING_SENSOR")
#     config.SIMULATOR.SCENE='/data/data/gibson_semantic/Allensville.glb'
#     config.DATASET.DATA_PATH='/habitat-api/pointnavs/Allensville.json.gz'
#     config.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE=False
#     config.freeze()
    
#     with SimpleRLEnv(config=config) as env:           
        
#         goal = env.episodes[episode_id].goals[0].position  
#         print ('*'*50, goal)
#         # return              
#         goal_radius = env.episodes[episode_id].goals[0].radius        
#         if goal_radius is None:
#             goal_radius = config.SIMULATOR.FORWARD_STEP_SIZE
#         follower = ShortestPathFollower(env.habitat_env.sim, goal_radius, False)
        
#         env.reset()

#         print ('-------------'*10)
#         print ('-------------'*10)
#         print ('-------------'*10)
#         print (env.habitat_env.current_episode)
#         print ('-------------'*10)
#         print ('-------------'*10)
#         print ('-------------'*10)
#         raise Exception()
        
#         dirname = os.path.join(IMAGE_DIR, "shortest_path_example", "%02d" % episode_id)
        
#         if os.path.exists(dirname):
#             shutil.rmtree(dirname)
#         os.makedirs(dirname)
        
#         print("Agent stepping around inside environment.")
        
#         images = []
#         while not env.habitat_env.episode_over:
#             best_action = follower.get_next_action(
#                 env.habitat_env.current_episode.goals[0].position
#             )
#             if best_action is None:
#                 break

#             observations, reward, done, info = env.step(best_action)
#             im = observations["rgb"]
#             top_down_map = draw_top_down_map(
#                 info, observations["heading"][0], im.shape[0]
#             )
#             output_im = np.concatenate((im, top_down_map), axis=1)
#             images.append(output_im)
        
#         images_to_video(images, dirname, "trajectory_%d"%(episode_id))
#         print("Episode finished")


def main():
    shortest_path_example(0)
    # shortest_path_example(1)
    # shortest_path_example(2)


if __name__ == "__main__":
    main()
