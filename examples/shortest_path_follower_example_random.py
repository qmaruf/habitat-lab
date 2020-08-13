#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


#/habitat-api/habitat/datasets/pointnav/data/datasets/pointnav/gibson/v1/all/content/

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
from habitat_sim.utils.data.data_extractor import ImageExtractor

np.random.seed(1)
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
    print (original_map_size)
    
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


config = habitat.get_config(config_paths="configs/tasks/pointnav.yaml")
config.defrost()
config.SIMULATOR.SCENE='/data/data/gibson_semantic/Allensville.glb'
config.DATASET.DATA_PATH = '/habitat-api/habitat/datasets/pointnav/data/datasets/pointnav/gibson/v1/all/content/Allensville.json.gz'
config.freeze()

print (config)
# exit()
extractor = ImageExtractor(
   	 config.SIMULATOR.SCENE,
	 labels=[0.0],
         img_size=(256, 256),
         output=["rgba", "depth", "semantic"],
         meters_per_pixel=0.1,
         shuffle=True)
poses = extractor.poses
print (len(poses))
#exit()
extractor.close()


def shortest_path_example(start_pose, end_pose):
    config.defrost()
    config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    config.TASK.SENSORS.append("HEADING_SENSOR")
    config.SIMULATOR.AGENT_0.IS_SET_START_STATE=True
    print (config.SIMULATOR.AGENT_0.START_POSITION, poses[0][0])
    config.SIMULATOR.AGENT_0.START_POSITION=poses[start_pose][0].tolist()
    config.freeze()
    print (config)
    

    print (config.SIMULATOR.SCENE)
    
    with SimpleRLEnv(config=config) as env:
        goal = poses[end_pose][0] #env.episodes[1500].goals[0].position
        goal_radius = None #env.episodes[0].goals[0].radius
        if goal_radius is None:
            goal_radius = config.SIMULATOR.FORWARD_STEP_SIZE
        follower = ShortestPathFollower(
            env.habitat_env.sim, goal_radius, False
        )

        print("Environment creation successful")
        step = 0
        n_pose=0

        for episode in range(1):
            done=False
            env.reset()      
            dirname = os.path.join(
                IMAGE_DIR, "shortest_path_example", "%02d" % episode
            )
            if os.path.exists(dirname):
                shutil.rmtree(dirname)
            os.makedirs(dirname)
            print("Agent stepping around inside environment.")
            images = []
            while not done: # env.habitat_env.episode_over:
                step += 1
                if step > 100: break
                best_action = follower.get_next_action(
                  # env.habitat_env.current_episode.goals[0].position
                  poses[end_pose][0]
                   
                )
                print ('best_action', best_action)
#                print (env.habitat_env.current_episode.goals[0])
        
                if best_action is None:
                    break

                observations, reward, done, info = env.step(best_action)
                print ('done', done)
                done=False
                env.reset()
                im = observations["rgb"]
                top_down_map = draw_top_down_map(
                    info, observations["heading"][0], im.shape[0]
                )

                output_im = np.concatenate((im, top_down_map), axis=1)
                images.append(output_im)
            images_to_video(images, dirname, "trajectory_%d"%episode)
            print("Episode finished")


def main():
    shortest_path_example(0, 10)


if __name__ == "__main__":
    main()
