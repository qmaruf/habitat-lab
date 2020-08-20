import glob
import gzip
import json
import multiprocessing
import os
import os.path as osp
from pathlib import Path
import tqdm

import habitat
import habitat_sim
from habitat.datasets.pointnav.pointnav_generator import generate_pointnav_episode

num_episodes_per_scene = int(1e2)


def _generate_fn(scene):
    cfg = habitat.get_config()
    cfg.defrost()
    cfg.SIMULATOR.SCENE = scene
    cfg.SIMULATOR.AGENT_0.SENSORS = []
    cfg.freeze()

    sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)

    dset = habitat.datasets.make_dataset("PointNav-v1")
    dset.episodes = list(
        generate_pointnav_episode(
            sim, num_episodes_per_scene, is_gen_shortest_path=False
        )
    )
    for ep in dset.episodes:        
        ep.scene_id = ep.scene_id        

    if dataset == 'replica':
        scene_key = Path(scene).parts[-3]
        out_file = f"/habitat-api/pointnavs/replica/{scene_key}.json.gz"
    elif dataset == 'gibson':
        scene_key = Path(scene).stem
        out_file = f"/habitat-api/pointnavs/gibson/{scene_key}.json.gz"
    elif dataset == 'mp3d':
        raise Exception()
        scene_key = Path(scene).stem
        out_file = f"/habitat-api/pointnavs/{scene_key}.json.gz"

    print ('out_file', out_file)


    os.makedirs(osp.dirname(out_file), exist_ok=True)
    with gzip.open(out_file, "wt") as f:
        f.write(dset.to_json())

# scenes = glob.glob("./data/scene_datasets/gibson/*.glb")
# scenes = glob.glob("/data/data/gibson_semantic/*.glb")


dataset = 'gibson'

if dataset == 'mp3d':
    scenes = glob.glob("/data/data/matterport3d/v1/tasks/mp3d/*/*.glb")
elif dataset == 'replica':
    scenes = glob.glob("/data/data/Replica-Dataset/dataset/*/habitat/mesh_semantic.ply")
elif dataset == 'gibson':
    scenes = glob.glob("/data/data/gibson_semantic/*_semantic.ply")
else:
    raise Exception()
print (scenes)

for scene in scenes:
    print (scene)
    _generate_fn(scene)

print (scenes)
