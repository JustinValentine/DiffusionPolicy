from typing import Dict
import torch
import numpy as np
import json
import csv
import ast
from tqdm import tqdm
import zarr
import os
import shutil
import copy
from filelock import FileLock
from threadpoolctl import threadpool_limits
import concurrent.futures
import multiprocessing
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)
register_codecs()



class DoodleDataset(BaseDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            abs_action=False,
            use_legacy_normalizer=False,
            use_cache=False,
            seed=42,
            val_ratio=0.0
        ):

        replay_buffer = None
        if use_cache:
            cache_zarr_path = dataset_path + '.zarr.zip'
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # cache does not exists
                    try:
                        print('Cache does not exist. Creating!')
                        # store = zarr.DirectoryStore(cache_zarr_path)
                        replay_buffer = _convert_doodle_to_replay()
                        print('Saving cache to disk.')
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(
                                store=zip_store
                            )
                    except Exception as e:
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:
            replay_buffer = _convert_doodle_to_replay()

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
        
        # for key in rgb_keys:
        #     replay_buffer[key].compressor.numthreads=1

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.abs_action = abs_action
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_legacy_normalizer = use_legacy_normalizer

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer['action'])
        # if self.abs_action:
        #     if stat['mean'].shape[-1] > 10:
        #         # dual arm
        #         this_normalizer = robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)
        #     else:
        #         this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)
            
        #     if self.use_legacy_normalizer:
        #         this_normalizer = normalizer_from_stat(stat)
        # else:
        #     # already normalized
        #     this_normalizer = get_identity_normalizer_from_stat(stat)

        print(self.replay_buffer["action"].dtype)

        this_normalizer = get_range_normalizer_from_stat(stat)

        normalizer['action'] = this_normalizer

        # obs
        normalizer_obs = LinearNormalizer()
        print("low keys", self.lowdim_keys)
        for key in self.lowdim_keys:
            print(self.replay_buffer[key].shape)

            if key.endswith('pos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                # quaternion is in [-1,1] already
                # stat = array_to_stats(self.replay_buffer[key])
                this_normalizer = SingleFieldLinearNormalizer.create_identity()
            elif key.endswith('qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                this_normalizer = get_range_normalizer_from_stat(stat)
                print("Using default range normalize for key", key)
                # raise RuntimeError('unsupported')
            normalizer_obs[key] = this_normalizer

        normalizer['obs'] = normalizer_obs
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        # return len(self.sampler)
        return self.replay_buffer.n_episodes

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)

        index = self.replay_buffer.get_episode_slice(idx)

        data = dict()
        for key in list(self.replay_buffer.keys()):
            input_arr = self.replay_buffer[key][index]

            if key == 'class_quat':
                data[key] = np.tile(input_arr[0], (64, 1))
            elif key == 'on_paper_quat' or key == 'termination_quat':
                data[key] = np.full((64,) + input_arr.shape[1:], -1)
            else:
                data[key] = np.zeros((64,)+input_arr.shape[1:])
            
            data[key][:len(input_arr)] = input_arr


        # data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]


        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(data['action'].astype(np.float32))
        }

        torch_data = dict_apply(torch_data, lambda x: x.to(torch.float32))

        return torch_data


def _convert_doodle_to_replay():
    replay_buffer = ReplayBuffer.create_empty_numpy()

    with open('/home/odin/DiffusionPolicy/data/doodle/easy_class_index.json', 'r') as f:
        class_to_index = json.load(f)

    # Process the data
    max_trajectory_lenght = 64

    with open('/home/odin/DiffusionPolicy/data/doodle/easy_data_train.csv', 'r') as f:
        reader = csv.reader(f)
        i = 0
        for row in reader:
            class_name = row[0]
            trajectories_str = row[1]

            trajectories = ast.literal_eval(trajectories_str)

            if len(trajectories) > max_trajectory_lenght:
                continue

            # One-hot encode the class
            class_index = class_to_index[class_name]
            class_value = np.zeros(len(class_to_index.keys()), dtype=int).tolist()
            class_value[class_index] = 1

            class_quat = []
            on_paper_quat = []
            termination_quat = []
            action = []

            for point in trajectories:
                x, y, p1, p2 = point

                if p1 == 1:
                    on_paper_one_hot = 1
                else:
                    on_paper_one_hot = -1
                if p2 == 1:
                    termination = 1
                else:
                    termination = -1

                class_quat.append(class_value)
                on_paper_quat.append(on_paper_one_hot)
                termination_quat.append(termination)

                action_point = [x, y, on_paper_one_hot, termination]
                action.append(action_point)

            # Convert lists to NumPy arrays
            class_quat = np.array(class_quat, dtype=np.int8)
            on_paper_quat = np.array(on_paper_quat, dtype=np.int8)
            termination_quat = np.array(termination_quat, dtype=np.int8)
            action = np.array(action, dtype=float)

            data = {
                'class_quat': class_quat,
                'on_paper_quat': on_paper_quat,
                'termination_quat': termination_quat,
                'action': action
            }

            print(replay_buffer.n_episodes)
            replay_buffer.add_episode(data)
            i += 1

    print('fin')
    return replay_buffer

def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )
