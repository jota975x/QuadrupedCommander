# example_dataset_dataset_builder.py
from typing import Iterator, Tuple, Any
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

class CustomDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for single-step episodes with image, text, and (x,y) label."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.'
    }

    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(960, 1280, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation (960x1280x3).'
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action, consists of [7x joint velocities, '
                            '2x gripper velocities, 1x terminate episode].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount factor, set to 1.0 since no RL sequence.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward, set to 0.0 for non-RL data.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True since only one step per episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True since only one step per episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True since only one step per episode.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    )
                }),
            })
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        # If you have separate val/test splits, you can also add them here.
        return {
            'train': self._generate_examples(path='data/train/episode_*.npy'),
            # 'val': self._generate_examples(path='data/val/episode_*.npy'),
        }

    # def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
    #     episode_paths = glob.glob(path)
        
    #     for episode_path in episode_paths:
    #         data = np.load(episode_path, allow_pickle=True)
    #         # data is a single-step episode: a list with one dict: {'image', 'text', 'label'}

    #         step = data[0]
    #         episode = [{
    #             'observation': {
    #                 'image': step['image'],
    #                 'text': step['text'],
    #                 'label': step['label'],
    #             },
    #             'discount': 1.0,
    #             'reward': 0.0,
    #             'is_first': True,
    #             'is_last': True,
    #             'is_terminal': True,

    #         }]

    #         sample = {
    #             'steps': episode,
    #             'episode_metadata': {
    #                 'file_path': episode_path
    #             }
    #         }
            
    #         yield episode_path, sample

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            data = np.load(episode_path, allow_pickle=True)     # this is a list of dicts in our case

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i, step in enumerate(data):
                # compute Kona language embedding
                language_embedding = self._embed([step['language_instruction']])[0].numpy()

                episode.append({
                    'observation': {
                        'image': step['image'],
                        # 'text': step['text'],
                        # 'label': step['label'],
                    },
                    'action': step['label'],
                    'discount': 1.0,
                    'reward': 1,
                    'is_first': True,
                    'is_last': True,
                    'is_terminal': True,
                    'language_instruction': step['language_instruction'],
                    'language_embedding': language_embedding,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = glob.glob(path)

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )


