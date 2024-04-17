import os
import glob
import ntpath
from collections import Counter

import numpy as np
import imageio


from vespa.package import ApplicationPackage, Field, HNSW, RankProfile, QueryTypeField
from vespa.application import Vespa
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage
import clip
from tenacity import retry, wait_exponential, stop_after_attempt


def translate_model_names_to_valid_vespa_field_names(model_name):
    return model_name.replace("/", "_").replace("-", "_").lower()


def sample_images(images, number_to_sample):
    """
    Sample equally spaced frames from a list of image frames

    :param images: a list of image frames.
    :param number_to_sample: int representing the number os frames to sample.

    :return: a numpy array containing the sample of frames.
    """
    if number_to_sample < len(images):
        idx = np.round(np.linspace(0, len(images) - 1, number_to_sample)).astype(int)
        return np.array(images)[idx]
    else:
        return np.array(images)


def extract_images(video_path, number_frames):
    """
    Extract equally spaced frames from a video.

    :param video_path: Full .mp4 video path.
    :param number_frames: Number of frames to sample.

    :return: a numpy array containing the sample of frames.
    """
    reader = imageio.get_reader(video_path, fps=1)
    frames = []
    for i, im in enumerate(reader):
        frames.append(im)
    return sample_images(frames, number_frames)


class VideoFeedDataset(Dataset):
    def __init__(self, video_dir, model_name, number_frames_per_video):
        """
        PyTorch Dataset to compute video embeddings and return pyvespa-compatible feed data.

        :param video_dir: Folder containing .mp4 video files.
        :param model_name: CLIP model name.
        :param number_frames_per_video: Number of embeddings per video.
        """
        self.video_dir = video_dir
        self.number_frames_per_video = number_frames_per_video
        valid_vespa_model_name = translate_model_names_to_valid_vespa_field_names(
            model_name
        )
        self.from_tensor_to_PIL = ToPILImage()
        self.model, self.preprocess = clip.load(model_name)
        self.video_file_names = glob.glob(os.path.join(video_dir, "*.mp4"))
        self.image_embedding_name = valid_vespa_model_name + "_image"

    def _from_image_to_vector(self, x):
        """
        From image to embedding.

        :param x: PIL images
        :return: normalized image embeddings.
        """
        with torch.no_grad():
            image_features = self.model.encode_image(
                self.preprocess(x).unsqueeze(0)
            ).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    def __len__(self):
        return len(self.video_file_names)

    def __getitem__(self, idx):
        video_file_name = self.video_file_names[idx]
        images = extract_images(
            video_path=video_file_name, number_frames=self.number_frames_per_video
        )
        pil_images = [self.from_tensor_to_PIL(x) for x in images]
        frames = []
        for idx, image in enumerate(pil_images):
            image = self._from_image_to_vector(image)
            video_base_name = ntpath.basename(video_file_name)
            frames.append(
                {
                    "id": video_base_name.split(".mp4")[0] + "_{}".format(idx),
                    "fields": {
                        "video_file_name": video_base_name,
                        self.image_embedding_name: {"values": image.tolist()[0]},
                    },
                    "create": True,
                }
            )
        return frames


@retry(wait=wait_exponential(multiplier=1), stop=stop_after_attempt(3))
def send_video_embeddings(app, batch):
    """
    Send pyvespa-compatible batch to Vespa app.

    :param app: pyvespa connection to a Vespa instance
    :param batch: pyvespa-compatible list of data points to be updated.
    :return: None
    """
    responses = app.update_batch(batch=batch)
    status_code_summary = Counter([x.status_code for x in responses])
    if status_code_summary[200] != len(batch):
        print([response.json for response in responses if response.status_code != 200])
        raise ValueError("Failed to send data.")
    print("Successfully sent {} data points.".format(status_code_summary[200]))


def compute_and_send_video_embeddings(
    app, batch_size, clip_model_names, number_frames_per_video, video_dir, num_workers=0
):
    """
    Loop through video folder, compute embeddings and send to Vespa app.

    :param app: pyvespa connection to a Vespa instance
    :param batch_size: Number of images to process per iteration.
    :param clip_model_names: CLIP models names. It will generate one image embedding per model name.
    :param number_frames_per_video: Number of frames to use per video.
    :param video_dir: Complete path of the folder containing .mp4 video files.
    :param num_workers: Number of workers to use (refers to the DataLoader parallelization)
    :return: None
    """
    for model_name in clip_model_names:
        video_dataset = VideoFeedDataset(
            video_dir=video_dir,  # Folder containing image files
            model_name=model_name,  # CLIP model name used to convert image into vector
            number_frames_per_video=number_frames_per_video,
        )
        dataloader = DataLoader(
            video_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: [item for sublist in x for item in sublist],
            num_workers=num_workers,
        )
        for idx, batch in enumerate(dataloader):
            print(
                "Model name: {}. Iteration: {}/{}".format(
                    model_name, idx, len(dataloader)
                )
            )
            send_video_embeddings(app=app, batch=batch)


class TextProcessor(object):
    def __init__(self, model_name):
        """
        Python-based text processor.

        :param model_name: CLIP model name to use embedding text.
        """
        self.model, _ = clip.load(model_name)
        self.model_name = model_name

    def embed(self, text):
        """
        Convert text to (normalized) embedding

        :param text: a string to be embedded.
        :return: Normalized embedding vector.
        """
        text_tokens = clip.tokenize(text)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.tolist()[0]


def create_text_video_app(model_info):
    """
    Create text to video search app based on a variety of CLIP models

    :param model_info: dict containing model names as keys and embedding size as values.
        Check `clip.available_models()` to check which models are available.

    :return: A Vespa application package.
    """
    app_package = ApplicationPackage(name="videosearch")

    app_package.schema.add_fields(
        Field(name="video_file_name", type="string", indexing=["summary", "attribute"]),
    )
    for model_name, embedding_size in model_info.items():
        model_name = translate_model_names_to_valid_vespa_field_names(model_name)
        app_package.schema.add_fields(
            Field(
                name=model_name + "_image",
                type="tensor<float>(x[{}])".format(embedding_size),
                indexing=["attribute", "index"],
                ann=HNSW(
                    distance_metric="euclidean",
                    max_links_per_node=16,
                    neighbors_to_explore_at_insert=500,
                ),
            )
        )
        app_package.schema.add_rank_profile(
            RankProfile(
                name=model_name + "_similarity",
                inherits="default",
                first_phase="closeness({})".format(model_name + "_image"),
            )
        )
        app_package.query_profile_type.add_fields(
            QueryTypeField(
                name="ranking.features.query({})".format(model_name + "_text"),
                type="tensor<float>(x[{}])".format(embedding_size),
            )
        )
    return app_package


def create_vespa_query(query, text_processor, number_videos):
    """
    Create the body of a Vespa query.

    :param query: a string representing the query.
    :param text_processor: an instance of `TextProcessor` to convert string to embedding.
    :param number_videos: Number of videos to return.
    :return: body of a Vespa query request.
    """
    valid_vespa_model_name = translate_model_names_to_valid_vespa_field_names(
        text_processor.model_name
    )
    image_field_name = valid_vespa_model_name + "_image"
    text_field_name = valid_vespa_model_name + "_text"
    ranking_name = valid_vespa_model_name + "_similarity"

    return {
        "yql": 'select * from sources * where ({{"targetNumHits":100}}nearestNeighbor({},{})) | all(group(video_file_name) max({}) order(-max(relevance())) each( max(1) each(output(summary())) as(frame)) as(video))'.format(
            image_field_name, text_field_name, number_videos
        ),
        "hits": 0,
        "ranking.features.query({})".format(text_field_name): text_processor.embed(
            query
        ),
        "ranking.profile": ranking_name,
        "timeout": 10,
    }


def search_video_file_names(app, query, text_processor, number_videos):
    """
    Parse the output of the Vespa query.

    Parse the output of the Vespa query to return a list with the video file name and
    relevance score for each hit.

    :param app: The pyvespa Vespa connection to the app.
    :param query: The text query to be sent.
    :param text_processor: An instance of the TextProcessor to turn text into embedding.
    :param number_videos: The number of videos to be retrieved.
    :return: a list with the video file name and relevance score for each hit.
    """

    result = app.query(
        body=create_vespa_query(
            query=query, text_processor=text_processor, number_videos=number_videos
        )
    )
    parsed_results = [
        {
            "video_file_name": video["children"][0]["children"][0]["fields"][
                "video_file_name"
            ],
            "relevance": video["children"][0]["children"][0]["relevance"],
        }
        for video in result.json["root"]["children"][0]["children"][0]["children"]
    ]
    return parsed_results


class VideoSearchApp(object):
    def __init__(self, app: Vespa, clip_model_name=None, text_processor=None):
        """
        Video search app with custom query for video retrieval.

        :param app: The pyvespa Vespa connection to the app.
        :param clip_model_name: CLIP model name to turn text into embedding
        :param text_processor: TextProcessor instance. `clip_model_name` will
            be ignored if an instance is provided.
        """
        if text_processor:
            self.text_processor = text_processor
        elif clip_model_name:
            self.text_processor = TextProcessor(clip_model_name)
        else:
            ValueError("Provide a clip_model_name or an instance of TextProcessor")
        self.app = app

    def query(self, text, number_videos):
        """
        Video search

        :param text: Text query describing an action.
        :param number_videos: Number of videos to retrieve.
        :return: a list with the video file name and relevance score for each hit.
        """
        return search_video_file_names(
            app=self.app,
            query=text,
            text_processor=self.text_processor,
            number_videos=number_videos,
        )
