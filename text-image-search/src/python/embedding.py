import os
import glob
import ntpath
from collections import Counter

from vespa.package import ApplicationPackage, Field, HNSW, RankProfile, QueryTypeField
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import clip
from tenacity import retry, wait_exponential, stop_after_attempt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def translate_model_names_to_valid_vespa_field_names(model_name):
    return model_name.replace("/", "_").replace("-", "_").lower()


class ImageFeedDataset(Dataset):
    def __init__(self, img_dir, model_name):
        """
        PyTorch Dataset to compute image embeddings and return pyvespa-compatible feed data.

        :param img_dir: Folder containing image files.
        :param model_name: CLIP model name.
        """
        valid_vespa_model_name = translate_model_names_to_valid_vespa_field_names(
            model_name
        )
        self.model, self.preprocess = clip.load(model_name)
        self.img_dir = img_dir
        self.image_file_names = glob.glob(os.path.join(img_dir, "*.jpg"))
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
        return len(self.image_file_names)

    def __getitem__(self, idx):
        image_file_name = self.image_file_names[idx]
        image = Image.open(image_file_name)
        image = self._from_image_to_vector(image)
        image_base_name = ntpath.basename(image_file_name)
        return {
            "id": image_base_name.split(".jpg")[0],
            "fields": {
                "image_file_name": image_base_name,
                self.image_embedding_name: {"values": image.tolist()[0]},
            },
            "create": True,
        }


@retry(wait=wait_exponential(multiplier=1), stop=stop_after_attempt(3))
def send_image_embeddings(app, batch, schema=None):
    """
    Send pyvespa-compatible batch to Vespa app.

    :param app: pyvespa connection to a Vespa instance
    :param batch: pyvespa-compatible list of data points to be updated.
    :return: None
    """
    responses = app.update_batch(batch=batch, schema=schema)
    status_code_summary = Counter([x.status_code for x in responses])
    if status_code_summary[200] != len(batch):
        print([response.json for response in responses if response.status_code != 200])
        raise ValueError("Failed to send data.")
    print("Successfully sent {} data points.".format(status_code_summary[200]))


def compute_and_send_image_embeddings(app, batch_size, clip_model_names, num_workers=0, schema=None):
    """
    Loop through image folder, compute embeddings and send to Vespa app.

    :param app: pyvespa connection to a Vespa instance
    :param batch_size: Number of images to process per iteration.
    :param clip_model_names: CLIP models names. It will generate one image embedding per model name.
    :param num_workers: Number of workers to use (refers to the DataLoader parallelization)
    :return: None
    """
    for model_name in clip_model_names:
        image_dataset = ImageFeedDataset(
            img_dir=os.environ["IMG_DIR"],  # Folder containing image files
            model_name=model_name,  # CLIP model name used to convert image into vector
        )
        dataloader = DataLoader(
            image_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: x,
            num_workers=num_workers,
        )
        for idx, batch in enumerate(dataloader):
            print(
                "Model name: {}. Iteration: {}/{}".format(
                    model_name, idx, len(dataloader)
                )
            )
            send_image_embeddings(app=app, batch=batch, schema=schema)


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


def create_text_image_app(model_info):
    """
    Create text to image search app based on a variety of CLIP models

    :param model_info: dict containing model names as keys and embedding size as values.
        Check `clip.available_models()` to check which models are available.

    :return: A Vespa application package.

    """
    app_package = ApplicationPackage(name="image_search")

    app_package.schema.add_fields(
        Field(name="image_file_name", type="string", indexing=["summary", "attribute"]),
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
                    neighbors_to_explore_at_insert=200,
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


def create_vespa_query(query, text_processor):
    """
    Create the body of a Vespa query.

    :param query: a string representing the query.
    :param text_processor: an instance of `TextProcessor` to convert string to embedding.
    :return: body of a Vespa query request.
    """
    valid_vespa_model_name = translate_model_names_to_valid_vespa_field_names(
        text_processor.model_name
    )
    image_field_name = valid_vespa_model_name + "_image"
    text_field_name = valid_vespa_model_name + "_text"
    ranking_name = valid_vespa_model_name + "_similarity"

    return {
        "yql": 'select * from sources * where ({{"targetNumHits":100}}nearestNeighbor({},{}))'.format(
            image_field_name, text_field_name
        ),
        "hits": 100,
        "ranking.features.query({})".format(text_field_name): text_processor.embed(
            query
        ),
        "ranking.profile": ranking_name,
        "timeout": 10,
    }


def create_vespa_query_body_function(model_name):
    """
    Create a function that take string as input and returns the body of a vespa query request as output.

    :param model_name: CLIP model name that will be used to map string to embedding.
    :return: a function take string as input and returns the body of a vespa query request as output.
    """
    text_processor = TextProcessor(model_name=model_name)
    return lambda x: create_vespa_query(x, text_processor=text_processor)


def plot_images(query_result, relative_image_folder):
    """
    Plot images from query results

    :param query_result: Output from app.query
    :param relative_image_folder: folder containing image files.
    :return: plot images.
    """
    image_file_names = [
        hit["fields"]["image_file_name"] for hit in query_result.hits[:4]
    ]
    fig = plt.figure(figsize=(10, 10))
    for idx, image_file_name in enumerate(image_file_names):
        sub = fig.add_subplot(2, 2, idx + 1)
        _ = plt.imshow(
            mpimg.imread(os.path.join(relative_image_folder, image_file_name))
        )
        sub.set_title("Rank: {}".format(idx))
    plt.tight_layout()
