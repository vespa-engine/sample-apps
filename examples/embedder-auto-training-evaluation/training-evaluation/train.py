from sentence_transformers import SentenceTransformer
from sentence_transformers import models, losses, InputExample
from torch.utils.data import Dataset, DataLoader
import json
import argparse


class TrainDataset(Dataset):
    def __init__(self, documents: str, queries: str):
        self.documents = {}
        with open(documents, "r") as file:
            for line in file.readlines():
                document = json.loads(line)
                self.documents[document["doc_id"]] = document

        self.queries = {}
        with open(queries, "r") as file:
            for line in file.readlines():
                query = json.loads(line)
                self.queries[query["query_id"]] = query
        self.queries_ids = list(self.queries.keys())

    def __getitem__(self, item):
        query = self.queries[self.queries_ids[item]]
        query_text = query["query"]

        pos_id = query["positives"].pop(0)
        pos_doc = self.documents[pos_id]
        query["positives"].append(pos_id)

        neg_id = query["negatives"].pop(0)
        neg_doc = self.documents[neg_id]
        query["negatives"].append(neg_id)

        return InputExample(
            texts=[
                {"query": query_text},
                {"doc": pos_doc["text"]},
                {"doc": neg_doc["text"]},
            ]
        )

    def __len__(self):
        return len(self.queries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--documents", type=str, required=True)
    parser.add_argument("--queries", type=str, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    model = SentenceTransformer(args.model)

    train_dataset = TrainDataset(args.documents, args.queries)
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=8, drop_last=True
    )
    train_loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=1000,
        use_amp=True,
        checkpoint_path=args.output_dir,
        checkpoint_save_steps=len(train_dataloader),
        checkpoint_save_total_limit=2,
        optimizer_params={"lr": 2e-5},
    )
    model.save(args.output_dir)


if __name__ == "__main__":
    main()
