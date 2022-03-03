import unittest
from exploration_app import (
    create_weakAND_operator,
    create_ANN_operator,
    create_yql,
)


class TestYQL(unittest.TestCase):
    def test_create_weakAND_operator(self):
        wand_operator = create_weakAND_operator(query=" this is a query")
        self.assertEqual(
            wand_operator,
            '([{"targetNumHits": 1000}]weakAnd(default contains "this", default contains "is", '
            'default contains "a", default contains "query"))',
        )

    def test_create_ANN_operator(self):
        self.assertEqual(
            create_ANN_operator(
                ann_operator="title", embedding="word2vec", target_hits=1000
            ),
            '([{"targetNumHits": 1000, "label": "nns"}]nearestNeighbor(title_word2vec, tensor))',
        )
        self.assertEqual(
            create_ANN_operator(ann_operator="body", embedding="bert", target_hits=800),
            '([{"targetNumHits": 800, "label": "nns"}]nearestNeighbor(body_bert, tensor_bert))',
        )
        self.assertEqual(
            create_ANN_operator(
                ann_operator="title_body", embedding="gse", target_hits=1000
            ),
            '([{"targetNumHits": 1000, "label": "nns"}]nearestNeighbor(title_gse, tensor_gse)) or '
            '([{"targetNumHits": 1000, "label": "nns"}]nearestNeighbor(body_gse, tensor_gse))',
        )
        self.assertIsNone(
            create_ANN_operator(ann_operator=None, embedding="gse", target_hits=1000)
        )
        with self.assertRaises(ValueError):
            create_ANN_operator(
                ann_operator="invalid", embedding="gse", target_hits=1000
            )

    def test_create_yql(self):
        query = " this is a query"
        embedding = "gse"
        grammar_operator = "weakAND"
        ann_operator = "title_body"
        self.assertEqual(
            create_yql(query, grammar_operator, ann_operator, embedding),
            'select * from sources * where ({"targetNumHits": 1000}weakAnd(default contains "this", '
            'default contains "is", default contains "a", default contains "query")) or '
            '({"targetNumHits": 1000, "label": "nns"}nearestNeighbor(title_gse, tensor_gse)) or '
            '({"targetNumHits": 1000, "label": "nns"}nearestNeighbor(body_gse, tensor_gse))',
        )
        # create_yql(query, None, None, embedding)
