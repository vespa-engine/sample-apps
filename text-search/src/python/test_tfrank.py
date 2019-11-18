import unittest
from unittest.mock import patch
from numpy.testing import assert_array_equal
import pandas as pd

from tfrank import data_generator


class TestDataGenerator(unittest.TestCase):
    @patch("tfrank.sample")
    def test_data_generator(self, mock_sample):
        mock_sample.return_value = [1, 2]
        dataset = pd.DataFrame(
            {
                "qid": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                "docid": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                "relevant": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                "f1": [11, 12, 13, 14, 15, 16, 17, 18, 19],
                "f2": [21, 22, 23, 24, 25, 26, 27, 28, 29],
            }
        )
        data_gen = data_generator(
            dataset=dataset,
            features=["f1", "f2"],
            label="relevant",
            queries={1, 2},
            num_docs=3,
            batch_size=1,
            num_epochs=2,
        )

        def train_input_fn():
            return next(data_gen)

        x, y = train_input_fn()
        x = x["x_raw"]
        assert_array_equal(x, [[[11, 21], [12, 22], [13, 23]]])
        self.assertEqual(x.shape, (1, 3, 2))
        assert_array_equal(y, [[1, 0, 0]])
        self.assertEqual(y.shape, (1, 3))

        x, y = train_input_fn()
        x = x["x_raw"]
        assert_array_equal(x, [[[14, 24], [15, 25], [16, 26]]])
        self.assertEqual(x.shape, (1, 3, 2))
        assert_array_equal(y, [[0, 1, 0]])
        self.assertEqual(y.shape, (1, 3))

        x, y = train_input_fn()
        x = x["x_raw"]
        assert_array_equal(x, [[[11, 21], [12, 22], [13, 23]]])
        self.assertEqual(x.shape, (1, 3, 2))
        assert_array_equal(y, [[1, 0, 0]])
        self.assertEqual(y.shape, (1, 3))

        x, y = train_input_fn()
        x = x["x_raw"]
        assert_array_equal(x, [[[14, 24], [15, 25], [16, 26]]])
        self.assertEqual(x.shape, (1, 3, 2))
        assert_array_equal(y, [[0, 1, 0]])
        self.assertEqual(y.shape, (1, 3))

        with self.assertRaises(StopIteration):
            train_input_fn()

    @patch("tfrank.sample")
    def test_data_generator_variable_number_docs(self, mock_sample):
        mock_sample.return_value = [1, 2, 3]

        dataset = pd.DataFrame(
            {
                "qid": [1, 1, 1, 2, 2, 3, 3, 3, 3],
                "docid": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                "relevant": [1, 0, 0, 0, 1, 0, 0, 1, 0],
                "f1": [11, 12, 13, 14, 15, 16, 17, 18, 19],
                "f2": [21, 22, 23, 24, 25, 26, 27, 28, 29],
            }
        )

        data_gen = data_generator(
            dataset=dataset,
            features=["f1", "f2"],
            label="relevant",
            queries={1, 2, 3},
            num_docs=3,
            batch_size=3,
            num_epochs=2,
        )

        def train_input_fn():
            return next(data_gen)

        x, y = train_input_fn()
        x = x["x_raw"]
        assert_array_equal(
            x,
            [
                [[11, 21], [12, 22], [13, 23]],
                [[14, 24], [15, 25], [0, 0]],
                [[16, 26], [17, 27], [18, 28]],
            ],
        )
        self.assertEqual(x.shape, (3, 3, 2))
        assert_array_equal(y, [[1, 0, 0], [0, 1, -1], [0, 0, 1]])
        self.assertEqual(y.shape, (3, 3))

        x, y = train_input_fn()
        x = x["x_raw"]
        assert_array_equal(
            x,
            [
                [[11, 21], [12, 22], [13, 23]],
                [[14, 24], [15, 25], [0, 0]],
                [[16, 26], [17, 27], [18, 28]],
            ],
        )
        self.assertEqual(x.shape, (3, 3, 2))
        assert_array_equal(y, [[1, 0, 0], [0, 1, -1], [0, 0, 1]])
        self.assertEqual(y.shape, (3, 3))

        with self.assertRaises(StopIteration):
            train_input_fn()
