import unittest
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from unittest.mock import patch
from ais_bench.benchmark.utils.datasets import get_sample_data  


class TestGetSampleData(unittest.TestCase):
    def setUp(self):
        self.test_data = ["prompt1", "prompt2", "prompt3", "prompt4", "prompt5"]

    @patch('ais_bench.benchmark.utils.datasets.logger.info')
    def test_no_request_count(self, mock_warning):
        result = get_sample_data(self.test_data)
        mock_warning.assert_called_once()
        self.assertEqual(result, self.test_data)

    def test_request_count_larger_than_data(self):
        result = get_sample_data(self.test_data, request_count=7)
        self.assertEqual(result, self.test_data + ["prompt1", "prompt2"])

    def test_valid_request_count(self):
        result = get_sample_data(self.test_data, request_count=3)
        self.assertEqual(len(result), 3)
        self.assertEqual(result, self.test_data[:3])

    def test_default_sample_mode(self):
        result = get_sample_data(self.test_data, sample_mode="default", request_count=3)
        self.assertEqual(result, self.test_data[:3])

    def test_random_sample_mode(self):
        result = get_sample_data(self.test_data, sample_mode="random", request_count=3)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(item in self.test_data for item in result))

    def test_shuffle_sample_mode(self):
        result = get_sample_data(self.test_data, sample_mode="shuffle", request_count=3)
        self.assertEqual(len(result), 3)

    def test_invalid_sample_mode(self):
        with self.assertRaises(ValueError):
            get_sample_data(self.test_data, sample_mode="invalid_mode")


if __name__ == '__main__':
    unittest.main()