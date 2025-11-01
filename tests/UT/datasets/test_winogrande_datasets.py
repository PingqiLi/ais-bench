# import unittest
# from unittest.mock import patch, mock_open

# from datasets import Dataset, DatasetDict

# from ais_bench.benchmark.datasets.winogrande import (
#     WinograndeDataset,
#     WinograndeDatasetV2,
#     WinograndeDatasetV3,
# )


# class TestWinogrande(unittest.TestCase):
#     @patch("ais_bench.benchmark.datasets.winogrande.get_data_path", return_value="/fake/path")
#     @patch("builtins.open")
#     def test_v1(self, mock_open_file, mock_get_path):
#         line = '{"sentence": "x _ y", "option1": "A", "option2": "B", "answer": "A"}'
#         m = mock_open(read_data=line + "\n")
#         mock_open_file.return_value = m.return_value
#         ds = WinograndeDataset.load("/any")
#         self.assertIsInstance(ds, Dataset)
#         self.assertGreaterEqual(len(ds), 1)

#     @patch("ais_bench.benchmark.datasets.winogrande.get_data_path", return_value="/fake/path")
#     @patch("builtins.open")
#     def test_v2(self, mock_open_file, mock_get_path):
#         line = '{"sentence": "x _ y", "option1": "A", "option2": "B", "answer": "1"}'
#         m = mock_open(read_data=line + "\n")
#         mock_open_file.return_value = m.return_value
#         ds = WinograndeDatasetV2.load("/any")
#         self.assertIsInstance(ds, Dataset)
#         self.assertGreaterEqual(len(ds), 1)

#     @patch("ais_bench.benchmark.datasets.winogrande.get_data_path", return_value="/fake/path")
#     @patch("builtins.open")
#     def test_v3(self, mock_open_file, mock_get_path):
#         line = '{"sentence": "x _ y", "option1": "A", "option2": "B", "answer": "1"}'
#         m = mock_open(read_data=line + "\n")
#         # train_xs 与 dev 两次打开
#         mock_open_file.side_effect = [m.return_value, m.return_value]
#         ds = WinograndeDatasetV3.load("/any")
#         self.assertIsInstance(ds, DatasetDict)
#         self.assertIn("train_xs", ds)
#         self.assertIn("dev", ds)


# if __name__ == "__main__":
#     unittest.main()
