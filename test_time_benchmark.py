import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import time_benchmark
from rescore_stretch import collect_source_usages, compute_scores_streaming


class ExactStretchBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        graph_directory = Path(self.directory.name) / "graph"
        graph_directory.mkdir()
        pd.DataFrame({"id": [10, 20, 30]}).to_csv(graph_directory / "nodes.csv", index=False)
        pd.DataFrame({
            "start_id": [10, 20, 10, 30],
            "end_id": [20, 30, 30, 10],
            "length": [1.0, 2.0, 2.0, 0.1],
        }).to_csv(graph_directory / "edges.csv", index=False)
        self.dataset = SimpleNamespace(
            graph_data_dir=str(graph_directory),
            edge_lengths=torch.tensor([1.0, 2.0, 2.0, 0.1]),
            graph_data=SimpleNamespace(edge_sd=torch.tensor([[10, 20, 10, 30], [20, 30, 30, 10]])),
        )
        self.batch = {
            "path2pair": torch.tensor([0, 0]),
            "node_sequences": [[10, 20, 30], [10, 30]],
            "stretch_scores": [[0.0, 0.0]],
            "subpath_indices": {(10, 30): {(10, 30): 999.0}},
        }

    def test_exact_directed_scores_ignore_pool_and_stored_labels(self):
        calculator = time_benchmark.GroundTruthCalculator(self.dataset)
        np.testing.assert_allclose(calculator.calculate_stretch_batch(self.batch), [[50.0, 100.0]])
        self.batch.pop("subpath_indices")
        np.testing.assert_allclose(calculator.calculate_stretch_batch(self.batch), [[50.0, 100.0]])

    def test_epsilon_is_respected(self):
        calculator = time_benchmark.GroundTruthCalculator(self.dataset, epsilon=0.5)
        np.testing.assert_allclose(calculator.calculate_stretch_batch(self.batch), [[100.0, 100.0]])

    def test_matches_exact_label_scorer(self):
        calculator = time_benchmark.GroundTruthCalculator(self.dataset)
        frame = pd.DataFrame({"paths": ["10,20,30;10,30;10,20,30,10,30"]})
        paths, usages = collect_source_usages(frame, calculator.node_index)
        accumulators = compute_scores_streaming(
            paths, usages, calculator.exact_edge_lengths, calculator.node_index,
            calculator.road_graph, [0.1], n_workers=1, batch_size=2,
        )
        total, satisfying = accumulators[:2]
        self.batch["path2pair"] = torch.tensor([0, 0, 0])
        self.batch["node_sequences"].append([10, 20, 30, 10, 30])
        expected = [[100.0 * satisfying[0][(0, index)] / total[(0, index)] for index in range(3)]]
        np.testing.assert_allclose(calculator.calculate_stretch_batch(self.batch), expected)

    def test_repeated_calls_recompute_distances(self):
        calculator = time_benchmark.GroundTruthCalculator(self.dataset)
        with patch.object(time_benchmark, "dijkstra", wraps=time_benchmark.dijkstra) as solver:
            calculator.calculate_stretch_batch(self.batch)
            first_count = solver.call_count
            calculator.calculate_stretch_batch(self.batch)
            self.assertEqual(first_count, 3)
            self.assertEqual(solver.call_count, 2 * first_count)

    def test_empty_od_and_degenerate_path(self):
        calculator = time_benchmark.GroundTruthCalculator(self.dataset)
        self.batch["stretch_scores"] = [[], [0.0]]
        self.batch["path2pair"] = torch.tensor([1])
        self.batch["node_sequences"] = [[10]]
        self.assertEqual(calculator.calculate_stretch_batch(self.batch), [[], [100.0]])

    def test_invalid_epsilon_is_rejected(self):
        for epsilon in (-0.1, float("inf"), float("nan")):
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                time_benchmark.GroundTruthCalculator(self.dataset, epsilon=epsilon)


if __name__ == "__main__":
    unittest.main()