#!/usr/bin/env python3
"""Convert graph CSV files into the text format used by the Java generators."""

import argparse
import csv
from pathlib import Path


def convert_graph(nodes_csv: Path, edges_csv: Path, output_dir: Path, city: str) -> tuple[int, int]:
    with nodes_csv.open(newline="", encoding="utf-8") as source:
        nodes = list(csv.DictReader(source))
    nodes.sort(key=lambda row: int(row["id"]))
    node_ids = [int(row["id"]) for row in nodes]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError(f"Duplicate node IDs in {nodes_csv}")
    node_map = {node_id: index + 1 for index, node_id in enumerate(node_ids)}

    converted_edges = []
    with edges_csv.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            start = int(row["start_id"])
            end = int(row["end_id"])
            if start not in node_map or end not in node_map:
                raise ValueError(f"Edge references a missing node: {start}->{end}")
            converted_edges.append((node_map[start], node_map[end], float(row["length"])))

    output_dir.mkdir(parents=True, exist_ok=True)
    coordinates_path = output_dir / f"{city}_converted_coordinates.txt"
    edges_path = output_dir / f"{city}_converted_edges.txt"

    with coordinates_path.open("w", encoding="utf-8", newline="\n") as output:
        for row, node_id in zip(nodes, node_ids):
            new_id = node_map[node_id]
            output.write(f"node{new_id} {new_id} {row['latitude']} {row['longitude']}\n")

    with edges_path.open("w", encoding="utf-8", newline="\n") as output:
        output.write(f"{len(nodes)}\n{len(converted_edges)}\n")
        for start, end, length in converted_edges:
            output.write(f"E {start} {end} {length:.6f}\n")

    return len(nodes), len(converted_edges)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("city", help="City key used by the Java Step 1 config")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--java-conf-dir", type=Path, default=Path("candidate_paths_generator/conf")
    )
    args = parser.parse_args()

    graph_dir = args.data_root / args.city / "graph"
    node_count, edge_count = convert_graph(
        graph_dir / "nodes.csv",
        graph_dir / "edges.csv",
        args.java_conf_dir,
        args.city,
    )
    print(f"Converted {node_count} nodes and {edge_count} edges for {args.city}.")


if __name__ == "__main__":
    main()
