import osmnx as ox
import csv
from osmnx.truncate import largest_component
import os

def save_graph_and_nodes_edges_for_place(
    place_name: str,
    edge_csv: str,
    node_csv: str
):
    G = ox.graph_from_place(place_name, network_type='drive', simplify=True)
    G = largest_component(G, strongly=True)
    
    # Additional safety: remove any remaining self-loops
    import networkx as nx
    G.remove_edges_from(nx.selfloop_edges(G))
    
    node_mapping = {orig: new for new, orig in enumerate(G.nodes())}
    
    os.makedirs(os.path.dirname(edge_csv), exist_ok=True)
    
    with open(edge_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['start_id', 'end_id', 'length'])
        for u, v, key, data in G.edges(keys=True, data=True):
            u2 = node_mapping[u]
            v2 = node_mapping[v]
            
            # Extra check: skip if somehow still a self-loop
            if u2 == v2:
                continue
            
            length = data.get('length')
            if length is None:
                continue
            w.writerow([u2, v2, float(length)])
    
    with open(node_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'latitude', 'longitude'])
        for orig, new in node_mapping.items():
            y = G.nodes[orig]['y']
            x = G.nodes[orig]['x']
            w.writerow([new, y, x])

def save_graph_from_bbox(north, south, east, west, edge_csv, node_csv):
    G = ox.graph_from_bbox(bbox=(north, south, east, west), 
                           network_type='drive', simplify=True)
    G = largest_component(G, strongly=True)
    
    # Additional safety: remove any remaining self-loops
    import networkx as nx
    G.remove_edges_from(nx.selfloop_edges(G))
    
    node_mapping = {orig: new for new, orig in enumerate(G.nodes())}
    
    os.makedirs(os.path.dirname(edge_csv), exist_ok=True)
    
    with open(edge_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['start_id', 'end_id', 'length'])
        for u, v, key, data in G.edges(keys=True, data=True):
            u2 = node_mapping[u]
            v2 = node_mapping[v]
            
            # Extra check: skip if somehow still a self-loop
            if u2 == v2:
                continue
            
            length = data.get('length')
            if length is None:
                continue
            w.writerow([u2, v2, float(length)])
    
    with open(node_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'latitude', 'longitude'])
        for orig, new in node_mapping.items():
            y = G.nodes[orig]['y']
            x = G.nodes[orig]['x']
            w.writerow([new, y, x])

def maybe_download_city(place_name, city_key):
    graph_dir = os.path.join('data', city_key, 'graph')
    edge_csv = os.path.join(graph_dir, 'edges.csv')
    node_csv = os.path.join(graph_dir, 'nodes.csv')

    if os.path.isfile(edge_csv) and os.path.isfile(node_csv):
        print(f"[SKIP] {city_key}")
        return

    print(f"Downloading {city_key}: {place_name}")

    save_graph_and_nodes_edges_for_place(
        place_name=place_name,
        edge_csv=edge_csv,
        node_csv=node_csv
    )

    print(f"Finished {city_key}\n")

if __name__ == '__main__':

    cities = {
        "london": "Greater London, England, United Kingdom",
        "manhattan": "Manhattan, New York, USA",
        "seattle": "Seattle, Washington, USA",
        "paris": "Paris, France",
        "bangalore": "Bangalore, Karnataka, India",
        "manchester": "Manchester, England, United Kingdom",
        "sydney": "Sydney, New South Wales, Australia",
        "tokyo": "Tokyo, Japan",
        "amsterdam": "Amsterdam, North Holland, Netherlands",
        "singapore": "Singapore",
        "rome": "Rome, Lazio, Italy",
    }

    for city_key, place_name in cities.items():
        maybe_download_city(place_name, city_key)
