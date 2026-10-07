package models;

import java.util.*;

/**
 * Yen's K-Shortest Paths Algorithm
 * Finds K shortest loopless paths between two nodes in a graph
 */
public class YenKShortestPaths {
    private final EdgeWeightedGraph graph;
    private final int source;
    private final int target;
    private final int K;
    private final double maxStretch;
    
    private List<Path> kShortestPaths;
    
    public static class Path implements Comparable<Path> {
        public final List<Integer> nodes;
        public final double distance;
        
        public Path(List<Integer> nodes, double distance) {
            this.nodes = new ArrayList<>(nodes);
            this.distance = distance;
        }
        
        @Override
        public int compareTo(Path other) {
            return Double.compare(this.distance, other.distance);
        }
        
        @Override
        public boolean equals(Object obj) {
            if (!(obj instanceof Path)) return false;
            Path other = (Path) obj;
            return this.nodes.equals(other.nodes);
        }
        
        @Override
        public int hashCode() {
            return nodes.hashCode();
        }
    }
    
    public YenKShortestPaths(EdgeWeightedGraph graph, int source, int target, int K, double maxStretch) {
        this.graph = graph;
        this.source = source;
        this.target = target;
        this.K = K;
        this.maxStretch = maxStretch;
        this.kShortestPaths = new ArrayList<>();
        
        computeKShortestPaths();
    }
    
    public List<Path> getPaths() {
        return new ArrayList<>(kShortestPaths);
    }
    
    public List<List<Integer>> getPathNodes() {
        List<List<Integer>> result = new ArrayList<>();
        for (Path p : kShortestPaths) {
            result.add(new ArrayList<>(p.nodes));
        }
        return result;
    }
    
    private void computeKShortestPaths() {
        // Priority queue for candidate paths
        PriorityQueue<Path> candidates = new PriorityQueue<>();
        Set<List<Integer>> seenPaths = new HashSet<>();
        
        // Step 1: Find the shortest path
        DijkstraUndirectedSP sp = new DijkstraUndirectedSP(graph, source, target);
        if (!sp.hasPathTo(target)) {
            return; // No path exists
        }
        
        List<Integer> shortestPath = sp.pathTo(target);
        double shortestDist = sp.distTo(target);
        double maxAllowedDist = shortestDist * maxStretch;
        
        Path firstPath = new Path(shortestPath, shortestDist);
        kShortestPaths.add(firstPath);
        seenPaths.add(shortestPath);
        
        // Step 2: Find K-1 more paths
        for (int k = 1; k < K; k++) {
            Path lastPath = kShortestPaths.get(kShortestPaths.size() - 1);
            
            // For each node in the last found path (except the target)
            for (int i = 0; i < lastPath.nodes.size() - 1; i++) {
                int spurNode = lastPath.nodes.get(i);
                
                // Extract the root path (source to spur node)
                List<Integer> rootPath = new ArrayList<>(lastPath.nodes.subList(0, i + 1));
                double rootDist = calculatePathDistance(rootPath);
                
                // Find paths that share this root
                Set<Integer> blockedNodes = new HashSet<>();
                Set<Edge> blockedEdges = new HashSet<>();
                
                // Block edges that would create duplicate sub-paths
                for (Path existingPath : kShortestPaths) {
                    if (existingPath.nodes.size() > i + 1 && 
                        pathsShareRoot(existingPath.nodes, rootPath)) {
                        // Block the edge from spur node to next node in existing path
                        int nextNode = existingPath.nodes.get(i + 1);
                        blockEdge(blockedEdges, spurNode, nextNode);
                    }
                }
                
                // Block nodes in root path (except spur node) to avoid loops
                for (int j = 0; j < i; j++) {
                    blockedNodes.add(rootPath.get(j));
                }
                
                // Find shortest path from spur node to target with restrictions
                List<Integer> spurPath = findRestrictedPath(spurNode, target, 
                                                            blockedNodes, blockedEdges);
                
                if (spurPath != null) {
                    // Combine root and spur paths
                    List<Integer> totalPath = new ArrayList<>(rootPath);
                    for (int j = 1; j < spurPath.size(); j++) {
                        totalPath.add(spurPath.get(j));
                    }
                    
                    double totalDist = calculatePathDistance(totalPath);
                    
                    // Check if path is within stretch limit and not seen before
                    if (totalDist <= maxAllowedDist && seenPaths.add(totalPath)) {
                        candidates.add(new Path(totalPath, totalDist));
                    }
                }
            }
            
            // Select the best candidate
            if (candidates.isEmpty()) {
                break; // No more paths to find
            }
            
            Path bestCandidate = candidates.poll();
            kShortestPaths.add(bestCandidate);
        }
    }
    
    private boolean pathsShareRoot(List<Integer> path1, List<Integer> rootPath) {
        if (path1.size() < rootPath.size()) return false;
        
        for (int i = 0; i < rootPath.size(); i++) {
            if (!path1.get(i).equals(rootPath.get(i))) {
                return false;
            }
        }
        return true;
    }
    
    private void blockEdge(Set<Edge> blockedEdges, int u, int v) {
        // Find and block the edge between u and v
        for (Edge e : graph.adj(u)) {
            if (e.other(u) == v) {
                blockedEdges.add(e);
                break;
            }
        }
    }
    
    private List<Integer> findRestrictedPath(int start, int end, 
                                            Set<Integer> blockedNodes, 
                                            Set<Edge> blockedEdges) {
        // Modified Dijkstra with restrictions
        double[] distTo = new double[graph.V() + 1];
        Edge[] edgeTo = new Edge[graph.V() + 1];
        boolean[] visited = new boolean[graph.V() + 1];
        
        Arrays.fill(distTo, Double.POSITIVE_INFINITY);
        distTo[start] = 0;
        
        PriorityQueue<Node> pq = new PriorityQueue<>();
        pq.offer(new Node(0, start));
        
        while (!pq.isEmpty()) {
            Node node = pq.poll();
            int v = node.getV();
            
            if (visited[v]) continue;
            visited[v] = true;
            
            if (v == end) {
                // Reconstruct path
                return reconstructPath(start, end, edgeTo);
            }
            
            // Explore neighbors
            for (Edge e : graph.adj(v)) {
                int w = e.other(v);
                
                // Skip if node or edge is blocked
                if (blockedNodes.contains(w) || blockedEdges.contains(e)) {
                    continue;
                }
                
                double newDist = distTo[v] + e.weight();
                if (newDist < distTo[w]) {
                    distTo[w] = newDist;
                    edgeTo[w] = e;
                    pq.offer(new Node(newDist, w));
                }
            }
        }
        
        return null; // No path found
    }
    
    private List<Integer> reconstructPath(int start, int end, Edge[] edgeTo) {
        List<Integer> path = new ArrayList<>();
        int x = end;
        
        while (x != start) {
            path.add(x);
            Edge e = edgeTo[x];
            if (e == null) return null;
            x = e.other(x);
        }
        path.add(start);
        
        Collections.reverse(path);
        return path;
    }
    
    private double calculatePathDistance(List<Integer> path) {
        double distance = 0;
        for (int i = 0; i < path.size() - 1; i++) {
            distance += getEdgeWeight(path.get(i), path.get(i + 1));
        }
        return distance;
    }
    
    private double getEdgeWeight(int u, int v) {
        for (Edge e : graph.adj(u)) {
            if (e.other(u) == v) {
                return e.weight();
            }
        }
        return Double.POSITIVE_INFINITY;
    }
}