package models;

import java.util.*;

/**
 * Router using K-Shortest Paths (Yen's Algorithm) 
 * Alternative to the plateau/via-node Router
 */
public class RouterKSP {
    private int s;
    private int t;
    private int TK;
    private double UB;
    private EdgeWeightedGraph TG;

    public RouterKSP(int s, int t, EdgeWeightedGraph G, int tk, double ub) {
        this.s = s;
        this.t = t;
        this.TG = G;
        this.TK = tk;
        this.UB = ub;
    }

    /**
     * Return the first TK paths using Yen's K-shortest paths algorithm.
     * Each element is a Pair where:
     *   key   = total path length
     *   value = list of node IDs forming the path
     */
    public List<Pair<Double, List<Integer>>> getOrderedPaths() {
        List<Pair<Double, List<Integer>>> result = new ArrayList<>();
        
        // Use Yen's algorithm to find K shortest paths
        YenKShortestPaths yen = new YenKShortestPaths(TG, s, t, TK, UB);
        List<YenKShortestPaths.Path> kPaths = yen.getPaths();
        
        // Convert to the expected format
        for (YenKShortestPaths.Path path : kPaths) {
            result.add(new Pair<>(path.distance, path.nodes));
        }
        
        return result;
    }
}