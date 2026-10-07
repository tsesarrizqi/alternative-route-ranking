package datagen;

import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import models.Pair;
import models.Edge;
import models.EdgeWeightedGraph;
import models.DijkstraUndirectedSP;
import models.RouterKSP;

/**
 * STEP 1: Path Generation using K-Shortest Paths
 * Alternative to plateau/via-node method
 * 
 * INDEXING CONVENTION:
 * - Internal graph operations: 1-indexed (EdgeWeightedGraph uses 1 to V)
 * - CSV output: 0-indexed (for Python compatibility)
 * - Conversion happens only at I/O boundaries
 */
public class Step1_PathGeneratorKSP {
    // These are set from config at startup — do not use as constants
    private static int TK;
    private static double UB;
    private static int TOTAL_PAIRS;
    private static double MIN_DISTANCE;
    private static double MAX_DISTANCE;
    private static double[][] DISTANCE_BUCKETS;
    private static int PAIRS_PER_BUCKET;
    private static int MIN_EDGES;
    private static int MAX_EDGES;
    private static int MIN_PATH_ALTERNATIVES;
    private static int NUM_THREADS;

    static class PathData {
        int source;
        int destination;
        List<List<Integer>> paths;
        double optimalDistance;
        int optimalPathEdges;
        int bucketId;
        
        PathData(int s, int d, List<List<Integer>> p, double od, int ope, int bid) {
            this.source = s;
            this.destination = d;
            this.paths = p;
            this.optimalDistance = od;
            this.optimalPathEdges = ope;
            this.bucketId = bid;
        }
    }

    public static void main(String[] args) throws InterruptedException {

        if (args.length < 1) {
            System.err.println("Usage: java datagen.Step1_PathGeneratorKSP <output_csv> [config_json]");
            System.exit(1);
        }

        String outputFile = args[0];
        String configPath = args.length > 1 ? args[1] : "conf/step1_config.json";

        Step1Config config = Step1Config.load(configPath);

        // Populate static fields from config
        TK                   = config.tk;
        UB                   = config.ub;
        TOTAL_PAIRS          = config.totalPairs;
        MIN_DISTANCE         = config.minDistance;
        MAX_DISTANCE         = config.maxDistance;
        DISTANCE_BUCKETS     = config.distanceBuckets;
        PAIRS_PER_BUCKET     = config.pairsPerBucket;
        MIN_EDGES            = config.minEdges;
        MAX_EDGES            = config.maxEdges;
        MIN_PATH_ALTERNATIVES = config.minPathAlternatives;
        NUM_THREADS          = config.numThreads;

        EdgeWeightedGraph G = new EdgeWeightedGraph(config.edgesFile, config.coordinatesFile);
        int V = G.V();
        
        System.out.println("=== STEP 1: Path Generation (K-Shortest Paths) ===");
        System.out.println("Graph loaded: " + V + " nodes");
        System.out.println("Generating " + TOTAL_PAIRS + " OD pairs with K-shortest paths...");
        System.out.println("Target per bucket: " + PAIRS_PER_BUCKET);
        
        // Thread-safe collections
        Map<Integer, List<PathData>> bucketData = new ConcurrentHashMap<>();
        Map<Integer, Set<String>> bucketUsedPairs = new ConcurrentHashMap<>();
        
        for (int i = 0; i < DISTANCE_BUCKETS.length; i++) {
            bucketData.put(i, Collections.synchronizedList(new ArrayList<>()));
            bucketUsedPairs.put(i, ConcurrentHashMap.newKeySet());
        }
        
        // Create thread pool
        ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
        List<Future<Integer>> futures = new ArrayList<>();
        
        // Launch worker threads
        for (int t = 0; t < NUM_THREADS; t++) {
            final int threadId = t;
            futures.add(executor.submit(() -> 
                generatePaths(G, bucketData, bucketUsedPairs, threadId)
            ));
        }
        
        // Progress monitor
        Thread monitor = new Thread(() -> {
            while (!Thread.interrupted()) {
                printProgress(bucketData);
                try {
                    Thread.sleep(5000);
                } catch (InterruptedException e) {
                    break;
                }
            }
        });
        monitor.start();
        
        // Wait for completion
        for (Future<Integer> future : futures) {
            try {
                future.get();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        
        executor.shutdown();
        monitor.interrupt();
        
        // Final report
        System.out.println("\n=== Final Results ===");
        printProgress(bucketData);
        
        // Save to file
        saveToFile(bucketData, outputFile);
        System.out.println("\nPaths saved to: " + outputFile);
        System.out.println("Ready for Step 2: Scoring");
    }
    
    private static int generatePaths(EdgeWeightedGraph G, 
                                    Map<Integer, List<PathData>> bucketData,
                                    Map<Integer, Set<String>> bucketUsedPairs,
                                    int threadId) {
        Random rand = new Random(System.nanoTime() + threadId);
        int V = G.V();
        int attempts = 0;
        int successes = 0;
        
        while (anyBucketNeedsMore(bucketData)) {
            attempts++;
            if (attempts > 50000) break;
            
            try {
                // Random OD pair (1-indexed)
                int s = rand.nextInt(V) + 1;
                int t = rand.nextInt(V) + 1;
                if (s == t) continue;
                
                // Find shortest path
                DijkstraUndirectedSP sp = new DijkstraUndirectedSP(G, s, t);
                if (!sp.hasPathTo(t)) continue;
                
                double optimalDistance = sp.distTo(t);
                if (optimalDistance < MIN_DISTANCE || optimalDistance > MAX_DISTANCE) continue;
                
                // Determine bucket
                int bucket = getBucketId(optimalDistance);
                if (bucketData.get(bucket).size() >= PAIRS_PER_BUCKET) continue;
                
                // Check if pair already used
                String pairKey = s + "_" + t;
                if (!bucketUsedPairs.get(bucket).add(pairKey)) continue;
                
                // Get optimal path
                List<Integer> optimalPath = sp.pathTo(t);
                if (optimalPath == null || optimalPath.size() < 2) continue;
                
                int optimalPathEdges = optimalPath.size() - 1;
                if (optimalPathEdges < MIN_EDGES || optimalPathEdges > MAX_EDGES) continue;
                
                // Use K-Shortest Paths Router
                RouterKSP router = new RouterKSP(s, t, G, TK, UB);
                List<Pair<Double, List<Integer>>> ordered = router.getOrderedPaths();
                
                if (ordered == null || ordered.size() < MIN_PATH_ALTERNATIVES) continue;
                
                // Extract paths (NO SCORING YET)
                List<List<Integer>> paths = new ArrayList<>();
                for (int i = 0; i < Math.min(TK, ordered.size()); i++) {
                    List<Integer> path = ordered.get(i).value;
                    if (path != null && path.size() >= 2) {
                        paths.add(path);
                    }
                }
                
                if (paths.size() < MIN_PATH_ALTERNATIVES) continue;
                
                // Add to results
                PathData data = new PathData(s, t, paths, optimalDistance, optimalPathEdges, bucket);
                bucketData.get(bucket).add(data);
                bucketUsedPairs.get(bucket).add(pairKey);
                successes++;
                
                if (successes % 50 == 0) {
                    System.out.printf("Thread %d: %d pairs found (%.1f%% success rate)\n",
                        threadId, successes, 100.0 * successes / attempts);
                }
                
            } catch (Exception e) {
                // Skip problematic pairs
            }
        }
        
        return successes;
    }
    
    private static boolean anyBucketNeedsMore(Map<Integer, List<PathData>> bucketData) {
        for (int i = 0; i < DISTANCE_BUCKETS.length; i++) {
            if (bucketData.get(i).size() < PAIRS_PER_BUCKET) {
                return true;
            }
        }
        return false;
    }
    
    private static int getBucketId(double distance) {
        for (int i = 0; i < DISTANCE_BUCKETS.length; i++) {
            if (distance >= DISTANCE_BUCKETS[i][0] && distance < DISTANCE_BUCKETS[i][1]) {
                return i;
            }
        }
        return DISTANCE_BUCKETS.length - 1;
    }
    
    private static void printProgress(Map<Integer, List<PathData>> bucketData) {
        System.out.println("\nProgress by bucket:");
        int total = 0;
        for (int i = 0; i < DISTANCE_BUCKETS.length; i++) {
            int count = bucketData.get(i).size();
            total += count;
            System.out.printf("  Bucket %d [%.0f-%.0f]: %d/%d\n", 
                i, DISTANCE_BUCKETS[i][0], DISTANCE_BUCKETS[i][1], 
                count, PAIRS_PER_BUCKET);
        }
        System.out.println("  Total: " + total + "/" + TOTAL_PAIRS);
    }
    
    private static void saveToFile(Map<Integer, List<PathData>> bucketData, String filename) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            // Header
            writer.write("source,destination,paths,optimal_distance,optimal_edges,bucket_id");
            writer.newLine();
            
            // Write data
            for (int bucket = 0; bucket < DISTANCE_BUCKETS.length; bucket++) {
                for (PathData data : bucketData.get(bucket)) {
                    // Convert paths to string format
                    List<String> pathStrings = new ArrayList<>();
                    for (List<Integer> path : data.paths) {
                        List<String> nodeIds = new ArrayList<>();
                        for (int node : path) {
                            nodeIds.add(String.valueOf(node - 1)); // 0-indexed
                        }
                        pathStrings.add(String.join(",", nodeIds));
                    }
                    
                    // Write row
                    writer.write(String.format("%d,%d,\"%s\",%.1f,%d,%d",
                        data.source - 1,  // 0-indexed
                        data.destination - 1,  // 0-indexed
                        String.join(";", pathStrings),
                        data.optimalDistance,
                        data.optimalPathEdges,
                        data.bucketId));
                    writer.newLine();
                }
            }
            
        } catch (IOException e) {
            System.err.println("Error writing to file: " + e.getMessage());
        }
    }
}