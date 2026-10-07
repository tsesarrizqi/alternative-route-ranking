package datagen;

import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import models.Pair;
import models.Edge;
import models.EdgeWeightedGraph;
import models.DijkstraUndirectedSP;
import models.Router;

/**
 * STEP 1: Path Generation
 * 
 * INDEXING CONVENTION:
 * - Internal graph operations: 1-indexed (EdgeWeightedGraph uses 1 to V)
 * - CSV output: 0-indexed (for Python compatibility)
 * - Conversion happens only at I/O boundaries
 */
public class Step1_PlateauPathGenerator {

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
            System.err.println("Usage: java datagen.Step1_PathGenerator <output_csv> [config_json]");
            System.exit(1);
        }

        String outputFile = args[0];
        String configPath = args.length > 1 ? args[1] : "conf/step1_config.json";

        // Load configuration
        Step1Config config = Step1Config.load(configPath);

        EdgeWeightedGraph G = new EdgeWeightedGraph(config.edgesFile, config.coordinatesFile);
        int V = G.V();

        System.out.println("=== STEP 1: Path Generation ===");
        System.out.println("Graph loaded: " + V + " nodes");
        System.out.println("Generating " + config.totalPairs + " OD pairs with paths...");
        System.out.println("Target per bucket: " + config.pairsPerBucket);

        // Thread-safe collections
        Map<Integer, List<PathData>> bucketData = new ConcurrentHashMap<>();
        Map<Integer, Set<String>> bucketUsedPairs = new ConcurrentHashMap<>();

        for (int i = 0; i < config.distanceBuckets.length; i++) {
            bucketData.put(i, Collections.synchronizedList(new ArrayList<>()));
            bucketUsedPairs.put(i, ConcurrentHashMap.newKeySet());
        }

        // Create thread pool
        ExecutorService executor = Executors.newFixedThreadPool(config.numThreads);
        List<Future<Integer>> futures = new ArrayList<>();

        // Launch worker threads
        for (int t = 0; t < config.numThreads; t++) {
            final int threadId = t;
            futures.add(executor.submit(() ->
                generatePaths(G, bucketData, bucketUsedPairs, threadId, config)
            ));
        }

        // Progress monitor
        Thread monitor = new Thread(() -> {
            while (!Thread.interrupted()) {
                printProgress(bucketData, config);
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
        printProgress(bucketData, config);

        // Save to file
        saveToFile(bucketData, outputFile, config);
        System.out.println("\nPaths saved to: " + outputFile);
        System.out.println("Ready for Step 2: Scoring");
    }
    
    private static int generatePaths(EdgeWeightedGraph G,
                                    Map<Integer, List<PathData>> bucketData,
                                    Map<Integer, Set<String>> bucketUsedPairs,
                                    int threadId,
                                    Step1Config config) {
        Random rand = new Random(System.currentTimeMillis() + threadId * 1000);
        int V = G.V();
        int attempts = 0;
        int successes = 0;

        while (getTotalPairs(bucketData) < config.totalPairs && attempts < 500000) {
            attempts++;

            // Random pair selection
            int s = rand.nextInt(V) + 1;
            int t = rand.nextInt(V) + 1;
            if (s == t) continue;

            String pairKey = s + "," + t;

            // Check if used
            boolean used = false;
            for (Set<String> usedSet : bucketUsedPairs.values()) {
                if (usedSet.contains(pairKey)) {
                    used = true;
                    break;
                }
            }
            if (used) continue;

            try {
                // Quick distance check
                DijkstraUndirectedSP spt = new DijkstraUndirectedSP(G, s, t, config.ub);
                if (spt.distTo(t) == Double.POSITIVE_INFINITY) continue;

                double optimalDistance = spt.distTo(t);
                if (optimalDistance < config.minDistance || optimalDistance > config.maxDistance) continue;

                // Get optimal path
                List<Integer> optimalPath = getOptimalPath(spt, s, t);
                int optimalPathEdges = optimalPath.size() - 1;
                if (optimalPathEdges < config.minEdges || optimalPathEdges > config.maxEdges) continue;

                // Check bucket
                int bucket = getDistanceBucketIndex(optimalDistance, config);
                if (bucket == -1) continue;

                // Check if bucket is full
                if (bucketData.get(bucket).size() >= config.pairsPerBucket) continue;

                // Generate alternative paths
                DijkstraUndirectedSP tpt = new DijkstraUndirectedSP(G, t, spt, true, config.ub);
                if (tpt.distTo(s) == Double.POSITIVE_INFINITY) continue;

                Router router = new Router(s, t, G, spt, tpt, config.tk, config.ub, config.pen, config.sim, config.dm);
                List<Pair<Double, List<Integer>>> ordered = router.getOrderedPaths();

                if (ordered == null || ordered.size() < config.minPathAlternatives) continue;
                
                // Extract paths (NO SCORING YET)
                List<List<Integer>> paths = new ArrayList<>();
                for (int i = 0; i < Math.min(config.tk, ordered.size()); i++) {
                    List<Integer> path = ordered.get(i).value;
                    if (path != null && path.size() >= 2) {
                        paths.add(path);
                    }
                }

                if (paths.size() < config.minPathAlternatives) continue;
                
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
    
    private static void saveToFile(Map<Integer, List<PathData>> bucketData, String filename, Step1Config config) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            // Header
            writer.write("source,destination,paths,optimal_distance,optimal_edges,bucket_id");
            writer.newLine();

            // Write data
            for (int bucket = 0; bucket < config.distanceBuckets.length; bucket++) {
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
                    
                    // Write line
                    writer.write(String.format("%d,%d,\"%s\",%.2f,%d,%d",
                        data.source - 1,  // Convert to 0-indexed for output
                        data.destination - 1,  // Convert to 0-indexed for output
                        String.join(";", pathStrings),
                        data.optimalDistance,
                        data.optimalPathEdges,
                        data.bucketId
                    ));
                    writer.newLine();
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
    private static void printProgress(Map<Integer, List<PathData>> bucketData, Step1Config config) {
        System.out.println("\n--- Progress ---");
        int total = 0;
        for (int i = 0; i < config.distanceBuckets.length; i++) {
            int count = bucketData.get(i).size();
            total += count;
            System.out.printf("Bucket %d [%.0f-%.0f km]: %d/%d (%.1f%%)\n",
                i, config.distanceBuckets[i][0]/1000, config.distanceBuckets[i][1]/1000,
                count, config.pairsPerBucket, 100.0 * count / config.pairsPerBucket);
        }
        System.out.println("Total: " + total + "/" + config.totalPairs);
    }
    
    private static int getTotalPairs(Map<Integer, List<PathData>> bucketData) {
        int total = 0;
        for (List<PathData> list : bucketData.values()) {
            total += list.size();
        }
        return total;
    }
    
    private static int getDistanceBucketIndex(double distance, Step1Config config) {
        for (int i = 0; i < config.distanceBuckets.length; i++) {
            if (distance >= config.distanceBuckets[i][0] && distance < config.distanceBuckets[i][1]) {
                return i;
            }
        }
        return -1;
    }
    
    private static List<Integer> getOptimalPath(DijkstraUndirectedSP spt, int s, int t) {
        List<Integer> path = new ArrayList<>();
        int current = t;
        
        while (current != s && current != Integer.MAX_VALUE) {
            path.add(current);
            Edge e = spt.getEdge(current);
            if (e == null) break;
            current = e.either() == current ? e.other(current) : e.either();
        }
        
        if (current == s) {
            path.add(s);
            Collections.reverse(path);
        }
        
        return path;
    }
}