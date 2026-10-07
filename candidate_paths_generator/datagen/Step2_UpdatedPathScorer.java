package datagen;

import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import models.Edge;
import models.EdgeWeightedGraph;
import models.DijkstraUndirectedSP;


public class Step2_UpdatedPathScorer {
    // private static final String inputFile = "paths_raw_step1_manchester_ub_2_dm_4_sim_07_part1.csv";
    // private static final String outputFile = "paths_scored_step2_manchester_ub_2_dm_4_sim_07_part1.csv";
    
    private static String inputFile;
    private static String outputFile;
    private static Map<String, Double> shortestDistanceCache;

    // Scoring weights from paper
    private static final double W_SHARING = 0.4;
    private static final double W_STRETCH = 0.6;
    private static final double W_LOCALITY = 0;
    
    private static final int NUM_THREADS = 8;
    
    // NEW: Configuration for stretch scoring method
    private static final boolean USE_MAX_STRETCH_SCORING = false; // Set to true to use max stretch instead of ratio
    
    static class ScoredPath {
        List<Integer> path;
        double pathDistance;
        double sharingScore;
        double stretchScore;
        double localityScore;
        double finalScore;
        
        ScoredPath(List<Integer> path) {
            this.path = path;
        }
        
        void calculateFinalScore() {
            this.finalScore = W_SHARING * sharingScore + 
                            W_STRETCH * stretchScore + 
                            W_LOCALITY * localityScore;
        }
    }
    
    static class ODPairTask {
        String[] row;
        int index;
        
        ODPairTask(String[] row, int index) {
            this.row = row;
            this.index = index;
        }
    }

    public static void main(String[] args) throws InterruptedException, ExecutionException {

        if (args.length < 2) {
            System.err.println(
                "Usage: java -cp bin datagen.Step2_UpdatedPathScorer <input_csv> <output_csv> [config_json]"
            );
            System.exit(1);
        }

        inputFile = args[0];
        outputFile = args[1];
        String configPath = args.length > 2 ? args[2] : "conf/step1_config.json";

        Step1Config config = Step1Config.load(configPath);

        EdgeWeightedGraph G = new EdgeWeightedGraph(config.edgesFile);
        
        System.out.println("=== UPDATED STEP 2: Path Scoring (using shortest from list) ===");
        System.out.println("Stretch scoring method: " + (USE_MAX_STRETCH_SCORING ? "MAX STRETCH" : "RATIO-BASED"));
        System.out.println("Loading paths from: " + inputFile);
        
        List<String[]> rawData = loadRawPaths(inputFile);
        System.out.println("Loaded " + rawData.size() + " OD pairs");
        System.out.println("Using " + NUM_THREADS + " threads for parallel processing");

        buildShortestDistanceCache(G, rawData);
        
        // Create tasks
        BlockingQueue<ODPairTask> taskQueue = new LinkedBlockingQueue<>();
        for (int i = 0; i < rawData.size(); i++) {
            taskQueue.offer(new ODPairTask(rawData.get(i), i));
        }
        
        // Results collection
        ConcurrentHashMap<Integer, String> results = new ConcurrentHashMap<>();
        AtomicInteger processed = new AtomicInteger(0);
        long startTime = System.currentTimeMillis();
        
        // Create thread pool
        ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
        List<Future<?>> futures = new ArrayList<>();
        
        // Launch worker threads
        for (int t = 0; t < NUM_THREADS; t++) {
            final int threadId = t;
            Future<?> future = executor.submit(() -> {
                processODPairs(G, taskQueue, results, processed,
                             rawData.size(), threadId, startTime);
            });
            futures.add(future);
        }
        
        // Wait for all threads to complete
        for (Future<?> future : futures) {
            future.get();
        }
        
        executor.shutdown();
        
        // Write results in order
        System.out.println("\nWriting results to file...");
        saveResults(results, rawData.size(), outputFile);
        
        long totalTime = System.currentTimeMillis() - startTime;
        System.out.printf("\n=== Scoring Complete ===\n");
        System.out.printf("Total time: %.1f seconds\n", totalTime / 1000.0);
        System.out.printf("Average rate: %.1f OD pairs/sec\n", rawData.size() * 1000.0 / totalTime);
        System.out.printf("Output saved to: %s\n", outputFile);
    }

    private static void buildShortestDistanceCache(EdgeWeightedGraph G, List<String[]> rawData) {
        System.out.println("Building shortest distance cache...");
        shortestDistanceCache = new ConcurrentHashMap<>();
        
        Set<String> nodePairs = ConcurrentHashMap.newKeySet();
        
        for (String[] row : rawData) {
            String pathsStr = row[2];
            String[] pathStrings = pathsStr.split(";");
            
            for (String pathStr : pathStrings) {
                String[] nodeStrs = pathStr.split(",");
                List<Integer> path = new ArrayList<>();
                for (String nodeStr : nodeStrs) {
                    path.add(Integer.parseInt(nodeStr) + 1);
                }
                
                // Keep directional pairs (removed Math.min/Math.max)
                for (int i = 0; i < path.size(); i++) {
                    for (int j = i + 2; j <= path.size(); j++) {
                        int u = path.get(i);
                        int v = path.get(j - 1);
                        nodePairs.add(u + "," + v);  // Actual direction
                    }
                }
            }
        }
        
        System.out.println("Found " + nodePairs.size() + " unique directed node pairs");
        System.out.println("Computing shortest distances (parallel)...");
        
        AtomicInteger computed = new AtomicInteger(0);
        nodePairs.parallelStream().forEach(pair -> {
            String[] parts = pair.split(",");
            int u = Integer.parseInt(parts[0]);
            int v = Integer.parseInt(parts[1]);
            
            DijkstraUndirectedSP sp = new DijkstraUndirectedSP(G, u, v);
            double dist = sp.distTo(v);
            
            // Only store actual direction
            shortestDistanceCache.put(u + "," + v, dist);
            
            int count = computed.incrementAndGet();
            if (count % 1000 == 0) {
                System.out.printf("  Computed %d/%d (%.1f%%)\n", 
                    count, nodePairs.size(), 100.0 * count / nodePairs.size());
            }
        });
        
        System.out.println("Cache built: " + shortestDistanceCache.size() + " entries");
    }

    private static double getShortestDistance(EdgeWeightedGraph G, int u, int v) {
        String key = u + "," + v;
        Double cached = shortestDistanceCache.get(key);
        if (cached != null) {
            return cached;
        }
        // Should never reach here if cache is complete
        System.err.println("WARNING: Cache miss for " + u + "," + v);
        DijkstraUndirectedSP sp = new DijkstraUndirectedSP(G, u, v);
        return sp.distTo(v);
    }

    private static void processODPairs(EdgeWeightedGraph G, 
                                     BlockingQueue<ODPairTask> taskQueue,
                                     ConcurrentHashMap<Integer, String> results,
                                     AtomicInteger processed,
                                     int totalTasks,
                                     int threadId,
                                     long startTime) {
        while (true) {
            ODPairTask task = taskQueue.poll();
            if (task == null) break;
            
            try {
                String outputLine = scoreODPair(G, task.row);
                results.put(task.index, outputLine);
                
                int count = processed.incrementAndGet();
                if (count % 50 == 0) {
                    long elapsed = System.currentTimeMillis() - startTime;
                    double rate = count * 1000.0 / elapsed;
                    double eta = (totalTasks - count) / rate;
                    System.out.printf("Thread %d - Processed: %d/%d (%.1f%%) | Rate: %.1f OD/sec | ETA: %.1f min\n",
                        threadId, count, totalTasks, 
                        100.0 * count / totalTasks, rate, eta / 60);
                }
            } catch (Exception e) {
                System.err.println("Error processing OD pair at index " + task.index);
                e.printStackTrace();
            }
        }
    }
    
    private static String scoreODPair(EdgeWeightedGraph G, String[] row) {
        int source = Integer.parseInt(row[0]) + 1; // Convert back to 1-indexed
        int destination = Integer.parseInt(row[1]) + 1;
        String pathsStr = row[2];
        int bucketId = Integer.parseInt(row[5]);
        
        // Parse paths
        String[] pathStrings = pathsStr.split(";");
        List<List<Integer>> paths = new ArrayList<>();
        
        for (String pathStr : pathStrings) {
            String[] nodeStrs = pathStr.split(",");
            List<Integer> path = new ArrayList<>();
            for (String nodeStr : nodeStrs) {
                path.add(Integer.parseInt(nodeStr) + 1); // Convert back to 1-indexed
            }
            paths.add(path);
        }
        
        // Find the shortest path from the list
        List<Integer> shortestPath = null;
        double shortestDistance = Double.POSITIVE_INFINITY;
        
        // Calculate distance for each path
        List<Double> pathDistances = new ArrayList<>();
        for (int i = 0; i < paths.size(); i++) {
            List<Integer> path = paths.get(i);
            double distance = calculatePathDistance(path, G);
            pathDistances.add(distance);
            
            if (distance < shortestDistance) {
                shortestDistance = distance;
                shortestPath = path;
            }
        }
        
        // Use the shortest path from list as reference for scoring
        List<Integer> optimalPath = shortestPath;
        double optimalDistance = shortestDistance;
        
        // Score each path
        List<ScoredPath> scoredPaths = new ArrayList<>();
        double T = 0.75 * optimalDistance; // T parameter for locality
        
        for (int i = 0; i < paths.size(); i++) {
            List<Integer> path = paths.get(i);
            ScoredPath scored = new ScoredPath(path);
            scored.pathDistance = pathDistances.get(i);
            
            // Calculate three scores
            scored.sharingScore = calculateSharingScore(path, optimalPath, G);
            // UPDATED: Pass the USE_MAX_STRETCH_SCORING flag
            double[] stretchAndLocality = calculateStretchAndLocalityScores(
                path, G, T, USE_MAX_STRETCH_SCORING);
            scored.stretchScore = stretchAndLocality[0];
            scored.localityScore = stretchAndLocality[1];
            
            scored.calculateFinalScore();
            scoredPaths.add(scored);
        }
        
        // Sort by final score (descending)
        scoredPaths.sort((a, b) -> Double.compare(b.finalScore, a.finalScore));
        
        // Build output line
        List<String> pathStrs = new ArrayList<>();
        List<String> finalScores = new ArrayList<>();
        List<String> sharingScores = new ArrayList<>();
        List<String> stretchScores = new ArrayList<>();
        List<String> localityScores = new ArrayList<>();
        
        for (ScoredPath sp : scoredPaths) {
            // Convert path back to 0-indexed string
            List<String> nodeStrs = new ArrayList<>();
            for (int node : sp.path) {
                nodeStrs.add(String.valueOf(node - 1));
            }
            pathStrs.add(String.join(",", nodeStrs));
            
            // Format scores
            finalScores.add(String.format("%.6f", sp.finalScore * 100));
            sharingScores.add(String.format("%.6f", sp.sharingScore * 100));
            stretchScores.add(String.format("%.6f", sp.stretchScore * 100));
            localityScores.add(String.format("%.6f", sp.localityScore * 100));
        }
        
        // Create output line
        return String.format("%s,%s,\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",%s,%s,%.6f",
            row[0], row[1],
            String.join(";", pathStrs),
            String.join(";", finalScores),
            String.join(";", sharingScores),
            String.join(";", stretchScores),
            String.join(";", localityScores),
            bucketId,
            row[4], // optimal_edges
            optimalDistance
        );
    }
    
    private static double calculatePathDistance(List<Integer> path, EdgeWeightedGraph G) {
        double distance = 0;
        for (int i = 0; i < path.size() - 1; i++) {
            distance += getEdgeWeight(G, path.get(i), path.get(i + 1));
        }
        return distance;
    }
    
    private static List<String[]> loadRawPaths(String filename) {
        List<String[]> data = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            reader.readLine(); // Skip header
            String line;
            while ((line = reader.readLine()) != null) {
                // Handle quoted paths field
                String[] parts = line.split(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", -1);
                parts[2] = parts[2].replaceAll("^\"|\"$", ""); // Remove quotes
                data.add(parts);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return data;
    }
    
    private static void saveResults(ConcurrentHashMap<Integer, String> results, 
                                  int totalCount, String filename) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            // Header
            writer.write("source,destination,paths,final_scores,sharing_scores,stretch_scores,locality_scores,bucket_id,optimal_edges,optimal_distance");
            writer.newLine();
            
            // Write results in order
            for (int i = 0; i < totalCount; i++) {
                String line = results.get(i);
                if (line != null) {
                    writer.write(line);
                    writer.newLine();
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static double calculateSharingScore(List<Integer> path, List<Integer> optimalPath, EdgeWeightedGraph G) {
        Set<String> optimalEdges = new HashSet<>();

        for (int i = 0; i < optimalPath.size() - 1; i++) {
            int u = optimalPath.get(i);
            int v = optimalPath.get(i + 1);
            String edge = Math.min(u, v) + "-" + Math.max(u, v);
            optimalEdges.add(edge);
        }

        double sharedLength = 0;
        double pathLength = 0;

        for (int i = 0; i < path.size() - 1; i++) {
            int u = path.get(i);
            int v = path.get(i + 1);
            String edge = Math.min(u, v) + "-" + Math.max(u, v);
            double edgeWeight = getEdgeWeight(G, u, v);

            pathLength += edgeWeight;

            if (optimalEdges.contains(edge)) {
                sharedLength += edgeWeight;
            }
        }

        double sharingRatio = sharedLength / pathLength;

        return 1.0 - sharingRatio;
    }

    private static double[] calculateStretchAndLocalityScores(List<Integer> path, EdgeWeightedGraph G, 
                                                             double T,
                                                             boolean useMaxStretch) {
        double epsilon = 0.1;
        double maxAllowedStretch = 1.0 + epsilon;
        double totalSubpathLength = 0.0;
        double satisfyingUBSLength = 0.0;
        double totalCheckedLength = 0.0;
        double locallyOptimalLength = 0.0;
        double maxStretch = 1.0;

        for (int start = 0; start < path.size() - 1; start++) {
            for (int end = start + 2; end <= path.size(); end++) {
                double subpathLength = 0;
                for (int i = start; i < end - 1; i++) {
                    subpathLength += getEdgeWeight(G, path.get(i), path.get(i + 1));
                }

                double shortestLength = getShortestDistance(G, path.get(start), path.get(end - 1));

                if (shortestLength > 0 && shortestLength < Double.POSITIVE_INFINITY) {
                    double stretch = subpathLength / shortestLength;
                    maxStretch = Math.max(maxStretch, stretch);
                    totalSubpathLength += subpathLength;
                    if (stretch <= maxAllowedStretch) {
                        satisfyingUBSLength += subpathLength;
                    }

                    if (subpathLength <= T) {
                        totalCheckedLength += subpathLength;
                        if (Math.abs(subpathLength - shortestLength) < 1e-6) {
                            locallyOptimalLength += subpathLength;
                        }
                    } else if (end - start > 2) {
                        double firstEdgeLength = getEdgeWeight(G, path.get(start), path.get(start + 1));
                        double lastEdgeLength = getEdgeWeight(G, path.get(end - 2), path.get(end - 1));
                        double innerLength = subpathLength - firstEdgeLength - lastEdgeLength;

                        if (innerLength < T) {
                            totalCheckedLength += subpathLength;
                            if (Math.abs(subpathLength - shortestLength) < 1e-6) {
                                locallyOptimalLength += subpathLength;
                            }
                        }
                    }
                }
            }
        }

        double stretchScore = useMaxStretch
            ? 1.0 / maxStretch
            : (totalSubpathLength > 0 ? satisfyingUBSLength / totalSubpathLength : 1.0);
        double localityScore = (totalCheckedLength > 0) ? 
            locallyOptimalLength / totalCheckedLength : 1.0;
        return new double[]{stretchScore, localityScore};
    }
    
    private static double getEdgeWeight(EdgeWeightedGraph G, int u, int v) {
        for (Edge e : G.adj(u)) {
            if (e.other(u) == v) {
                return e.weight();
            }
        }
        return Double.POSITIVE_INFINITY;
    }
}