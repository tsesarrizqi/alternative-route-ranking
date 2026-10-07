package datagen;

import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import models.Edge;
import models.EdgeWeightedGraph;
import models.DijkstraUndirectedSP;

/**
 * STEP 1: Random Walk Path Generation
 *
 * Generates diverse alternative paths using a randomized walk:
 *   - At every hop, the next edge is chosen at random from unvisited neighbors.
 *   - A configurable destination-bias makes the walk prefer nodes that are
 *     closer (Dijkstra distance) to the destination, keeping runtime bounded.
 *   - Paths that exceed the length/distance budget are discarded and retried.
 *
 * Output format is identical to Step1_PathGenerator (same CSV columns).
 * Configuration is read from the same Step1Config JSON file.
 *
 * Usage:
 *   java datagen.Step1_RandomWalkPathGenerator <output_csv> [config_json]
 *
 * INDEXING CONVENTION (same as Step1_PathGenerator):
 *   - Internal graph operations: 1-indexed
 *   - CSV output: 0-indexed (for Python compatibility)
 */
public class Step1_RandomWalkPathGenerator {

    // -----------------------------------------------------------------------
    // Random-walk specific tuning constants
    // -----------------------------------------------------------------------

    /**
     * Probability [0, 1] that the next hop is chosen by a greedy
     * "closest-to-destination" rule rather than pure uniform random.
     * Higher values converge faster but produce less diverse paths.
     */
    private static final double DEST_BIAS = 0.55;

    /**
     * Maximum number of nodes that a single random-walk attempt may visit
     * before it is aborted as "too long".  Expressed as a multiplier of the
     * optimal path's edge count.  E.g. 4 means we allow up to 4× the number
     * of edges the shortest path uses.
     */
    private static final double MAX_LENGTH_FACTOR = 4.0;

    /**
     * Fallback absolute maximum nodes per path, used when the optimal edge
     * count cannot be determined.
     */
    private static final int ABS_MAX_PATH_NODES = 200;

    /**
     * Total random-walk attempts made per OD pair while trying to collect
     * tk distinct paths.  Capped to keep latency per pair predictable.
     */
    private static final int MAX_WALK_ATTEMPTS = 300;

    // -----------------------------------------------------------------------
    // Output row container (mirrors Step1_PathGenerator.PathData)
    // -----------------------------------------------------------------------

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

    // -----------------------------------------------------------------------
    // Random-walk router
    // -----------------------------------------------------------------------

    /**
     * Generates multiple distinct paths from {@code source} to {@code dest}
     * using a randomized walk on the graph.
     *
     * <p>At each node the algorithm:
     * <ol>
     *   <li>Collects all edges to unvisited neighbors.</li>
     *   <li>With probability {@link #DEST_BIAS} picks the neighbor that
     *       minimises remaining Dijkstra distance to the destination
     *       ("greedy" move).</li>
     *   <li>Otherwise picks uniformly at random ("exploratory" move).</li>
     *   <li>Aborts and retries when the path grows beyond the length cap or
     *       reaches a dead end.</li>
     * </ol>
     */
    static class RandomWalkRouter {
        private final EdgeWeightedGraph graph;
        private final int source;
        private final int dest;
        private final int maxNodes;          // per-path node-count cap
        private final double maxDist;        // per-path distance cap
        private final Random rand;
        private final double[] distToDest;   // precomputed Dijkstra from dest

        RandomWalkRouter(EdgeWeightedGraph g,
                         int s, int t,
                         int maxNodes, double maxDist,
                         long seed,
                         DijkstraUndirectedSP destSPT) {
            this.graph    = g;
            this.source   = s;
            this.dest     = t;
            this.maxNodes = maxNodes;
            this.maxDist  = maxDist;
            this.rand     = new Random(seed);

            // Cache distTo array for O(1) heuristic lookups
            int V = g.V();
            distToDest = new double[V + 1];
            for (int v = 1; v <= V; v++) {
                distToDest[v] = destSPT.distTo(v);
            }
        }

        /**
         * Try to collect {@code numPaths} unique valid paths.
         * Returns however many were found within {@link #MAX_WALK_ATTEMPTS}.
         */
        List<List<Integer>> generatePaths(int numPaths) {
            Set<List<Integer>> found = new LinkedHashSet<>();
            int attempts = 0;

            while (found.size() < numPaths && attempts < MAX_WALK_ATTEMPTS) {
                attempts++;
                List<Integer> path = singleWalk();
                if (path != null) {
                    found.add(path);
                }
            }

            return new ArrayList<>(found);
        }

        /**
         * One random-walk attempt: returns a valid path or {@code null}.
         */
        private List<Integer> singleWalk() {
            List<Integer> path   = new ArrayList<>();
            Set<Integer>  visited = new HashSet<>();
            double        dist   = 0.0;

            int current = source;
            path.add(current);
            visited.add(current);

            while (current != dest) {
                if (path.size() >= maxNodes) return null;   // too long

                // Collect unvisited neighbors
                List<Edge> candidates = new ArrayList<>();
                for (Edge e : graph.adj(current)) {
                    int nbr = e.other(current);
                    if (!visited.contains(nbr)) {
                        candidates.add(e);
                    }
                }

                if (candidates.isEmpty()) return null;      // dead end

                // Choose next edge
                Edge chosen;
                if (rand.nextDouble() < DEST_BIAS) {
                    // Greedy: prefer neighbor with smallest remaining distance
                    chosen = greedyEdge(current, candidates);
                } else {
                    // Pure random
                    chosen = candidates.get(rand.nextInt(candidates.size()));
                }

                dist += chosen.weight();
                if (dist > maxDist) return null;            // distance budget exceeded

                int next = chosen.other(current);
                path.add(next);
                visited.add(next);
                current = next;
            }

            return path;   // successfully reached destination
        }

        /** Pick the edge whose far endpoint is closest (Dijkstra) to dest. */
        private Edge greedyEdge(int current, List<Edge> candidates) {
            Edge   best    = null;
            double bestDist = Double.POSITIVE_INFINITY;
            for (Edge e : candidates) {
                double d = distToDest[e.other(current)];
                if (d < bestDist) {
                    bestDist = d;
                    best     = e;
                }
            }
            return best != null ? best : candidates.get(rand.nextInt(candidates.size()));
        }
    }

    // -----------------------------------------------------------------------
    // Entry point
    // -----------------------------------------------------------------------

    public static void main(String[] args) throws InterruptedException {
        if (args.length < 1) {
            System.err.println(
                "Usage: java datagen.Step1_RandomWalkPathGenerator <output_csv> [config_json]");
            System.exit(1);
        }

        String outputFile = args[0];
        String configPath = args.length > 1 ? args[1] : "conf/step1_config.json";

        Step1Config config = Step1Config.load(configPath);
        EdgeWeightedGraph G = new EdgeWeightedGraph(config.edgesFile, config.coordinatesFile);
        int V = G.V();

        System.out.println("=== STEP 1: Random Walk Path Generation ===");
        System.out.println("Graph loaded: " + V + " nodes");
        System.out.println("Generating " + config.totalPairs + " OD pairs ...");
        System.out.println("Target per bucket: " + config.pairsPerBucket);
        System.out.printf("Walk config: DEST_BIAS=%.2f  MAX_LENGTH_FACTOR=%.1f  MAX_WALK_ATTEMPTS=%d%n",
            DEST_BIAS, MAX_LENGTH_FACTOR, MAX_WALK_ATTEMPTS);

        // Thread-safe result containers
        Map<Integer, List<PathData>> bucketData      = new ConcurrentHashMap<>();
        Map<Integer, Set<String>>    bucketUsedPairs = new ConcurrentHashMap<>();

        for (int i = 0; i < config.distanceBuckets.length; i++) {
            bucketData.put(i, Collections.synchronizedList(new ArrayList<>()));
            bucketUsedPairs.put(i, ConcurrentHashMap.newKeySet());
        }

        ExecutorService       executor = Executors.newFixedThreadPool(config.numThreads);
        List<Future<Integer>> futures  = new ArrayList<>();

        for (int t = 0; t < config.numThreads; t++) {
            final int tid = t;
            futures.add(executor.submit(() ->
                workerThread(G, bucketData, bucketUsedPairs, tid, config)
            ));
        }

        // Progress monitor (same cadence as Step1_PathGenerator)
        Thread monitor = new Thread(() -> {
            while (!Thread.interrupted()) {
                printProgress(bucketData, config);
                try { Thread.sleep(5000); } catch (InterruptedException e) { break; }
            }
        });
        monitor.setDaemon(true);
        monitor.start();

        for (Future<Integer> f : futures) {
            try { f.get(); } catch (Exception e) { e.printStackTrace(); }
        }

        executor.shutdown();
        monitor.interrupt();

        System.out.println("\n=== Final Results ===");
        printProgress(bucketData, config);

        saveToFile(bucketData, outputFile, config);
        System.out.println("\nPaths saved to: " + outputFile);
        System.out.println("Ready for Step 2: Scoring");
    }

    // -----------------------------------------------------------------------
    // Worker thread
    // -----------------------------------------------------------------------

    private static int workerThread(EdgeWeightedGraph G,
                                    Map<Integer, List<PathData>> bucketData,
                                    Map<Integer, Set<String>>    bucketUsedPairs,
                                    int threadId,
                                    Step1Config config) {
        Random rand  = new Random(System.currentTimeMillis() + threadId * 1_000_007L);
        int    V     = G.V();
        int    attempts  = 0;
        int    successes = 0;

        while (getTotalPairs(bucketData) < config.totalPairs && attempts < 500_000) {
            attempts++;

            // ---- 1. Random OD pair ----------------------------------------
            int s = rand.nextInt(V) + 1;
            int t = rand.nextInt(V) + 1;
            if (s == t) continue;

            String pairKey = s + "," + t;
            boolean used = false;
            for (Set<String> usedSet : bucketUsedPairs.values()) {
                if (usedSet.contains(pairKey)) { used = true; break; }
            }
            if (used) continue;

            try {
                // ---- 2. Compute shortest path (for bucket/filter only) -----
                // UB is passed as +inf here so Dijkstra explores the full graph;
                // the bound is enforced in the walk itself via the distance cap.
                DijkstraUndirectedSP srcSPT = new DijkstraUndirectedSP(G, s, t, config.ub);
                double optimalDist = srcSPT.distTo(t);
                if (optimalDist == Double.POSITIVE_INFINITY) continue;
                if (optimalDist < config.minDistance || optimalDist > config.maxDistance) continue;

                List<Integer> optimalPath  = extractOptimalPath(srcSPT, s, t);
                int           optimalEdges = optimalPath.size() - 1;
                if (optimalEdges < config.minEdges || optimalEdges > config.maxEdges) continue;

                int bucket = bucketIndex(optimalDist, config);
                if (bucket == -1) continue;
                if (bucketData.get(bucket).size() >= config.pairsPerBucket) continue;

                // ---- 3. Reverse SPT (from dest) used as heuristic -----------
                DijkstraUndirectedSP destSPT = new DijkstraUndirectedSP(G, t);

                // ---- 4. Random-walk path generation -------------------------
                int maxNodes = Math.min(
                    (int)(MAX_LENGTH_FACTOR * optimalEdges + 2),
                    ABS_MAX_PATH_NODES);
                double maxDist = optimalDist * config.ub;

                RandomWalkRouter walker = new RandomWalkRouter(
                    G, s, t,
                    maxNodes, maxDist,
                    System.currentTimeMillis() + attempts * 31L + threadId,
                    destSPT);

                List<List<Integer>> randPaths = walker.generatePaths(config.tk - 1);

                // Always include the Dijkstra-optimal path as the first entry
                Set<List<Integer>> unique = new LinkedHashSet<>();
                unique.add(optimalPath);
                unique.addAll(randPaths);

                List<List<Integer>> paths = new ArrayList<>(unique);
                if (paths.size() > config.tk) {
                    paths = paths.subList(0, config.tk);
                }

                if (paths.size() < config.minPathAlternatives) continue;

                // ---- 5. Record result ---------------------------------------
                PathData data = new PathData(
                    s, t, paths, optimalDist, optimalEdges, bucket);
                bucketData.get(bucket).add(data);
                bucketUsedPairs.get(bucket).add(pairKey);
                successes++;

                if (successes % 50 == 0) {
                    System.out.printf("Thread %d: %d pairs (%.1f%% success rate)%n",
                        threadId, successes, 100.0 * successes / attempts);
                }

            } catch (Exception ignored) {
                // skip problematic pairs silently
            }
        }

        return successes;
    }

    // -----------------------------------------------------------------------
    // I/O helpers  (same format as Step1_PathGenerator)
    // -----------------------------------------------------------------------

    private static void saveToFile(Map<Integer, List<PathData>> bucketData,
                                   String filename,
                                   Step1Config config) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            writer.write("source,destination,paths,optimal_distance,optimal_edges,bucket_id");
            writer.newLine();

            for (int b = 0; b < config.distanceBuckets.length; b++) {
                for (PathData data : bucketData.get(b)) {
                    List<String> pathStrings = new ArrayList<>();
                    for (List<Integer> path : data.paths) {
                        List<String> nodeIds = new ArrayList<>();
                        for (int node : path) {
                            nodeIds.add(String.valueOf(node - 1)); // 1→0 indexed
                        }
                        pathStrings.add(String.join(",", nodeIds));
                    }
                    writer.write(String.format("%d,%d,\"%s\",%.2f,%d,%d",
                        data.source      - 1,
                        data.destination - 1,
                        String.join(";", pathStrings),
                        data.optimalDistance,
                        data.optimalPathEdges,
                        data.bucketId));
                    writer.newLine();
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // -----------------------------------------------------------------------
    // Utility helpers
    // -----------------------------------------------------------------------

    private static void printProgress(Map<Integer, List<PathData>> bucketData,
                                      Step1Config config) {
        System.out.println("\n--- Progress ---");
        int total = 0;
        for (int i = 0; i < config.distanceBuckets.length; i++) {
            int cnt = bucketData.get(i).size();
            total += cnt;
            System.out.printf("Bucket %d [%.0f-%.0f km]: %d/%d (%.1f%%)%n",
                i,
                config.distanceBuckets[i][0] / 1000,
                config.distanceBuckets[i][1] / 1000,
                cnt, config.pairsPerBucket,
                100.0 * cnt / config.pairsPerBucket);
        }
        System.out.println("Total: " + total + "/" + config.totalPairs);
    }

    private static int getTotalPairs(Map<Integer, List<PathData>> bucketData) {
        int total = 0;
        for (List<PathData> list : bucketData.values()) total += list.size();
        return total;
    }

    private static int bucketIndex(double distance, Step1Config config) {
        for (int i = 0; i < config.distanceBuckets.length; i++) {
            if (distance >= config.distanceBuckets[i][0]
                    && distance < config.distanceBuckets[i][1]) {
                return i;
            }
        }
        return -1;
    }

    private static List<Integer> extractOptimalPath(DijkstraUndirectedSP spt, int s, int t) {
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
