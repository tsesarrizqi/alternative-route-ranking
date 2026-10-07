package datagen;

import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import models.Edge;
import models.EdgeWeightedGraph;
import models.DijkstraUndirectedSP;

/**
 * STEP 1: Stochastic Path Generation
 * Based on simulation-based path generation (Nielsen 2000, Frejinger 2007)
 * Perturbs edge costs with Gumbel noise and re-runs Dijkstra to get diverse paths.
 *
 * Usage:
 *   java datagen.Step1_StochasticPathGenerator <output_csv> [config_json] [mu]
 *
 * INDEXING CONVENTION:
 * - Internal: 1-indexed (EdgeWeightedGraph uses 1 to V)
 * - CSV output: 0-indexed (for Python compatibility)
 */
public class Step1_StochasticPathGenerator {

    /**
     * Gumbel perturbation strength. Higher = more path diversity, less optimality.
     * Typical range: 0.5 (mild) – 3.0 (aggressive). Default: 1.5.
     * Can be overridden via the third CLI argument.
     */
    private static double MU = 1.5;
    
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

    static class StochasticRouter {
        private EdgeWeightedGraph graph;
        private int source;
        private int destination;
        private double maxDistance;
        private Random rand;
        
        public StochasticRouter(EdgeWeightedGraph g, int s, int t, double maxDist, long seed) {
            this.graph = g;
            this.source = s;
            this.destination = t;
            this.maxDistance = maxDist;
            this.rand = new Random(seed);
        }
        
        public List<List<Integer>> generatePaths(int numPaths) {
            Set<List<Integer>> uniquePaths = new LinkedHashSet<>();
            int attempts = 0;
            int maxAttempts = numPaths * 20;
            
            while (uniquePaths.size() < numPaths && attempts < maxAttempts) {
                attempts++;
                
                List<Integer> path = findPerturbedShortestPath();
                
                if (path != null && path.size() >= 2) {
                    uniquePaths.add(path);
                }
            }
            
            return new ArrayList<>(uniquePaths);
        }
        
        private List<Integer> findPerturbedShortestPath() {
            Map<Edge, Double> perturbationMap = new HashMap<>();
            
            double[] distTo = new double[graph.V() + 1];
            Edge[] edgeTo = new Edge[graph.V() + 1];
            boolean[] inPQ = new boolean[graph.V() + 1];
            PriorityQueue<Node> pq = new PriorityQueue<>();
            
            for (int v = 1; v <= graph.V(); v++) {
                distTo[v] = Double.POSITIVE_INFINITY;
            }
            
            distTo[source] = 0;
            pq.offer(new Node(0, source));
            inPQ[source] = true;
            
            while (!pq.isEmpty()) {
                Node node = pq.poll();
                int v = node.v;
                inPQ[v] = false;
                
                if (node.dist > distTo[v]) continue;
                if (distTo[v] > maxDistance) continue;
                if (v == destination) break;
                
                for (Edge e : graph.adj(v)) {
                    int w = e.other(v);
                    
                    double perturbedWeight = perturbationMap.computeIfAbsent(e, 
                        edge -> edge.weight() * Math.exp(generateGumbel() * MU));
                    
                    if (distTo[w] > distTo[v] + perturbedWeight) {
                        distTo[w] = distTo[v] + perturbedWeight;
                        edgeTo[w] = e;
                        
                        if (!inPQ[w]) {
                            pq.offer(new Node(distTo[w], w));
                            inPQ[w] = true;
                        }
                    }
                }
            }
            
            if (distTo[destination] == Double.POSITIVE_INFINITY || distTo[destination] > maxDistance) {
                return null;
            }
            
            return extractPath(edgeTo, source, destination);
        }
        
        private double generateGumbel() {
            double u = rand.nextDouble();
            while (u == 0) u = rand.nextDouble();
            return -Math.log(-Math.log(u));
        }
        
        private List<Integer> extractPath(Edge[] edgeTo, int s, int t) {
            List<Integer> path = new ArrayList<>();
            int current = t;
            
            while (current != s && current != Integer.MAX_VALUE) {
                path.add(current);
                Edge e = edgeTo[current];
                if (e == null) break;
                current = e.either() == current ? e.other(current) : e.either();
            }
            
            if (current == s) {
                path.add(s);
                Collections.reverse(path);
            }
            
            return path;
        }
        
        static class Node implements Comparable<Node> {
            double dist;
            int v;
            
            Node(double dist, int v) {
                this.dist = dist;
                this.v = v;
            }
            
            public int compareTo(Node other) {
                return Double.compare(this.dist, other.dist);
            }
        }
    }

    public static void main(String[] args) throws InterruptedException {
        if (args.length < 1) {
            System.err.println(
                "Usage: java datagen.Step1_StochasticPathGenerator <output_csv> [config_json] [mu]");
            System.exit(1);
        }

        String outputFile = args[0];
        String configPath = args.length > 1 ? args[1] : "conf/step1_config.json";
        if (args.length > 2) {
            MU = Double.parseDouble(args[2]);
        }

        Step1Config config = Step1Config.load(configPath);
        EdgeWeightedGraph G = new EdgeWeightedGraph(config.edgesFile, config.coordinatesFile);
        int V = G.V();

        System.out.println("=== STEP 1: Stochastic Path Generation ===");
        System.out.println("Graph loaded: " + V + " nodes");
        System.out.println("Generating " + config.totalPairs + " OD pairs with stochastic paths...");
        System.out.println("Perturbation parameter: " + MU);
        System.out.println("Target per bucket: " + config.pairsPerBucket);

        Map<Integer, List<PathData>> bucketData = new ConcurrentHashMap<>();
        Map<Integer, Set<String>> bucketUsedPairs = new ConcurrentHashMap<>();

        for (int i = 0; i < config.distanceBuckets.length; i++) {
            bucketData.put(i, Collections.synchronizedList(new ArrayList<>()));
            bucketUsedPairs.put(i, ConcurrentHashMap.newKeySet());
        }

        ExecutorService executor = Executors.newFixedThreadPool(config.numThreads);
        List<Future<Integer>> futures = new ArrayList<>();

        for (int t = 0; t < config.numThreads; t++) {
            final int threadId = t;
            futures.add(executor.submit(() ->
                generateStochasticPaths(G, bucketData, bucketUsedPairs, threadId, config)
            ));
        }

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
        monitor.setDaemon(true);
        monitor.start();

        for (Future<Integer> future : futures) {
            try {
                future.get();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        executor.shutdown();
        monitor.interrupt();

        System.out.println("\n=== Final Results ===");
        printProgress(bucketData, config);

        saveToFile(bucketData, outputFile, config);
        System.out.println("\nStochastic paths saved to: " + outputFile);
        System.out.println("Ready for Step 2: Scoring");
    }
    
    private static int generateStochasticPaths(EdgeWeightedGraph G,
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

            int s = rand.nextInt(V) + 1;
            int t = rand.nextInt(V) + 1;
            if (s == t) continue;

            String pairKey = s + "," + t;

            boolean used = false;
            for (Set<String> usedSet : bucketUsedPairs.values()) {
                if (usedSet.contains(pairKey)) {
                    used = true;
                    break;
                }
            }
            if (used) continue;

            try {
                DijkstraUndirectedSP spt = new DijkstraUndirectedSP(G, s, t, config.ub);
                if (spt.distTo(t) == Double.POSITIVE_INFINITY) continue;

                double optimalDistance = spt.distTo(t);
                if (optimalDistance < config.minDistance || optimalDistance > config.maxDistance) continue;

                List<Integer> optimalPath = getOptimalPath(spt, s, t);
                int optimalPathEdges = optimalPath.size() - 1;
                if (optimalPathEdges < config.minEdges || optimalPathEdges > config.maxEdges) continue;

                int bucket = getDistanceBucketIndex(optimalDistance, config);
                if (bucket == -1) continue;

                if (bucketData.get(bucket).size() >= config.pairsPerBucket) continue;

                StochasticRouter router = new StochasticRouter(G, s, t,
                    optimalDistance * config.ub, System.currentTimeMillis() + attempts);
                List<List<Integer>> paths = router.generatePaths(config.tk - 1);

                paths.add(0, optimalPath);
                Set<List<Integer>> uniquePaths = new LinkedHashSet<>(paths);
                paths = new ArrayList<>(uniquePaths);

                if (paths.size() > config.tk) {
                    paths = paths.subList(0, config.tk);
                }

                if (paths.size() < config.minPathAlternatives) continue;

                PathData data = new PathData(s, t, paths, optimalDistance, optimalPathEdges, bucket);
                bucketData.get(bucket).add(data);
                bucketUsedPairs.get(bucket).add(pairKey);
                successes++;

                if (successes % 50 == 0) {
                    System.out.printf("Thread %d: %d pairs found (%.1f%% success rate)%n",
                        threadId, successes, 100.0 * successes / attempts);
                }

            } catch (Exception ignored) {
            }
        }

        return successes;
    }
    
    private static void saveToFile(Map<Integer, List<PathData>> bucketData,
                                    String filename,
                                    Step1Config config) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            writer.write("source,destination,paths,optimal_distance,optimal_edges,bucket_id");
            writer.newLine();

            for (int bucket = 0; bucket < config.distanceBuckets.length; bucket++) {
                for (PathData data : bucketData.get(bucket)) {
                    List<String> pathStrings = new ArrayList<>();
                    for (List<Integer> path : data.paths) {
                        List<String> nodeIds = new ArrayList<>();
                        for (int node : path) {
                            nodeIds.add(String.valueOf(node - 1)); // 1→0 indexed
                        }
                        pathStrings.add(String.join(",", nodeIds));
                    }
                    writer.write(String.format("%d,%d,\"%s\",%.2f,%d,%d",
                        data.source - 1,
                        data.destination - 1,
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

    private static void printProgress(Map<Integer, List<PathData>> bucketData,
                                      Step1Config config) {
        System.out.println("\n--- Progress ---");
        int total = 0;
        for (int i = 0; i < config.distanceBuckets.length; i++) {
            int count = bucketData.get(i).size();
            total += count;
            System.out.printf("Bucket %d [%.0f-%.0f km]: %d/%d (%.1f%%)%n",
                i,
                config.distanceBuckets[i][0] / 1000,
                config.distanceBuckets[i][1] / 1000,
                count, config.pairsPerBucket,
                100.0 * count / config.pairsPerBucket);
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