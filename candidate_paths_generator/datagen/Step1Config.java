package datagen;

import java.io.*;
import java.util.*;

/**
 * Configuration class for Step1_PathGenerator
 * Loads parameters from config.json and provides runtime-updatable configuration
 */
public class Step1Config {
    // Graph files
    public String edgesFile;
    public String coordinatesFile;

    // Algorithm parameters
    public int tk;
    public double ub;
    public double pen;
    public double sim;
    public double dm;

    // Generation parameters
    public int totalPairs;
    public double minDistance;
    public double maxDistance;
    public int minEdges;
    public int maxEdges;
    public int minPathAlternatives;
    public int numThreads;

    // Distance buckets
    public double[][] distanceBuckets;

    // Computed values
    public int pairsPerBucket;

    /**
     * Load configuration from file
     */
    public static Step1Config load(String configPath) {
        Step1Config config = new Step1Config();

        try (BufferedReader reader = new BufferedReader(new FileReader(configPath))) {
            StringBuilder json = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                json.append(line.trim());
            }

            String content = json.toString();

            // Parse graph file paths
            config.edgesFile = parseStringValue(content, "edges_file");
            config.coordinatesFile = parseStringValue(content, "coordinates_file");

            // Parse simple key-value pairs
            config.tk = parseIntValue(content, "tk");
            config.ub = parseDoubleValue(content, "ub");
            config.pen = parseDoubleValue(content, "pen");
            config.sim = parseDoubleValue(content, "sim");
            config.dm = parseDoubleValue(content, "dm");
            config.totalPairs = parseIntValue(content, "total_pairs");
            config.minDistance = parseDoubleValue(content, "min_distance");
            config.maxDistance = parseDoubleValue(content, "max_distance");
            config.minEdges = parseIntValue(content, "min_edges");
            config.maxEdges = parseIntValue(content, "max_edges");
            config.minPathAlternatives = parseIntValue(content, "min_path_alternatives");
            config.numThreads = parseIntValue(content, "num_threads");

            // Parse distance buckets array
            config.distanceBuckets = parseDistanceBuckets(content);

            // Compute derived values
            config.pairsPerBucket = config.totalPairs / config.distanceBuckets.length;

            System.out.println("Configuration loaded from: " + configPath);
            System.out.println("  Edges file: " + config.edgesFile);
            System.out.println("  Coordinates file: " + config.coordinatesFile);
            System.out.println("  TK=" + config.tk + ", UB=" + config.ub + ", PEN=" + config.pen +
                             ", SIM=" + config.sim + ", DM=" + config.dm);
            System.out.println("  Total pairs=" + config.totalPairs + ", Buckets=" + config.distanceBuckets.length);

        } catch (IOException e) {
            System.err.println("Error loading config: " + e.getMessage());
            System.err.println("Using default configuration");
            return getDefaultConfig();
        }

        return config;
    }

    /**
     * Get default configuration as fallback
     */
    private static Step1Config getDefaultConfig() {
        Step1Config config = new Step1Config();
        config.edgesFile = "conf/bangalore_converted_edges.txt";
        config.coordinatesFile = "conf/bangalore_converted_coordinates.txt";
        config.tk = 10;
        config.ub = 2.0;
        config.pen = 1.4;
        config.sim = 0.7;
        config.dm = 4.0;
        config.totalPairs = 10000;
        config.minDistance = 1000.0;
        config.maxDistance = 11000.0;
        config.minEdges = 3;
        config.maxEdges = 200;
        config.minPathAlternatives = 5;
        config.numThreads = 8;
        config.distanceBuckets = new double[][] {
            {1000, 2000}, {2000, 3000}, {3000, 4000}, {4000, 5000}, {5000, 6000},
            {6000, 7000}, {7000, 8000}, {8000, 9000}, {9000, 10000}, {10000, 11000}
        };
        config.pairsPerBucket = config.totalPairs / config.distanceBuckets.length;
        return config;
    }

    // Helper methods for parsing JSON
    private static String parseStringValue(String json, String key) {
        String pattern = "\"" + key + "\"\\s*:\\s*\"([^\"]+)\"";
        java.util.regex.Pattern p = java.util.regex.Pattern.compile(pattern);
        java.util.regex.Matcher m = p.matcher(json);
        if (m.find()) {
            return m.group(1);
        }
        throw new RuntimeException("Could not find string key: " + key);
    }

    private static int parseIntValue(String json, String key) {
        String pattern = "\"" + key + "\"\\s*:\\s*(\\d+)";
        java.util.regex.Pattern p = java.util.regex.Pattern.compile(pattern);
        java.util.regex.Matcher m = p.matcher(json);
        if (m.find()) {
            return Integer.parseInt(m.group(1));
        }
        throw new RuntimeException("Could not find key: " + key);
    }

    private static double parseDoubleValue(String json, String key) {
        String pattern = "\"" + key + "\"\\s*:\\s*(-?[0-9.]+)";
        java.util.regex.Pattern p = java.util.regex.Pattern.compile(pattern);
        java.util.regex.Matcher m = p.matcher(json);
        if (m.find()) {
            return Double.parseDouble(m.group(1));
        }
        throw new RuntimeException("Could not find key: " + key);
    }

    private static double[][] parseDistanceBuckets(String json) {
        List<double[]> buckets = new ArrayList<>();

        // Find the distance_buckets array
        int bucketsStart = json.indexOf("\"distance_buckets\"");
        if (bucketsStart == -1) {
            throw new RuntimeException("Could not find distance_buckets in config");
        }

        int arrayStart = json.indexOf("[", bucketsStart);
        int arrayEnd = json.indexOf("]", arrayStart);

        if (arrayStart != -1 && arrayEnd != -1) {
            String bucketsArray = json.substring(arrayStart + 1, arrayEnd);
            String[] bucketObjects = bucketsArray.split("\\},\\s*\\{");

            for (String bucketObj : bucketObjects) {
                bucketObj = bucketObj.replace("{", "").replace("}", "");

                String[] parts = bucketObj.split(",");
                double min = 0, max = 0;

                for (String part : parts) {
                    part = part.trim();
                    if (part.contains("\"min\"")) {
                        min = Double.parseDouble(part.split(":")[1].trim());
                    } else if (part.contains("\"max\"")) {
                        max = Double.parseDouble(part.split(":")[1].trim());
                    }
                }

                buckets.add(new double[]{min, max});
            }
        }

        return buckets.toArray(new double[0][]);
    }
}
