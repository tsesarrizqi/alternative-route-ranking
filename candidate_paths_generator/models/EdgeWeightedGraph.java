package models;

// import org.springframework.context.ApplicationContext;
// import org.springframework.context.support.ClassPathXmlApplicationContext;
// import org.springframework.core.io.ClassPathResource;
// import org.springframework.core.io.FileSystemResource;
// import org.springframework.core.io.Resource;
// import org.w3c.dom.Node;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.PriorityQueue;

public class EdgeWeightedGraph {
    private int V;
    private int E;
    private LatLng[] coo;
    private Bag<Edge>[] adj;
    private Bag<Edge>[] inc;

    private static final int rNum = 40;
    double[] verArr = new double[rNum + 1];
    double[] horArr = new double[rNum + 1];
    private List<List<Integer>> tiles;

    public EdgeWeightedGraph(boolean t) {
        this.loadGraph("conf/bangalore_converted_edges.txt");
    }

    /** Loads edges only (no coordinate/tile data). Sufficient for path scoring (Step 2). */
    public EdgeWeightedGraph(String edgesFile) {
        this.loadGraph(edgesFile);
    }

    public EdgeWeightedGraph() {
        this.loadGraph("conf/bangalore_converted_edges.txt");
        this.loadCoordinates("conf/bangalore_converted_coordinates.txt");
    }

    /**
     * Construct the graph from explicit file paths.
     * The bounding box is auto-computed from the coordinates file.
     */
    public EdgeWeightedGraph(String edgesFile, String coordinatesFile) {
        this.loadGraph(edgesFile);
        this.loadCoordinates(coordinatesFile);
    }

    public EdgeWeightedGraph(File file) {
        String line = null;

        try {
            FileReader fileReader = new FileReader(file);
            BufferedReader bufferedReader = new BufferedReader(fileReader);
            this.V = Integer.parseInt(bufferedReader.readLine().trim());
            this.E = Integer.parseInt(bufferedReader.readLine().trim());

            adj = (Bag<Edge>[]) new Bag[V + 1];
            for (int v = 1; v <= V; v++) {
                adj[v] = new Bag<Edge>();
            }


            while ((line = bufferedReader.readLine()) != null) {
                String[] fields = line.split(" ");
                int v = Integer.parseInt(fields[1].trim());
                int w = Integer.parseInt(fields[2].trim());
                double weight = Double.parseDouble(fields[3].trim());
                Edge e = new Edge(v, w, weight);
                addEdge(e);
                //System.out.println(line);
            }
            bufferedReader.close();
        } catch (FileNotFoundException ex) {
            System.out.println("Unable to open file");
        } catch (IOException ex) {
            System.out.println("Error reading file");
        }
    }

    public EdgeWeightedGraph(In in) {
        String line = null;
        if (in == null)
            throw new IllegalArgumentException("argument is null");
        try {
            this.V = in.readInt();
            if (V < 0)
                throw new IllegalArgumentException("number of vertices in a Digraph must be non-negative");
            adj = (Bag<Edge>[]) new Bag[V + 1];
            inc = (Bag<Edge>[]) new Bag[V + 1];
            for (int v = 1; v <= V; v++) {
                adj[v] = new Bag<Edge>();
                inc[v] = new Bag<Edge>();
            }

            int E = in.readInt();
            if (E < 0)
                throw new IllegalArgumentException("Number of edges must be non-negative");
            for (int i = 0; i < E; i++) {
                int v = in.readInt();
                int w = in.readInt();
                double weight = in.readDouble();  // was: int weight = in.readInt();
                addEdge(new Edge(v, w, weight));
            }

        } catch (NoSuchElementException e) {
            throw new IllegalArgumentException("invalid input format in EdgeWeightedDigraph constructor", e);
        }
    }

    public int V() {
        return V;
    }

    public int E() {
        return E;
    }

    public void addEdge(Edge e) {
        int v = e.either();
        int w = e.other(v);
        adj[v].add(e);       // outgoing
        inc[w].add(e);      // incoming    s ---  v --> w --- t

        // Keep directed edges in the outgoing adjacency list only.
        E++;
    }

    public Iterable<Edge> adj(int v) {
        return adj[v];
    }

    public Iterable<Edge> inc(int v) {
        return inc[v];
    }

    private void loadGraph(String filePath) {
        String line = null;
        File file = new File(filePath);

        try {
            FileReader fileReader = new FileReader(file);
            BufferedReader bufferedReader = new BufferedReader(fileReader);
            this.V = Integer.parseInt(bufferedReader.readLine().trim());
            bufferedReader.readLine();
            this.E = 0;

            adj = (Bag<Edge>[]) new Bag[V + 1];
            inc = (Bag<Edge>[]) new Bag[V + 1];
            for (int v = 1; v <= V; v++) {
                adj[v] = new Bag<Edge>();
                inc[v] = new Bag<Edge>();
            }

            while ((line = bufferedReader.readLine()) != null) {
                String[] fields = line.split(" ");
                int v = Integer.parseInt(fields[1].trim());
                int w = Integer.parseInt(fields[2].trim());
                double weight = Double.parseDouble(fields[3].trim());  // was: Integer.parseInt(fields[3].trim())
                Edge e = new Edge(v, w, weight);
                addEdge(e);
            }
            bufferedReader.close();
        } catch (FileNotFoundException ex) {
            System.out.println("Unable to open file");
        } catch (IOException ex) {
            System.out.println("Error reading file");
        }
    }

    private void loadCoordinates(String filePath) {
        File file = new File(filePath);
        String line = null;

        // First pass: collect all coordinates and compute bounding box
        List<double[]> rawCoords = new ArrayList<>();
        double minLat = Double.POSITIVE_INFINITY, maxLat = Double.NEGATIVE_INFINITY;
        double minLng = Double.POSITIVE_INFINITY, maxLng = Double.NEGATIVE_INFINITY;
        try {
            BufferedReader br = new BufferedReader(new FileReader(file));
            while ((line = br.readLine()) != null) {
                String[] fields = line.trim().split(" ");
                double lat = Double.parseDouble(fields[2]);
                double lng = Double.parseDouble(fields[3]);
                rawCoords.add(new double[]{lat, lng});
                if (lat < minLat) minLat = lat;
                if (lat > maxLat) maxLat = lat;
                if (lng < minLng) minLng = lng;
                if (lng > maxLng) maxLng = lng;
            }
            br.close();
        } catch (IOException ex) {
            System.out.println("Error reading coordinates file: " + ex.getMessage());
            return;
        }

        // Add a small margin so boundary nodes don't fall outside the grid
        double margin = 0.001;
        double bottom = minLat - margin, top = maxLat + margin;
        double left   = minLng - margin, right = maxLng + margin;

        double vertical   = (top - bottom) / rNum;
        double horizontal = (right - left) / rNum;
        verArr = new double[rNum + 1];
        horArr = new double[rNum + 1];
        for (int i = 0; i <= rNum; i++) {
            verArr[i] = bottom + vertical * i;
            horArr[i] = left + horizontal * i;
        }
        tiles = new ArrayList<List<Integer>>(rNum * rNum);
        for (int i = 0; i < rNum * rNum; i++) {
            tiles.add(new ArrayList<Integer>());
        }

        // Second pass: assign nodes to tiles
        coo = new LatLng[V + 1];
        coo[0] = new LatLng(-1.0, 1.0);
        for (int k = 0; k < rawCoords.size(); k++) {
            double lat = rawCoords.get(k)[0];
            double lng = rawCoords.get(k)[1];
            int renIdx = getRenIdx(lat, lng);
            tiles.get(renIdx).add(k + 1);
            coo[k + 1] = new LatLng(lat, lng);
        }
    }

    public LatLng getCoordinate(int v) {
        return coo[v];
    }

    public String getCooString(int v) {
        return coo[v].getLat() + "_" + coo[v].getLng();
    }

    public int getNearestNode(String latLng) {
        String[] lalo = latLng.trim().split("_");
        double distance = Double.POSITIVE_INFINITY;
        double currentDis = 0.0;
        int nodeId = 0;

        int rIdx = getRenIdx(Double.parseDouble(lalo[0]), Double.parseDouble(lalo[1]));
        List<Integer> subTiles = tiles.get(rIdx);

        if (subTiles.isEmpty()) {
            // Optimise this code
            int fIdx = rIdx + 1;
            while (fIdx < tiles.size()) {
                if (!tiles.get(fIdx).isEmpty()) {
                    subTiles = tiles.get(fIdx);
                    break;
                }
                fIdx++;
            }
            if (subTiles.isEmpty()) {
                int bIdx = rIdx - 1;
                while (bIdx > 0) {
                    if (!tiles.get(bIdx).isEmpty()) {
                        subTiles = tiles.get(bIdx);
                        break;
                    }
                    bIdx--;
                }
            }
        }

        for (Integer i : subTiles) {
            currentDis = getEDist(Double.parseDouble(lalo[0]), Double.parseDouble(lalo[1]), coo[i].getLat(), coo[i].getLng());
            if (currentDis < distance) {
                distance = currentDis;
                nodeId = i;
            }
        }

        return nodeId;
    }

    public PriorityQueue<Node> getKNN(String latLng, int n) {
        int currentDis = 0;
        String[] lalo = latLng.trim().split("_");
        PriorityQueue<Node> pq = new  PriorityQueue<>();

        int rIdx = getRenIdx(Double.parseDouble(lalo[0]), Double.parseDouble(lalo[1]));
        List<Integer> subTiles = tiles.get(rIdx);

        if (subTiles.isEmpty()) {
            // Optimise this code
            int fIdx = rIdx + 1;
            while (fIdx < tiles.size()) {
                if (!tiles.get(fIdx).isEmpty()) {
                    subTiles = tiles.get(fIdx);
                    break;
                }
                fIdx++;
            }
            if (subTiles.isEmpty()) {
                int bIdx = rIdx - 1;
                while (bIdx > 0) {
                    if (!tiles.get(bIdx).isEmpty()) {
                        subTiles = tiles.get(bIdx);
                        break;
                    }
                    bIdx--;
                }
            }
        }

        for (Integer i : subTiles) {
            currentDis = (int)getEDist(Double.parseDouble(lalo[0]), Double.parseDouble(lalo[1]), coo[i].getLat(), coo[i].getLng());
            if (pq.size() < n) {
                pq.add(new Node(currentDis, i));
            } else {
                assert pq.peek() != null;
                if (currentDis < pq.peek().getK()) {
                    pq.poll();
                    pq.add(new Node(currentDis, i));
                }
            }
        }
        return pq;
    }

    private int getRenIdx(double lat, double lng) {
        int verIdx = 0, horIdx = 0;
        for (int i = 0; i <= rNum; i++)
            if (Double.compare(lat, verArr[i]) <= 0) {
                verIdx = i == 0 ? 0 : (i - 1);
                break;
            }
        for (int i = 0; i <= rNum; i++)
            if (Double.compare(lng, horArr[i]) <= 0) {
                horIdx = i == 0 ? 0 : (i - 1);
                break;
            }
        //return horIdx * rNum + verIdx;
        return verIdx * rNum + horIdx;
    }

    private double getEDist(double lat1, double lon1, double lat2, double lon2) {
        double theta = lon1 - lon2;
        double dist = Math.sin(deg2rad(lat1)) * Math.sin(deg2rad(lat2)) + Math.cos(deg2rad(lat1)) * Math.cos(deg2rad(lat2)) * Math.cos(deg2rad(theta));
        dist = Math.acos(dist);
        dist = rad2deg(dist);
        dist = dist * 60 * 1.1515;
        dist = dist * 1.609344;

        //return (dist);             // return Kilometer
        return (dist * 10000);       // COL data
        //return (dist * 100000);
    }

    // This function converts decimal degrees to radians
    private double deg2rad(double deg) {
        return (deg * Math.PI / 180.0);
    }

    // This function converts radians to decimal degrees
    private double rad2deg(double rad) {
        return (rad * 180 / Math.PI);
    }

    public static void main(String[] args) {
    }
}
