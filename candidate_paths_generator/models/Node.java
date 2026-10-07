package models;

public class Node implements Comparable<Node> {
    private double k;  // Changed from int to double for distance
    private int v;     // vertex ID remains int

    public Node() {
        this.k = 0.0;
        this.v = 0;
    }

    public Node(double key, int value) {  // Changed parameter from int to double
        this.k = key;
        this.v = value;
    }

    public double getK() {  // Changed return type to double
        return k;
    }

    public int getV() {
        return v;
    }

    @Override
    // Increasing order (for min-heap in Dijkstra)
    public int compareTo(Node that) {
        return Double.compare(this.k, that.k);  // Changed to Double.compare
    }

    public String toString() {
        return String.format("%.3f-%d", k, v);  // Changed to show decimal
    }
}