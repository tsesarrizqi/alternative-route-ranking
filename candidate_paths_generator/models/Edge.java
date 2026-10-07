package models;

public class Edge implements Comparable<Edge> {

    private final int v;
    private final int w;
    private double weight;  // Changed from int to double

    public Edge() {
        this.v = 0;
        this.w = 0;
        this.weight = 0.0;  // Changed to 0.0
    }

    public Edge(int v, int w, double weight) {  // Changed parameter from int to double
        if (v < 0) throw new IllegalArgumentException("vertex index must be a nonnegative integer");
        if (w < 0) throw new IllegalArgumentException("vertex index must be a nonnegative integer");
        if (Double.isNaN(weight)) throw new IllegalArgumentException("Weight is NaN");
        this.v = v;
        this.w = w;
        this.weight = weight;
    }

    public double weight() {  // Changed return type from int to double
        return weight;
    }

    public void setWeight(double nw) {  // Changed parameter from int to double
        this.weight = nw;
    }

    public int either() {
        return v;
    }

    public int tail() {
        return v;
    }

    public int head() {
        return w;
    }

    public int other(int vertex) {
        if      (vertex == v) return w;
        else if (vertex == w) return v;
        else return -1;
    }

    /**
     * Compares two edges by weight.
     * Note that {@code compareTo()} is not consistent with {@code equals()},
     * which uses the reference equality implementation inherited from {@code Object}.
     *
     * @param  that the other edge
     * @return a negative integer, zero, or positive integer depending on whether
     *         the weight of this is less than, equal to, or greater than the
     *         argument edge
     */
    @Override
    // Decreasing order
    public int compareTo(Edge that) {
        return Double.compare(that.weight, this.weight);  // Changed to use Double.compare
    }

    /**
     * Returns a string representation of this edge.
     *
     * @return a string representation of this edge
     */
    public String toString() {
        return String.format("%d-%d %.3f", v, w, weight);  // Changed to show decimal places
    }

    @Override
    public boolean equals(Object o) {
        // If the object is compared with itself then return true
        if (o == this)
            return true;

        /* Check if o is an instance of Edge or not
          "null instanceof [type]" also returns false */
        if (!(o instanceof Edge)) {
            return false;
        }

        // typecast o to Edge so that we can compare data members
        Edge edge = (Edge) o;

        // Compare the data members and return accordingly
        return this.v == edge.either() && this.w == edge.other(edge.either())
                || this.w == edge.either() && this.v == edge.other(edge.either());

    }

    public static void main(String[] args) {
        Edge e1 = new Edge(12, 34, 5.678);  // Updated to use double
        Edge e2 = new Edge(12, 34, 5.678);
        Edge e3 = new Edge(34, 12, 9.123);
        Edge e4 = null;

        if (e2.equals(e3)) {
            StdOut.println("They are equal!");
        } else {
            StdOut.println("Not equal!");
        }

        StdOut.println(e1);
        e1.setWeight(10.456);  // Updated to use double
        StdOut.println(e1);
    }
}