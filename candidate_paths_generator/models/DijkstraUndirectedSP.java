package models;

// import org.springframework.core.io.ClassPathResource;
// import org.springframework.core.io.Resource;

import java.io.File;
import java.io.IOException;
import java.util.*;

public class DijkstraUndirectedSP {
    private double stDist;
    private double[] distTo;
    private Edge[] edgeTo;
    private PriorityQueue<Node> pq;
    private boolean[] inPQ;  // Track which vertices are in PQ

    private Stack<Integer> spTree = new Stack<Integer>();
    private int source;
    public DijkstraUndirectedSP(EdgeWeightedGraph G, int s) {
        this.source = s;
        distTo = new double[G.V() + 1];  // was: int[G.V() + 1]
        edgeTo = new Edge[G.V() + 1];
        for (int v = 0; v <= G.V(); v++)
            distTo[v] = Double.POSITIVE_INFINITY;
        distTo[s] = 0;
        edgeTo[s] = new Edge(Integer.MAX_VALUE, s, 0.0);
        pq = new PriorityQueue<Node>();
        inPQ = new boolean[G.V() + 1];
        pq.offer(new Node(distTo[s], s));
        inPQ[s] = true;
        while (!pq.isEmpty()) {
            Node node = pq.poll();
            int v = node.getV();
            inPQ[v] = false;
            if (node.getK() > distTo[v]) continue;
            
            spTree.push(v);
            for (Edge e : G.adj(v))
                relax(e, v);
        }
    }
    public DijkstraUndirectedSP(EdgeWeightedGraph G, int s, int t) {
        this.source = s;
        stDist = 0;
        distTo = new double[G.V() + 1];
        edgeTo = new Edge[G.V() + 1];
        for (int v = 0; v <= G.V(); v++)
            distTo[v] = Double.POSITIVE_INFINITY;
        distTo[s] = 0;
        edgeTo[s] = new Edge(Integer.MAX_VALUE, s, 0.0);
        pq = new PriorityQueue<Node>();
        inPQ = new boolean[G.V() + 1];
        pq.offer(new Node(distTo[s], s));
        inPQ[s] = true;
        while (!pq.isEmpty()) {
            Node node = pq.poll();
            int v = node.getV();
            inPQ[v] = false;
            if (node.getK() > distTo[v]) continue; 
            if (v == t) {
                stDist = distTo[v];
                break;
            }
            for (Edge e : G.adj(v))
                relax(e, v);
        }
    }
    public DijkstraUndirectedSP(EdgeWeightedGraph G, int s, int t, double ub) {
        this.source = s;
        stDist = 0;
        distTo = new double[G.V() + 1];
        edgeTo = new Edge[G.V() + 1];
        for (int v = 0; v <= G.V(); v++)
            distTo[v] = Double.POSITIVE_INFINITY;
        distTo[s] = 0;
        edgeTo[s] = new Edge(Integer.MAX_VALUE, s, 0.0);
        pq = new PriorityQueue<Node>();
        inPQ = new boolean[G.V() + 1];
        pq.offer(new Node(distTo[s], s));
        inPQ[s] = true;
        while (!pq.isEmpty()) {
            Node node = pq.poll();
            int v = node.getV();
            inPQ[v] = false;
            if (node.getK() > distTo[v]) continue; 
            if (v == t)
                stDist = distTo[v];
            if (stDist == 0) {
                spTree.push(v);
                for (Edge e : G.adj(v))
                    relax(e, v);
            } else if (stDist != 0 && distTo[v] < stDist * ub) {
                spTree.push(v);
                for (Edge e : G.adj(v))
                    relax(e, v);
            }
        }
    }
    public DijkstraUndirectedSP(EdgeWeightedGraph G, int s, DijkstraUndirectedSP spt) {
        distTo = new double[G.V() + 1];
        edgeTo = new Edge[G.V() + 1];
        for (int v = 1; v <= G.V(); v++)
            distTo[v] = Double.POSITIVE_INFINITY;
        distTo[s] = 0;
        edgeTo[s] = new Edge(Integer.MAX_VALUE, s, 0.0);
        pq = new PriorityQueue<Node>();
        inPQ = new boolean[G.V() + 1];
        pq.offer(new Node(distTo[s], s));
        inPQ[s] = true;
        while (!pq.isEmpty()) {
            Node node = pq.poll();
            int v = node.getV();
            inPQ[v] = false;
            if (node.getK() > distTo[v]) continue; 
            if (distTo[v] + spt.distTo[v] < spt.distTo[s] * 1.4) {
                spTree.push(v);
                for (Edge e : G.adj(v))
                    relax(e, v);
            }
        }
    }
    public DijkstraUndirectedSP(EdgeWeightedGraph G, int t, DijkstraUndirectedSP spt, boolean reverse, double ub) {
        distTo = new double[G.V() + 1];
        edgeTo = new Edge[G.V() + 1];
        for (int v = 1; v <= G.V(); v++)
            distTo[v] = Double.POSITIVE_INFINITY;
        distTo[t] = 0;
        edgeTo[t] = new Edge(Integer.MAX_VALUE, t, 0.0);
        pq = new PriorityQueue<Node>();
        inPQ = new boolean[G.V() + 1];
        pq.offer(new Node(distTo[t], t));  // Use 't' not 's'
        inPQ[t] = true;
        while (!pq.isEmpty()) {
            Node node = pq.poll();
            int v = node.getV();
            inPQ[v] = false;
            if (node.getK() > distTo[v]) continue; 
            if (distTo[v] + spt.distTo[v] < spt.distTo[t] * ub) {
                spTree.push(v);
                for (Edge e : G.inc(v))
                    reverseLax(e, v);
            }
        }
    }
    private void relax(Edge e, int v) {
        int w = e.other(v);
        if (distTo[w] > distTo[v] + e.weight()) {
            distTo[w] = distTo[v] + e.weight();
            edgeTo[w] = e;
            pq.offer(new Node(distTo[w], w));
            inPQ[w] = true;
        }
    }
    private void reverseLax(Edge e, int v) {
        int w = e.either();
        if (distTo[w] > distTo[v] + e.weight()) {
            distTo[w] = distTo[v] + e.weight();
            edgeTo[w] = e;
            pq.offer(new Node(distTo[w], w));
            inPQ[w] = true;
        }
    }
    public double distTo(int v) {
        return distTo[v];
    }
    public boolean hasPathTo(int v) {
        return distTo[v] < Double.POSITIVE_INFINITY;
    }
    public Edge getEdge(int v) {
        return edgeTo[v];
    }
    public double stDistance() {
        return stDist;
    }
    public Stack<Integer> getSpTree() {
        return spTree;
    }
    public List<Integer> pathTo(int v) {
        if (!hasPathTo(v))
            return null;
        List<Integer> path = new ArrayList<>();
        int x = v;
        while (x != source) {
            path.add(x);
            Edge e = edgeTo[x];
            if (e == null)
                break;
            int prev = (e.either() == x ? e.other(x) : e.either());
            x = prev;
        }
        path.add(source);
        Collections.reverse(path);
        return path;
    }
    public static void main(String[] args) {
        In in = new In("src/main/webapp/WEB-INF/classes/conf/tiny_tests.txt");
        EdgeWeightedGraph tg = new EdgeWeightedGraph(in);
        DijkstraUndirectedSP spt = new DijkstraUndirectedSP(tg, 1);
        DijkstraUndirectedSP tpt = new DijkstraUndirectedSP(tg, 2);
        boolean[] visited = new boolean[tg.V() + 1];
        Stack<Integer> treeNodes = tpt.getSpTree();
        double depth;
        int vid, pid, tail, head, plNodes = 1;
        while (!treeNodes.isEmpty()) {
            vid = treeNodes.pop();
            tail = head = vid;
            if (!visited[vid]) {
                visited[vid] = true;
                pid = tpt.getEdge(vid).either();
                while (pid != Integer.MAX_VALUE && !visited[pid]) {
                    if (spt.getEdge(pid).either() == vid) {
                        visited[pid] = true;
                        plNodes++;
                        if (plNodes == 2) {
                            tail = vid;
                            head = pid;
                        } else {
                            head = pid;
                        }
                        vid = pid;
                        pid = tpt.getEdge(vid).either();
                    } else
                        break;
                }
                if (plNodes >= 2) {
                    depth = tpt.distTo(tail) - tpt.distTo(head);
                    StdOut.println(tail + " --> " + head + " = " + depth);
                }
            }
            plNodes = 1;
        }
    }
}