//package models;
//
//import java.util.*;
//
//public class Plateaus {
//    private int s;
//    private int t;
//    private int TK;
//    private double UB;
//    private double PNT;
//    private double SIM;
//    private EdgeWeightedGraph tg;
//    private DijkstraUndirectedSP spt;
//    private DijkstraUndirectedSP tpt;
//
//    private PriorityQueue<Node> viaNodes = new PriorityQueue<Node>();
//
//
//    public Plateaus(int s, int t, EdgeWeightedGraph G, DijkstraUndirectedSP spt, DijkstraUndirectedSP tpt, int tk, double ub, double pen, double sim) {
//        this.s = s;
//        this.t = t;
//        this.tg = G;
//        this.spt = spt;
//        this.tpt = tpt;
//        this.TK = tk;
//        this.UB = ub;
//        this.PNT = pen;
//        this.SIM = sim;
//    }
//
//    private List<Edge> getNewPlateaus(int k) {
//        boolean[] visited = new boolean[tg.V() + 1];
//        Stack<Integer> treeNodes = tpt.getSpTree();
//        PriorityQueue<Edge> pls = new PriorityQueue<Edge>();
//        List<Edge> topKPls = new ArrayList<Edge>();
//
//        int depth;
//        int vid, pid, tail, head, plNodes = 1;
//        while (!treeNodes.isEmpty()) {
//            vid = treeNodes.pop();
//            if (spt.distTo(vid) != Integer.MAX_VALUE && spt.distTo(vid) + tpt.distTo(vid) <= spt.distTo(t) * UB) {     // Need to be improved search space not here
//                if (spt.distTo(vid) == tpt.distTo(vid) + spt.distTo(t) || tpt.distTo(vid) == spt.distTo(vid) + spt.distTo(t)) {
//                    visited[vid] = true;
//                    continue;
//                }
//                tail = head = vid;
//                if (!visited[vid]) {
//                    visited[vid] = true;
//                    pid = tpt.getEdge(vid).other(vid);
//                    while (pid != Integer.MAX_VALUE && !visited[pid]) {
//                        if (spt.getEdge(pid) != null && spt.getEdge(pid).either() == vid) {
//                            visited[pid] = true;
//                            plNodes++;
//                            if (plNodes == 2) {
//                                tail = vid;
//                                head = pid;
//                            } else {
//                                head = pid;
//                            }
//                            vid = pid;
//                            pid = tpt.getEdge(vid).other(vid);
//                        } else
//                            break;
//                    }
//                    if (plNodes >= 2) {
//                        depth = tpt.distTo(tail) - tpt.distTo(head);
//                        pls.add(new Edge(tail, head, depth));
//                        // Add a compressed via-node to min-heap
//                        viaNodes.add(new Node(spt.distTo(tail) + tpt.distTo(tail), tail));
//                    } else
//                        // Add single via-node to min-heap
//                        viaNodes.add(new Node(spt.distTo(vid) + tpt.distTo(vid), vid));
//                }
//                plNodes = 1;
//            } else
//                visited[vid] = true;
//        }
//        while (!pls.isEmpty()) {
//            Edge e = pls.poll();
//            if (topKPls.size() < k)
//                topKPls.add(e);
//            else
//                break;
//        }
//        return topKPls;
//    }
//
//    //void getPaths(int k) {
//    public Map<Double, List<Integer>> getPaths() {
//        //List<Edge> kPLs = getPlateaus(TK);
//        List<Edge> kPLs = getNewPlateaus(TK);
//        double offset = 60;
//        double ofs = 2.2;
//        Map<Double, List<Integer>> topKPath = new TreeMap<Double, List<Integer>>();
//        for (Edge e : kPLs) {
//            List<Integer> path;
//            path = usRecovery(e.tail());                    // Recovery sub-path u -> s
//            Collections.reverse(path);                      // Reverse sub-path s -> u
//            path.addAll(u2tRecovery(e.tail()));
//            //path.addAll(utRecovery(e.tail()));            // Append sub-path u.p -> t; undirected graph
//
//            //double pathLen = ((spt.distTo(e.tail()) + tpt.distTo(e.tail()) + ofs / kPLs.size()));     // Testing
//            double pathLen = ((spt.distTo(e.tail()) + tpt.distTo(e.tail()) + offset)) / 60000;          // Online
//
//            topKPath.put(pathLen, path);
//            ofs++;
//            offset++;
//        }
//        return topKPath;
//    }
//
//    public Map<Double, List<Integer>> simFunc() {
//        List<Integer> sp;
//        double offset = 60;
//        double ofs = 2.2;
//        Map<Double, List<Integer>> pathList = new TreeMap<Double, List<Integer>>();
//
//        int u = viaNodes.poll().getV();
//        sp = pathRecovery(u);
//        //double pathLen = ((spt.distTo(u) + tpt.distTo(u) + ofs / pathList.size()));
//        double pathLen = ((spt.distTo(u) + tpt.distTo(u) + offset)) / 60000;
//        pathList.put(pathLen, sp);
//
//        while (pathList.size() < TK && !viaNodes.isEmpty()) {
//            // Path from via-node heap
//            ++ofs;
//            ++offset;
//            boolean isValid = true;
//            u = viaNodes.poll().getV();
//            //pathLen = ((spt.distTo(u) + tpt.distTo(u) + ofs / pathList.size()));
//            pathLen = ((spt.distTo(u) + tpt.distTo(u) + offset)) / 60000;
//
//            List<Integer> path;
//            path = pathRecovery(u);
//            int midIndex = path.indexOf(u);
//            boolean[] o = new boolean[tg.V() + 1];
//            for (Integer i : path)
//                o[i] = true;
//
//            /*StdOut.print("via-node: " + u + "--> " + spt.distTo(u) + " " + tpt.distTo(u));
//            for (Integer i : path)
//                StdOut.print (i + "-");
//            StdOut.println();*/
//
//            for (Map.Entry<Double, List<Integer>> route : pathList.entrySet()) {
//                int cn, ct = 0, head = 0, tail = 0, overlap = 0;
//                List<Integer> rt = route.getValue();
//                int n = rt.size();
//                for (int j = 0; j < n; j++) {
//                    cn = rt.get(j);
//                    if (o[cn]) {
//                        if (ct == 0)
//                            tail = cn;
//                        else
//                            head = cn;
//                        ct++;
//                    } else {
//                        if (ct > 1) {
//                            if (path.indexOf(head) < midIndex)
//                                overlap += spt.distTo(head) - spt.distTo(tail);
//                            else
//                                overlap += tpt.distTo(tail) - tpt.distTo(head);
//                        }
//                        if (ct != 0)
//                            ct = head = tail = 0;
//                    }
//                }
//                if (ct > 1)
//                    overlap += tpt.distTo(tail) - tpt.distTo(head);
//
//                double minLen = route.getKey() >= pathLen ? pathLen : route.getKey();
//                //if ((overlap / minLen) > 0.5) {
//                if ((overlap / (60000 * minLen)) > this.SIM) {               // Similarity threshold
//                    isValid = false;
//                    break;
//                }
//            }
//            if (pathList.size() < TK) {
//                if (isValid)
//                    pathList.put(pathLen, path);
//            } else
//                break;
//        }
//        return pathList;
//    }
//
//    public Map<Double, List<Integer>> penFunc() {
//        int ct = 0;
//        double ofs = 2.2;
//        double offset = 60;
//        List<Integer> sp = new ArrayList<Integer>();
//        EdgeWeightedGraph CG = new EdgeWeightedGraph(true);
//        Map<Double, List<Integer>> pathList = new TreeMap<Double, List<Integer>>();
//        // Take out the original sp
//        int u = t, v = t;
//        while (u != s) {
//            sp.add(u);
//            v = u;
//            u = spt.getEdge(u).either();
//            for (Edge e : CG.adj(u)) {
//                if (e.other(u) == v) {
//                    e.setWeight((int) Math.ceil(e.weight() * PNT));
//                }
//            }
//        }
//        sp.add(s);
//        Collections.reverse(sp);
//        pathList.put((spt.distTo(t) + offset) * 1.0 / 60000, sp);
//        //pathList.put(spt.distTo(t) * 1.0, sp);
//
//        while (pathList.size() < TK) {
//            DijkstraUndirectedSP sptUpdate = new DijkstraUndirectedSP(CG, s, t);
//            if (sptUpdate.distTo(t) > spt.distTo(t) * UB)
//                break;
//            List<Integer> sps = new ArrayList<Integer>();
//            u = v = t;
//            while (u != s) {
//                sps.add(u);
//                v = u;
//                u = sptUpdate.getEdge(u).either();
//                for (Edge e : CG.adj(u)) {
//                    if (e.other(u) == v) {
//                        e.setWeight((int) Math.ceil(e.weight() * PNT));
//                    }
//                }
//            }
//            sps.add(s);
//            Collections.reverse(sps);
//
//            boolean isInsert = true;
//            for (List<Integer> pp : pathList.values()) {
//                if (pp.equals(sps)) {
//                    isInsert = false;
//                    break;
//                }
//            }
//            if (isInsert)   // Need to check UB
//                //pathList.put(sptUpdate.distTo(t) + ofs / (pathList.size() + 1), sps);
//                pathList.put((sptUpdate.distTo(t) + offset) * 1.0 / 60000, sps);
//
//            ofs++;
//            offset++;
//        }
//        return pathList;
//    }
//
//    private List<Integer> pathRecovery(int u) {
//        List<Integer> p;
//        p = usRecovery(u);
//        Collections.reverse(p);
//        p.addAll(u2tRecovery(u));
//        return p;
//    }
//
//    private List<Integer> usRecovery(int u) {
//        List<Integer> path = new ArrayList<Integer>();
//        int n = u;
//        while (n != Integer.MAX_VALUE) {
//            //StdOut.println(n);
//            path.add(n);
//            n = spt.getEdge(n).either();
//        }
//        return path;
//    }
//
//    private List<Integer> utRecovery(int u) {
//        List<Integer> path = new ArrayList<Integer>();
//        int n = tpt.getEdge(u).either();
//        while (n != Integer.MAX_VALUE) {
//            path.add(n);
//            n = tpt.getEdge(n).either();
//        }
//        return path;
//    }
//
//    private List<Integer> u2tRecovery(int u) {
//        List<Integer> path = new ArrayList<Integer>();
//        int n = tpt.getEdge(u).other(u);
//        while (n != Integer.MAX_VALUE) {
//            path.add(n);
//            n = tpt.getEdge(n).other(n);
//        }
//        return path;
//    }
//
//    public static void main(String[] args) {
//        // Construct the graph; need to include the coordinates
//        In in = new In("src/main/webapp/WEB-INF/classes/conf/tiny_tests.txt");
//        EdgeWeightedGraph tg = new EdgeWeightedGraph(in);
//
//        // Create forward and backward shortest path tree
//        double ub = 1.4;
//        int s = 1, t = 10;
//        DijkstraUndirectedSP spt = new DijkstraUndirectedSP(tg, s, t, ub);
//        DijkstraUndirectedSP tpt = new DijkstraUndirectedSP(tg, t, spt, true, ub);
//
//        // get top-k plateaus
//        Plateaus pls = new Plateaus(s, t, tg, spt, tpt, 5, ub, 1.2, 0.5);
//
//        StdOut.println(tg.V());
//
//        // Get k-plateaus
//        //       List<Edge> kPLs = pls.getNewPlateaus(5);
////        for (Edge e : kPLs) {
////            StdOut.println(e.toString());
////        }
//
//        // Get plateau-paths
//        /*Map<Double, List<Integer>> topKPath = pls.getPaths(1);
//        StdOut.println("Plateau path: " + topKPath.size());
//        for (List<Integer> pp : topKPath.values()) {
//            for (int p : pp) {
//                StdOut.print(p + " ");
//            }
//            StdOut.println();
//        }
//
//        // get sim-paths
//        Map<Double, List<Integer>> simKPath = pls.simFunc(5);
//        StdOut.println("Sim-path:" + simKPath.size());
//        for (List<Integer> pp : simKPath.values()) {
//            for (int p : pp) {
//                StdOut.print(p + " ");
//            }
//            StdOut.println();
//        }*/
//
//        /*DijkstraUndirectedSP penTree = new DijkstraUndirectedSP(tg, s, t);
//        StdOut.println("s to t dist: " + penTree.distTo(t));
//        List<Integer> sp = new ArrayList<Integer>();
//        int u = t, v = t;
//        while (u != s) {
//            sp.add(u);
//            v = u;
//            u = penTree.getEdge(u).either();
//            StdOut.print(u + "-" + v + "->");    // 2147483647 - 1
//            // Change edge weight in G: e = {u -> v}
//            for (Edge e : tg.adj(u)) {
//                if (e.other(u) == v) {
//                    StdOut.println(e.toString());
//                    e.setWeight((int)Math.ceil(e.weight() * 1.1));
//                }
//            }
//        }
//        sp.add(s);
//        for (Integer i : sp)
//            StdOut.print(i + " ");
//        StdOut.println();
//
//        DijkstraUndirectedSP penTree2 = new DijkstraUndirectedSP(tg, s, t);
//        StdOut.println("s to t dist: " + penTree2.distTo(t));
//        List<Integer> sp2 = new ArrayList<Integer>();
//        u = v = t;
//        while (u != s) {
//            sp2.add(u);
//            v = u;
//            u = penTree2.getEdge(u).either();
//            StdOut.print(u + "-" + v + "->");    // 2147483647 - 1
//            // Change edge weight in G: e = {u -> v}
//            for (Edge e : tg.adj(u)) {
//                if (e.other(u) == v) {
//                    StdOut.println(e.toString());
//                    e.setWeight((int)Math.ceil(e.weight() * 1.1));
//                }
//            }
//        }
//        sp2.add(s);
//        for (Integer i : sp2)
//            StdOut.print(i + " ");
//        StdOut.println();*/
//
//        // Penalty
////        Map<Double, List<Integer>> topKPath = pls.penFunc();
////        StdOut.println("Plateau path: " + topKPath.size());
////        for (List<Integer> pp : topKPath.values()) {
////            for (int p : pp) {
////                StdOut.print(p + " ");
////            }
////            StdOut.println();
////        }
//
//
//    }
//}
//
//
//// Undirected graph
//    /*public List<Edge> getPlateaus(int k) {
//        boolean[] visited = new boolean[tg.V() + 1];
//        Stack<Integer> treeNodes = tpt.getSpTree();
//        PriorityQueue<Edge> pls = new PriorityQueue<Edge>();
//        List<Edge> topKPls = new ArrayList<Edge>();
//
//        int depth;
//        int vid, pid, tail, head, plNodes = 1;
//        while (!treeNodes.isEmpty()) {
//            vid = treeNodes.pop();
//            if (spt.distTo(vid) + tpt.distTo(vid) <= spt.distTo(t) * 1.4) {     // Need to be improved search space not here
//                tail = head = vid;
//                if (!visited[vid]) {
//                    visited[vid] = true;
//                    pid = tpt.getEdge(vid).either();
//                    while (pid != Integer.MAX_VALUE && !visited[pid]) {
//                        if (spt.getEdge(pid) != null && spt.getEdge(pid).either() == vid) {
//                            visited[pid] = true;
//                            plNodes++;
//                            if (plNodes == 2) {
//                                tail = vid;
//                                head = pid;
//                            } else {
//                                head = pid;
//                            }
//                            vid = pid;
//                            pid = tpt.getEdge(vid).either();
//                        } else
//                            break;
//                    }
//                    if (plNodes >= 2) {
//                        depth = tpt.distTo(tail) - tpt.distTo(head);
//                        pls.add(new Edge(tail, head, depth));
//                    }
//                }
//                plNodes = 1;
//            } else
//                visited[vid] = true;
//        }
//        while (!pls.isEmpty()) {
//            Edge e = pls.poll();
//            if (topKPls.size() < k) {
//                topKPls.add(e);
//                //StdOut.println(e.toString());
//            } else {
//                break;
//            }
//        }
//        return topKPls;
//    }*/