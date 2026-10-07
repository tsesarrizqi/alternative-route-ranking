package models;

import java.util.*;

public class Router {
    private int s;
    private int t;
    private int TK;
    private double UB;
    private double PNT;
    private double SIM;
    private double DM;
    private EdgeWeightedGraph TG;
    private DijkstraUndirectedSP spt;
    private DijkstraUndirectedSP tpt;

    public Router(int s, int t, EdgeWeightedGraph G, DijkstraUndirectedSP spt, DijkstraUndirectedSP tpt, 
                  int tk, double ub, double pen, double sim, double dm) {
        this.s = s;
        this.t = t;
        this.TG = G;
        this.spt = spt;
        this.tpt = tpt;
        this.TK = tk;
        this.UB = ub;
        this.PNT = pen;
        this.SIM = sim;
        this.DM = dm;
    }

    /**
     * Return the first TK valid paths in discovery order.
     * Each element is a Pair where:
     *   key   = total path length
     *   value = list of node IDs forming the path
     */
    public List<Pair<Double, List<Integer>>> getOrderedPaths() {
        boolean[] visited = new boolean[TG.V() + 1];
        Stack<Integer> treeNodes = tpt.getSpTree();
        PriorityQueue<Edge> pls = new PriorityQueue<>();

        /* Traverse TPT and build plateaus */
        int plNodes = 1, vid, pid, tail, head;
        double depth;
        
        while (!treeNodes.isEmpty()) {
            vid = treeNodes.pop();
            if (spt.distTo(vid) != Double.POSITIVE_INFINITY &&
                spt.distTo(vid) + tpt.distTo(vid) <= spt.distTo(t) * UB) {

                /* skip main-path nodes */
                if (spt.distTo(vid) == tpt.distTo(vid) + spt.distTo(t) ||
                    tpt.distTo(vid) == spt.distTo(vid) + spt.distTo(t)) {
                    visited[vid] = true;
                    continue;
                }

                tail = head = vid;
                if (!visited[vid]) {
                    visited[vid] = true;
                    pid = tpt.getEdge(vid).other(vid);
                    while (pid != Integer.MAX_VALUE && !visited[pid]) {
                        if (spt.getEdge(pid) != null &&
                            spt.getEdge(pid).either() == vid) {
                            visited[pid] = true;
                            plNodes++;
                            if (plNodes == 2) { 
                                tail = vid; 
                                head = pid; 
                            } else { 
                                head = pid; 
                            }
                            vid = pid;
                            pid = tpt.getEdge(vid).other(vid);
                        } else break;
                    }
                    if (plNodes >= 2) {
                        depth = tpt.distTo(tail) - tpt.distTo(head);
                        pls.add(new Edge(tail, head, depth));
                    }
                    plNodes = 1;
                }
            } else visited[vid] = true;
        }

        /* Build paths in plateau order */
        List<Pair<Double, List<Integer>>> pathList = new ArrayList<>();
        double offset = 60;

        while (pathList.size() < TK && !pls.isEmpty()) {
            int u = pls.poll().tail();  // via-node
            double pathLen = (spt.distTo(u) + tpt.distTo(u) + offset) / 60000.0;
            List<Integer> path = pathRecovery(u);  // s → u → t

            /* Diversity checks against every accepted path */
            boolean isValid = true;
            for (Pair<Double, List<Integer>> route : pathList) {
                List<Integer> ref = route.value;
                int dev = s, merge = t;
                int i = 0;
                int limit = Math.min(path.size(), ref.size());

                /* deviation point */
                while (i < limit && path.get(i).equals(ref.get(i))) { 
                    i++; 
                }
                if (i >= 1) dev = path.get(i - 1);

                /* merge point */
                int xi = path.size() - 1, yj = ref.size() - 1;
                while (xi >= 0 && yj >= 0 && path.get(xi).equals(ref.get(yj))) { 
                    xi--; 
                    yj--; 
                }
                if (xi + 1 < path.size()) merge = path.get(xi + 1);

                if (merge == dev) { 
                    isValid = false; 
                    break; 
                }

                /* DM test */
                double xLen = spt.distTo(u) - spt.distTo(dev) + 
                             tpt.distTo(u) - tpt.distTo(merge);
                double yLen = Math.min(spt.distTo(merge) - spt.distTo(dev),
                                      tpt.distTo(dev) - tpt.distTo(merge));
                if (yLen <= 0 || (xLen / yLen) >= DM) {
                    isValid = false; 
                    break;
                }

                /* SIM test */
                boolean[] mark = new boolean[TG.V() + 1];
                for (int n : path) mark[n] = true;

                double overlap = 0;
                int ct = 0, tailSeg = 0, headSeg = 0;
                for (int n2 : ref) {
                    if (mark[n2]) {
                        if (ct == 0) tailSeg = n2; 
                        else headSeg = n2;
                        ct++;
                    } else if (ct > 1) {
                        overlap += (path.indexOf(headSeg) < path.indexOf(u))
                             ? (spt.distTo(headSeg) - spt.distTo(tailSeg))
                             : (tpt.distTo(tailSeg) - tpt.distTo(headSeg));
                        ct = 0;
                    }
                }
                if (ct > 1)
                    overlap += tpt.distTo(tailSeg) - tpt.distTo(headSeg);

                double minLen = Math.min(route.key, pathLen);
                if ((overlap / (60000 * minLen)) > SIM) {
                    isValid = false; 
                    break;
                }
            }

            if (isValid) {
                pathList.add(new Pair<>(pathLen, path)); 
                offset++;
            }
        }
        return pathList;
    }

    public Map<Double, List<Integer>> getPaths() {
        boolean[] visited = new boolean[TG.V() + 1];
        Stack<Integer> treeNodes = tpt.getSpTree();
        PriorityQueue<Edge> pls = new PriorityQueue<Edge>();
        PriorityQueue<Node> viaNodes = new PriorityQueue<Node>();

        double depth;
        int vid, pid, tail, head, plNodes = 1;
        
        while (!treeNodes.isEmpty()) {
            vid = treeNodes.pop();
            if (spt.distTo(vid) != Double.POSITIVE_INFINITY && 
                spt.distTo(vid) + tpt.distTo(vid) <= spt.distTo(t) * UB) {
                
                if (spt.distTo(vid) == tpt.distTo(vid) + spt.distTo(t) || 
                    tpt.distTo(vid) == spt.distTo(vid) + spt.distTo(t)) {
                    visited[vid] = true;
                    continue;
                }
                
                tail = head = vid;
                if (!visited[vid]) {
                    visited[vid] = true;
                    pid = tpt.getEdge(vid).other(vid);
                    while (pid != Integer.MAX_VALUE && !visited[pid]) {
                        if (spt.getEdge(pid) != null && spt.getEdge(pid).either() == vid) {
                            visited[pid] = true;
                            plNodes++;
                            if (plNodes == 2) {
                                tail = vid;
                                head = pid;
                            } else {
                                head = pid;
                            }
                            vid = pid;
                            pid = tpt.getEdge(vid).other(vid);
                        } else break;
                    }
                    if (plNodes >= 2) {
                        depth = tpt.distTo(tail) - tpt.distTo(head);
                        pls.add(new Edge(tail, head, depth));
                        viaNodes.add(new Node(spt.distTo(tail) + tpt.distTo(tail), tail));
                    } else {
                        viaNodes.add(new Node(spt.distTo(vid) + tpt.distTo(vid), vid));
                    }
                }
                plNodes = 1;
            } else visited[vid] = true;
        }

        // Top-k path recovery based on plateaus
        int u;
        double pathLen = 0.0;
        List<Integer> sp;
        double offset = 60;
        Map<Double, List<Integer>> pathList = new TreeMap<Double, List<Integer>>();
        
        if (!pls.isEmpty()) {
            u = pls.poll().tail();
            sp = pathRecovery(u);
            pathLen = ((spt.distTo(u) + tpt.distTo(u) + offset)) / 60000;
            pathList.put(pathLen, sp);
        }
        
        while (pathList.size() < TK && !pls.isEmpty()) {
            boolean isValid = true;
            u = pls.poll().tail();
            pathLen = ((spt.distTo(u) + tpt.distTo(u) + offset + 1)) / 60000;

            List<Integer> path = pathRecovery(u);

            // Validation check for current path
            for (Map.Entry<Double, List<Integer>> route : pathList.entrySet()) {
                int d = s, m = t;
                int xIdx = path.size() - 1;
                int yIdx = route.getValue().size() - 1;
                int size = Math.min(xIdx, yIdx);

                // get the deviation vertex
                for (int i = 0; i <= size; i++) {
                    if (!path.get(i).equals(route.getValue().get(i))) {
                        if (i >= 1) {
                            d = path.get(i - 1);
                            break;
                        }
                    }
                }

                // get the merging vertex
                while(xIdx >= 0 && yIdx >= 0) {
                    if (!path.get(xIdx).equals(route.getValue().get(yIdx))) {
                        m = path.get(xIdx + 1);
                        break;
                    }
                    xIdx--;
                    yIdx--;
                }

                if (m != d) {
                    double xLen = spt.distTo(u) - spt.distTo(d) + tpt.distTo(u) - tpt.distTo(m);
                    double yLen = Math.min(spt.distTo(m) - spt.distTo(d), 
                                          tpt.distTo(d) - tpt.distTo(m));
                    if (yLen <= 0) {
                        isValid = false;
                        break;
                    } else {
                        if ((xLen / yLen) >= this.DM) {
                            isValid = false;
                            break;
                        }
                    }
                } else {
                    isValid = false;
                    break;
                }
            }
            
            if (pathList.size() < TK) {
                if (isValid) {
                    pathList.put(pathLen, path);
                    offset++;
                }
            } else break;
        }
        return pathList;
    }

    public Map<Double, List<Integer>> penFunc() {
        double offset = 60;
        double pathLen;
        List<Integer> sp = new ArrayList<Integer>();
        EdgeWeightedGraph CG = new EdgeWeightedGraph(true);
        Map<Double, List<Integer>> pathList = new TreeMap<Double, List<Integer>>();

        // Get original shortest path
        int u = t, v = t;
        while (u != s) {
            sp.add(u);
            v = u;
            u = spt.getEdge(u).either();
            for (Edge e : CG.adj(u)) {
                if (e.other(u) == v) {
                    e.setWeight(e.weight() * PNT);
                }
            }
        }
        sp.add(s);
        Collections.reverse(sp);
        pathList.put((spt.distTo(t) + offset) / 60000, sp);

        // Find alternative paths by penalizing used edges
        while (pathList.size() < TK) {
            pathLen = 0.0;
            DijkstraUndirectedSP sptUpdate = new DijkstraUndirectedSP(CG, s, t);
            List<Integer> sps = new ArrayList<Integer>();
            u = v = t;
            
            while (u != s) {
                sps.add(u);
                v = u;
                u = sptUpdate.getEdge(u).either();
                for (Edge e : CG.adj(u)) {
                    if (e.other(u) == v) {
                        e.setWeight(e.weight() * PNT);
                    }
                }
                for (Edge e : TG.adj(u)) {
                    if (e.other(u) == v) {
                        pathLen += e.weight();
                    }
                }
            }
            
            if (pathLen > spt.distTo(t) * UB) break;

            sps.add(s);
            Collections.reverse(sps);

            boolean isInsert = true;
            for (List<Integer> pp : pathList.values()) {
                if (pp.equals(sps)) {
                    isInsert = false;
                    break;
                }
            }
            if (isInsert) {
                pathList.put((pathLen + offset) / 60000, sps);
            }
            offset++;
        }

        return pathList;
    }

    private List<Integer> pathRecovery(int u) {
        List<Integer> p = usRecovery(u);
        Collections.reverse(p);
        p.addAll(u2tRecovery(u));
        return p;
    }

    private List<Integer> usRecovery(int u) {
        List<Integer> path = new ArrayList<Integer>();
        int n = u;
        while (n != Integer.MAX_VALUE) {
            path.add(n);
            n = spt.getEdge(n).either();
        }
        return path;
    }

    private List<Integer> u2tRecovery(int u) {
        List<Integer> path = new ArrayList<Integer>();
        int n = tpt.getEdge(u).other(u);
        while (n != Integer.MAX_VALUE) {
            path.add(n);
            n = tpt.getEdge(n).other(n);
        }
        return path;
    }
}