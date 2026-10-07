package models;

/**
 * Created by llx on 18/02/2018.
 */
public class Parameter {
//    private int topK;
//    private double ub;
//    private double pen;
//    private double sim;

//    private double dm;

    private String source;
    private String target;
    private String rx;
    private String ry;
    private String rz;

    private String nptr;

    private String mel;
    private String goRate;
    private String plRate;
    private String simRate;
    private String penRate;
    private String comment;

//    private String[] dmList;

//    private String[] topkList;
//    private String[] ubList;
//    private String[] penList;
//    private String[] simList;

    public Parameter() {
    }

    public Parameter(String np) {
        this.nptr = np;
    }

    public Parameter(String source, String target) {
        this.source = source;
        this.target = target;
    }

    public Parameter(String goRate, String plRate, String simRate, String penRate, String comment) {
        this.goRate = goRate;
        this.plRate = plRate;
        this.simRate = simRate;
        this.penRate = penRate;
        this.comment = comment;
    }

//    public Parameter(String[] dmList) {
//        this.dmList = dmList;
//    }

//    public Parameter(String[] topkList, String[] ubList, String[] penList, String[] simList) {
//        this.topkList = topkList;
//        this.ubList = ubList;
//        this.penList = penList;
//        this.simList = simList;
//    }
//
//    public Parameter(int topK, String source, String target, String method, String[] topkList) {
//        this.topK = topK;
//        this.source = source;
//        this.target = target;
//        this.method = method;
//        this.topkList = topkList;
//    }
//
//    public void setSim(double sim) {
//        this.sim = sim;
//    }
//
//    public double getSim() {
//        return sim;
//    }
//
//    public void setPen(double pen) {
//        this.pen = pen;
//    }
//
//    public double getPen() {
//        return pen;
//    }
//
//    public void setUb(double ub) {
//        this.ub = ub;
//    }
//
//    public double getUb() {
//        return ub;
//    }
//
//    public int getTopK() {
//        return topK;
//    }
//
//    public void setTopK(int topK) {
//        this.topK = topK;
//    }

//    public void setDm(double dm) { this.dm = dm; }
//
//    public Double getDm() { return dm; }

    public String getSource() {
        return source;
    }

    public void setSource(String source) {
        this.source = source;
    }

    public String getTarget() {
        return target;
    }

    public void setTarget(String target) {
        this.target = target;
    }

    public String getRx() { return rx; };

    public void setRx(String x) { this.rx = x; };

    public String getRy() { return ry; };

    public void setRy(String y) {this.ry = y; };

    public String getRz() { return rz; };

    public void setRz(String z) {this.rz = z; };

    public void setMel(String m) { this.mel = m; }

    public String getMel() { return mel; }

    public void setGoRate(String r) { this.goRate = r; };

    public String getGoRate() { return goRate; }

    public void setPlRate(String r) { this.plRate = r; }

    public String getPlRate() { return plRate; }

    public void setSimRate(String r) { this.simRate = r; }

    public String getSimRate() { return simRate; }

    public void setPenRate(String r) { this.penRate = r; }

    public String getPenRate() { return penRate; }

    public void setComment(String s) { this.comment = s; }

    public String getComment() { return comment; }

    public void setNptr(String np) { this.nptr = np; }

    public String getNptr() { return nptr; }

//    public String[] getDmList() { return dmList; }


//    public String[] getTopkList() {
//        return topkList;
//    }
//
//    public String[] getUbList() {
//        return ubList;
//    }
//
//    public String[] getPenList() {
//        return penList;
//    }
//
//    public String[] getSimList() {
//        return simList;
//    }
}
