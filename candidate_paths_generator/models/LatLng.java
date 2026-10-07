package models;

import java.io.IOException;

public class LatLng {
    private double lat;
    private double lng;

    public LatLng() {
        lat = 0.0d;
        lng = 0.0d;
    }

    public LatLng(double lat, double lng) {
        this.lat = lat;
        this.lng = lng;
    }

    public double getLat() {
        return lat;
    }

    public double getLng() {
        return lng;
    }

    public String toString(){
        return lat + "-" + lng;
    }

    public static void main(String[] args) {
        LatLng t = new LatLng(-37.832645, 145.227459);
        StdOut.println(t.toString());
    }
}
