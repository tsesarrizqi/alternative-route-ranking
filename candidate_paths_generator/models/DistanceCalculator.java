package models;

/**
 * Created by llx on 23/02/2018.
 */
public class DistanceCalculator {
    public DistanceCalculator() {

    }

    public static double distance(double lat1, double lon1, double lat2, double lon2) {
        double theta = lon1 - lon2;
        double dist = Math.sin(deg2rad(lat1)) * Math.sin(deg2rad(lat2)) + Math.cos(deg2rad(lat1)) * Math.cos(deg2rad(lat2)) * Math.cos(deg2rad(theta));
        dist = Math.acos(dist);
        dist = rad2deg(dist);
        dist = dist * 60 * 1.1515;
        dist = dist * 1.609344;

        //return (dist);             // return Kilometer
        return (dist * 10000);       // COL data
    }

    // This function converts decimal degrees to radians
    private static double deg2rad(double deg) {
        return (deg * Math.PI / 180.0);
    }

	// This function converts radians to decimal degrees
    private static double rad2deg(double rad) {
        return (rad * 180 / Math.PI);
    }

    public static void main (String[] args) throws Exception
    {
        //-37.803166, 144.953404, -37.808492, 144.953970
        //System.out.println(distance(32.9697, -96.80322, 29.46786, -98.53506) + " Kilometers\n");
        //39.971957, -103.978333, 39.971728, -103.992898 //col
        //-37.884064, 145.0906985, -37.8840259, 145.0908863 // mel
        System.out.println(distance(39.971957, -103.978333, 39.971728, -103.992898) + " Kilometers\n");
        System.out.println(distance(39.971957, -103.978333, 39.971728, -103.992898) * 10000 + " centimeters\n"); // * 10000 for COL data

        //-37.8847831,145.0923066,-37.8850153,145.0944589
        System.out.println(distance(-37.8847831,145.0923066,-37.8850153,145.0944589) + " Kilometers\n");
        System.out.println(distance(-37.8847831,145.0923066,-37.8850153,145.0944589) * 100000 + " centimeters\n"); // * 10000 for COL data
    }
}
