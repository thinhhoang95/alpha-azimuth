// conflict_checker.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <utility>
#include <cmath>
#include <omp.h>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

namespace py = pybind11;

// Constants
constexpr double EARTH_RADIUS_KM = 6371.0;
constexpr double LATERAL_DISTANCE_THRESHOLD = 6.0 * 1.852; // 6 nautical miles in km
constexpr double VERTICAL_DISTANCE_THRESHOLD_LOWER = 1000.0; // feet
constexpr double VERTICAL_DISTANCE_THRESHOLD_UPPER = 2000.0; // feet
constexpr int INTERPOLATION_POINTS = 10;

// Structure to represent a Trajectory
struct Trajectory {
    std::vector<std::pair<double, double>> waypoints; // (lat, lon)
    std::vector<double> altitudes; // in feet
    std::vector<double> speeds; // in km/s
    int cat; // wake turbulence category
};

// Haversine formula to calculate distance between two lat/lon points in km
inline double haversine_distance(double lat1, double lon1, double lat2, double lon2) {
    double dlat = (lat2 - lat1) * M_PI / 180.0;
    double dlon = (lon2 - lon1) * M_PI / 180.0;
    double a = std::sin(dlat / 2.0) * std::sin(dlat / 2.0) +
               std::cos(lat1 * M_PI / 180.0) * std::cos(lat2 * M_PI / 180.0) *
               std::sin(dlon / 2.0) * std::sin(dlon / 2.0);
    double c = 2.0 * std::asin(std::sqrt(a));
    return EARTH_RADIUS_KM * c;
}

// Linear interpolation helper
inline void linear_interpolate(double start, double end, std::vector<double> &out) {
    double step = (end - start) / (INTERPOLATION_POINTS - 1);
    for(int i = 0; i < INTERPOLATION_POINTS; ++i) {
        out[i] = start + step * i;
    }
}

// Check if two segments are in conflict
bool segments_in_conflict(const Trajectory &traj1, int seg1_idx,
                          const Trajectory &traj2, int seg2_idx) {
    // Extract segment endpoints for traj1
    double p1_start_lat = traj1.waypoints[seg1_idx].first;
    double p1_start_lon = traj1.waypoints[seg1_idx].second;
    double p1_end_lat = traj1.waypoints[seg1_idx + 1].first;
    double p1_end_lon = traj1.waypoints[seg1_idx + 1].second;
    double alt1_start = traj1.altitudes[seg1_idx];
    double alt1_end = traj1.altitudes[seg1_idx + 1];
    
    // Extract segment endpoints for traj2
    double p2_start_lat = traj2.waypoints[seg2_idx].first;
    double p2_start_lon = traj2.waypoints[seg2_idx].second;
    double p2_end_lat = traj2.waypoints[seg2_idx + 1].first;
    double p2_end_lon = traj2.waypoints[seg2_idx + 1].second;
    double alt2_start = traj2.altitudes[seg2_idx];
    double alt2_end = traj2.altitudes[seg2_idx + 1];
    
    // Precompute interpolated points
    std::vector<double> t(INTERPOLATION_POINTS);
    double dt = 1.0 / (INTERPOLATION_POINTS - 1);
    for(int i = 0; i < INTERPOLATION_POINTS; ++i) {
        t[i] = dt * i;
    }
    
    // Interpolate positions and altitudes for traj1
    std::vector<std::pair<double, double>> traj1_interp(INTERPOLATION_POINTS);
    std::vector<double> alt1_interp(INTERPOLATION_POINTS);
    for(int i = 0; i < INTERPOLATION_POINTS; ++i) {
        double factor = t[i];
        traj1_interp[i].first = p1_start_lat + (p1_end_lat - p1_start_lat) * factor;
        traj1_interp[i].second = p1_start_lon + (p1_end_lon - p1_start_lon) * factor;
        alt1_interp[i] = alt1_start + (alt1_end - alt1_start) * factor;
    }
    
    // Interpolate positions and altitudes for traj2
    std::vector<std::pair<double, double>> traj2_interp(INTERPOLATION_POINTS);
    std::vector<double> alt2_interp(INTERPOLATION_POINTS);
    for(int i = 0; i < INTERPOLATION_POINTS; ++i) {
        double factor = t[i];
        traj2_interp[i].first = p2_start_lat + (p2_end_lat - p2_start_lat) * factor;
        traj2_interp[i].second = p2_start_lon + (p2_end_lon - p2_start_lon) * factor;
        alt2_interp[i] = alt2_start + (alt2_end - alt2_start) * factor;
    }
    
    // Check all pairs of interpolated points
    for(int i = 0; i < INTERPOLATION_POINTS; ++i) {
        for(int j = 0; j < INTERPOLATION_POINTS; ++j) {
            // Calculate lateral distance
            double lat1 = traj1_interp[i].first;
            double lon1 = traj1_interp[i].second;
            double lat2 = traj2_interp[j].first;
            double lon2 = traj2_interp[j].second;
            double lateral_distance = haversine_distance(lat1, lon1, lat2, lon2);
            
            // Calculate vertical separation
            double vertical_separation = std::abs(alt1_interp[i] - alt2_interp[j]);
            
            // Check conflict
            if (lateral_distance < LATERAL_DISTANCE_THRESHOLD &&
                vertical_separation < VERTICAL_DISTANCE_THRESHOLD_LOWER) {
                return true;
            }
        }
    }
    
    return false;
}

// Main conflict checking function
std::vector<std::tuple<int, int, int, int>> check_conflicts_cpp(const std::vector<Trajectory> &trajectories) {
    std::vector<std::tuple<int, int, int, int>> conflicts;
    int num_trajectories = trajectories.size();
    
    // Parallelize trajectory pair iterations
    #pragma omp parallel
    {
        std::vector<std::tuple<int, int, int, int>> local_conflicts;
        
        #pragma omp for nowait
        for(int i1 = 0; i1 < num_trajectories; ++i1) {
            for(int i2 = i1 + 1; i2 < num_trajectories; ++i2) {
                const Trajectory &traj1 = trajectories[i1];
                const Trajectory &traj2 = trajectories[i2];
                
                // Iterate over segments
                int num_segments1 = traj1.waypoints.size() - 1;
                int num_segments2 = traj2.waypoints.size() - 1;
                
                for(int seg1 = 0; seg1 < num_segments1; ++seg1) {
                    for(int seg2 = 0; seg2 < num_segments2; ++seg2) {
                        if (segments_in_conflict(traj1, seg1, traj2, seg2)) {
                            local_conflicts.emplace_back(i1, i2, seg1, seg2);
                        }
                    }
                }
            }
        }
        
        // Combine local conflicts into the main conflicts vector
        #pragma omp critical
        {
            conflicts.insert(conflicts.end(), local_conflicts.begin(), local_conflicts.end());
        }
    }
    
    return conflicts;
}

PYBIND11_MODULE(conflict_checker, m) {
    py::class_<Trajectory>(m, "Trajectory")
        .def(py::init<>())
        .def_readwrite("waypoints", &Trajectory::waypoints)
        .def_readwrite("altitudes", &Trajectory::altitudes)
        .def_readwrite("speeds", &Trajectory::speeds)
        .def_readwrite("cat", &Trajectory::cat);
    
    m.def("check_conflicts", &check_conflicts_cpp, "Check for conflicts between trajectories");
}
