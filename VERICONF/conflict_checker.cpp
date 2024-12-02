// conflict_checker.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <string>
#include <sstream>
#include <tuple>

namespace py = pybind11;

// Declare M_PI to avoid Visual Studio C++ warning
#define M_PI 3.14159265358979323846 

// Constants
const double EARTH_RADIUS = 6371.0; // km
const double LATERAL_DISTANCE_THRESHOLD = 6.0 * 1.852; // 6 nautical miles in km
const double VERTICAL_DISTANCE_THRESHOLD_LOWER = 1000.0; // feet
const double VERTICAL_DISTANCE_THRESHOLD_UPPER = 2000.0; // feet

// Helper function to compute haversine distance
double haversine_distance(double lat1, double lon1, double lat2, double lon2) {
    double dlat = (lat2 - lat1) * M_PI / 180.0;
    double dlon = (lon2 - lon1) * M_PI / 180.0;
    lat1 = lat1 * M_PI / 180.0;
    lat2 = lat2 * M_PI / 180.0;

    double a = std::sin(dlat / 2) * std::sin(dlat / 2) +
               std::cos(lat1) * std::cos(lat2) *
               std::sin(dlon / 2) * std::sin(dlon / 2);
    double c = 2 * std::asin(std::sqrt(a));
    return EARTH_RADIUS * c; // in km
}

class Trajectory {
public:
    // Member variables
    std::vector<std::vector<double>> waypoints; // N x 2
    std::vector<double> altitudes; // feet
    std::vector<double> speeds; // km/s
    std::string cat;
    double t0;

    std::vector<double> passing_time; // seconds
    std::vector<std::vector<double>> waypoints_xyz; // N x 3

    // Constructor
    Trajectory(py::array_t<double> waypoints_np,
               py::array_t<double> altitudes_np,
               py::array_t<double> speeds_np,
               std::string cat_,
               double t0_ = 0.0)
        : cat(cat_), t0(t0_) {
        
        // Convert waypoints
        auto wp = waypoints_np.unchecked<2>();
        size_t N = wp.shape(0);
        if (wp.shape(1) != 2) {
            throw std::runtime_error("Waypoints must be 2D coordinates (lat, lon)");
        }
        waypoints.reserve(N);
        for (size_t i = 0; i < N; ++i) {
            waypoints.emplace_back(std::vector<double>{wp(i,0), wp(i,1)});
        }

        // Convert altitudes
        auto alt = altitudes_np.unchecked<1>();
        if (altitudes_np.shape(0) != N) {
            throw std::runtime_error("Number of waypoints and altitudes must match");
        }
        altitudes.reserve(N);
        for (size_t i = 0; i < N; ++i) {
            altitudes.emplace_back(alt(i));
        }

        // Convert speeds
        auto spd = speeds_np.unchecked<1>();
        if (speeds_np.shape(0) != N) {
            throw std::runtime_error("Number of waypoints and speeds must match");
        }
        speeds.reserve(N);
        for (size_t i = 0; i < N; ++i) {
            speeds.emplace_back(spd(i));
        }

        // Initialize passing_time and waypoints_xyz
        passing_time.resize(N, 0.0);
        waypoints_xyz.resize(N, std::vector<double>(3, 0.0));

        compute_times();
        compute_cartesian_coordinates();
    }

    // Compute passing times
    void compute_times(double t0_new = -1.0) {
        if (t0_new >= 0.0) {
            t0 = t0_new;
        }
        size_t N = waypoints.size();
        passing_time[0] = t0;

        for (size_t i = 0; i < N -1; ++i) {
            double lat1 = waypoints[i][0];
            double lon1 = waypoints[i][1];
            double lat2 = waypoints[i+1][0];
            double lon2 = waypoints[i+1][1];

            double distance = haversine_distance(lat1, lon1, lat2, lon2); // km
            double avg_speed = (speeds[i] + speeds[i+1]) / 2.0; // km/s
            double segment_time = distance / avg_speed; // seconds

            passing_time[i+1] = passing_time[i] + segment_time;
        }
    }

    // Compute Cartesian coordinates
    void compute_cartesian_coordinates() {
        size_t N = waypoints.size();
        for (size_t i = 0; i < N; ++i) {
            double lat_rad = waypoints[i][0] * M_PI / 180.0;
            double lon_rad = waypoints[i][1] * M_PI / 180.0;

            double x = std::cos(lat_rad) * std::cos(lon_rad) * EARTH_RADIUS;
            double y = std::cos(lat_rad) * std::sin(lon_rad) * EARTH_RADIUS;
            double z = std::sin(lat_rad) * EARTH_RADIUS;

            waypoints_xyz[i][0] = x;
            waypoints_xyz[i][1] = y;
            waypoints_xyz[i][2] = z;
        }
    }

    // String representation
    std::string to_string() const {
        std::ostringstream oss;
        size_t N = waypoints.size();
        for (size_t i = 0; i < N; ++i) {
            oss << "Waypoint " << i << ": lat=" << waypoints[i][0]
                << ", lon=" << waypoints[i][1]
                << ", speed=" << speeds[i] << " km/s"
                << ", altitude=" << altitudes[i] << " ft"
                << ", time=" << passing_time[i] << " s\n";
        }
        return oss.str();
    }
};

// Function to check conflicts between segments
std::tuple<bool, double> segments_in_conflict_cartesian(
    const std::vector<double>& p1_start_xyz,
    const std::vector<double>& p1_end_xyz,
    double alt1_start,
    double alt1_end,
    const std::vector<double>& p2_start_xyz,
    const std::vector<double>& p2_end_xyz,
    double alt2_start,
    double alt2_end,
    double t1_start,
    double t1_end,
    double t2_start,
    double t2_end,
    double vertical_distance_threshold_lower = VERTICAL_DISTANCE_THRESHOLD_LOWER,
    double vertical_distance_threshold_upper = VERTICAL_DISTANCE_THRESHOLD_UPPER,
    double lateral_distance_threshold = LATERAL_DISTANCE_THRESHOLD
) {
    // Check vertical distances
    if (std::abs(alt1_start - alt2_start) > vertical_distance_threshold_lower ||
        std::abs(alt1_end - alt2_end) > vertical_distance_threshold_lower) {
        return std::make_tuple(false, std::nan(""));
    }

    // Rename variables for clarity
    std::vector<double> x1 = p1_start_xyz;
    std::vector<double> x3 = p2_start_xyz;
    std::vector<double> x2 = p1_end_xyz;
    std::vector<double> x4 = p2_end_xyz;

    double t1 = t1_start;
    double t2_ = t1_end;
    double t3 = t2_start;
    double t4 = t2_end;

    // Compute a and b vectors
    std::vector<double> a(3, 0.0);
    std::vector<double> b_vec(3, 0.0);
    for(int i=0;i<3;i++) {
        a[i] = x1[i] - x3[i] - t1 * (x2[i] - x1[i]) / (t2_ - t1) + t3 * (x4[i] - x3[i]) / (t4 - t3);
        b_vec[i] = (x2[i] - x1[i]) / (t2_ - t1) - (x4[i] - x3[i]) / (t4 - t3);
    }

    // Compute dot products
    double a_dot_b = 0.0;
    double b_dot_b = 0.0;
    double a_dot_a = 0.0;
    for(int i=0;i<3;i++) {
        a_dot_b += a[i] * b_vec[i];
        b_dot_b += b_vec[i] * b_vec[i];
        a_dot_a += a[i] * a[i];
    }

    // Compute delta
    double delta = (a_dot_b * a_dot_b) - b_dot_b * (a_dot_a - lateral_distance_threshold * lateral_distance_threshold);

    if (delta < 0) {
        return std::make_tuple(false, std::nan(""));
    }

    if (b_dot_b <= 1e-6) {
        if ((a_dot_a - lateral_distance_threshold * lateral_distance_threshold) > 0) {
            return std::make_tuple(false, std::nan(""));
        } else {
            return std::make_tuple(true, std::nan(""));
        }
    }

    double sqrt_delta = std::sqrt(delta);
    double t_lb = (-a_dot_b - sqrt_delta) / b_dot_b;
    double t_ub = (-a_dot_b + sqrt_delta) / b_dot_b;

    // Intersection time bounds
    double t_min = std::min({t1, t2_, t3, t4});
    double t_max = std::max({t1, t2_, t3, t4});

    // Check if t_lb and t_ub are within the segment bounds
    if (t_lb > t_min && t_lb < t_max && t_lb > t1 && t_lb < t2_ &&
        t_lb > t3 && t_lb < t4 &&
        t_ub > t_min && t_ub < t_max && t_ub > t1 && t_ub < t2_ &&
        t_ub > t3 && t_ub < t4) {
        return std::make_tuple(true, t_lb);
    }

    return std::make_tuple(false, std::nan(""));
}

// Function to check conflicts between trajectories
py::list check_conflicts(
    const std::vector<Trajectory>& trajectories,
    double vertical_distance_threshold_lower = VERTICAL_DISTANCE_THRESHOLD_LOWER,
    double vertical_distance_threshold_upper = VERTICAL_DISTANCE_THRESHOLD_UPPER,
    double lateral_distance_threshold = LATERAL_DISTANCE_THRESHOLD
) {
    py::list conflicts;

    size_t num_trajectories = trajectories.size();
    for (size_t i1 = 0; i1 < num_trajectories; ++i1) {
        for (size_t i2 = i1 + 1; i2 < num_trajectories; ++i2) {
            const Trajectory& traj1 = trajectories[i1];
            const Trajectory& traj2 = trajectories[i2];

            size_t seg1_max = traj1.waypoints.size() - 1;
            size_t seg2_max = traj2.waypoints.size() - 1;

            for (size_t seg1 = 0; seg1 < seg1_max; ++seg1) {
                for (size_t seg2 = 0; seg2 < seg2_max; ++seg2) {
                    auto [in_conflict, t_lb] = segments_in_conflict_cartesian(
                        traj1.waypoints_xyz[seg1],
                        traj1.waypoints_xyz[seg1 + 1],
                        traj1.altitudes[seg1],
                        traj1.altitudes[seg1 + 1],
                        traj2.waypoints_xyz[seg2],
                        traj2.waypoints_xyz[seg2 + 1],
                        traj2.altitudes[seg2],
                        traj2.altitudes[seg2 + 1],
                        traj1.passing_time[seg1],
                        traj1.passing_time[seg1 + 1],
                        traj2.passing_time[seg2],
                        traj2.passing_time[seg2 + 1],
                        vertical_distance_threshold_lower,
                        vertical_distance_threshold_upper,
                        lateral_distance_threshold
                    );

                    if (in_conflict) {
                        conflicts.append(py::make_tuple(i1, i2, seg1, seg2, t_lb));
                    }
                }
            }
        }
    }

    return conflicts;
}

PYBIND11_MODULE(conflict_checker, m) {
    m.doc() = "A module to check conflicts between aircraft trajectories";

    py::class_<Trajectory>(m, "Trajectory")
        .def(py::init<py::array_t<double>, py::array_t<double>, py::array_t<double>, std::string, double>(),
             py::arg("waypoints"), py::arg("altitudes"), py::arg("speeds"), py::arg("cat"), py::arg("t0")=0.0)
        .def("compute_times", &Trajectory::compute_times, py::arg("t0")=-1.0)
        .def("compute_cartesian_coordinates", &Trajectory::compute_cartesian_coordinates)
        .def("__str__", &Trajectory::to_string)
        .def_readonly("waypoints", &Trajectory::waypoints)
        .def_readonly("altitudes", &Trajectory::altitudes)
        .def_readonly("speeds", &Trajectory::speeds)
        .def_readonly("cat", &Trajectory::cat)
        .def_readonly("t0", &Trajectory::t0)
        .def_readonly("passing_time", &Trajectory::passing_time)
        .def_readonly("waypoints_xyz", &Trajectory::waypoints_xyz);

    m.def("check_conflicts", &check_conflicts,
          py::arg("trajectories"),
          py::arg("vertical_distance_threshold_lower") = VERTICAL_DISTANCE_THRESHOLD_LOWER,
          py::arg("vertical_distance_threshold_upper") = VERTICAL_DISTANCE_THRESHOLD_UPPER,
          py::arg("lateral_distance_threshold") = LATERAL_DISTANCE_THRESHOLD,
          "Check for conflicts between a list of Trajectory objects");
}