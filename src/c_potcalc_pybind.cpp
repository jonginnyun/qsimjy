#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <thread>
#include <vector>
#include <cmath>

namespace py = pybind11;

double GreenFunctionDer(double* r, double* rp) {
    double r_plus = std::sqrt(std::pow(r[0] - rp[0], 2) + std::pow(r[1] - rp[1], 2) + std::pow(r[2] + rp[2], 2));
    double r_minus = std::sqrt(std::pow(r[0] - rp[0], 2) + std::pow(r[1] - rp[1], 2) + std::pow(r[2] - rp[2], 2));
    return -(rp[2] - r[2]) / std::pow(r_minus, 3) + (rp[2] + r[2]) / std::pow(r_plus, 3);
}

// Scientific Constants
const double mu_0 = 1.25637e-6; // Vacuum permeability  [N * A^-2]
const double pi = 3.14159265358979323846;
const double prop_const = mu_0 / (pi * 4.0); // Proportionality constant for magnetic field

void PotentialCalculation(double* result, int start_idx1, int end_idx1, int ny, int np0, int np1,
    double* potential_array, double* gate_level_array,
    double* x_list_array, double* y_list_array,
    double* xp_list_array, double* yp_list_array,
    double* args) {
    double initial_oxide_thickness = args[0];
    double oxide_thickness = args[1]; 
    double qd_well_depth = args[2];

    for (int idx1 = start_idx1; idx1 < end_idx1; ++idx1) {
        for (int idx2 = 0; idx2 < ny; ++idx2) {
            double r[3] = { x_list_array[idx1], y_list_array[idx2], qd_well_depth };
            double sum = 0.0;
            for (int idx3 = 0; idx3 < np0; ++idx3) {
                for (int idx4 = 0; idx4 < np1; ++idx4) {
                    double rp_z = initial_oxide_thickness * (1 - gate_level_array[idx3 * np1 + idx4]) * oxide_thickness;
                    double rp[3] = { xp_list_array[idx3], yp_list_array[idx4], rp_z };
                    sum -= GreenFunctionDer(r, rp) * potential_array[idx3 * np1 + idx4];
                }
            }
            result[idx1 * ny + idx2] = sum;
        }
    }
}

void ParallelLookupCompute(py::array_t<double> answer_array_input,
    py::array_t<double> pot_lookup_array_input,
    py::array_t<double> gate_level_array_input,
    py::array_t<double> x_list_input,
    py::array_t<double> y_list_input,
    py::array_t<double> xp_list_input,
    py::array_t<double> yp_list_input,
    py::array_t<double> args_list,
    int num_threads = 4) {
    // GIL release
    py::gil_scoped_release release;


    //buffer generation : args_list is a list [initial_oxide_thickness, oxide_thickness, qd_well_depth]
    py::buffer_info answer_buffer = answer_array_input.request();
    py::buffer_info buffer_potential_lookup = pot_lookup_array_input.request();
    py::buffer_info buffer_gate_level = gate_level_array_input.request();
    py::buffer_info buffer_x = x_list_input.request();
    py::buffer_info buffer_y = y_list_input.request();
    py::buffer_info buffer_xp = xp_list_input.request();
    py::buffer_info buffer_yp = yp_list_input.request();
    py::buffer_info arg_buffer = args_list.request();

    //array conversion (all double assumed)
    double* answer_array = static_cast<double*>(answer_buffer.ptr);
    double* gate_level_array = static_cast<double*>(buffer_gate_level.ptr);
    double* potential_array = static_cast<double*>(buffer_potential_lookup.ptr);
    double* x_list_array = static_cast<double*>(buffer_x.ptr);
    double* y_list_array = static_cast<double*>(buffer_y.ptr);
    double* xp_list_array = static_cast<double*>(buffer_xp.ptr);
    double* yp_list_array = static_cast<double*>(buffer_yp.ptr);
    double* args = static_cast<double*>(arg_buffer.ptr);

    //array dimenstion information (<int> is used instead of <size_t> for the cross OS compatibility)
    int n_x = static_cast<int>(buffer_x.shape[0]);
    int n_y = static_cast<int>(buffer_y.shape[0]);
    int np0 = static_cast<int>(buffer_xp.shape[0]);
    int np1 = static_cast<int>(buffer_yp.shape[0]);

    //Set the number of threads to use in the multithreading
    int num_threads_max = std::thread::hardware_concurrency();
    num_threads = (num_threads_max > num_threads) ? num_threads : num_threads_max;

    // idx1 per thread
    int idx1_per_thread = n_x / num_threads;
    int remainder = n_x % num_threads;
    // ThreadPool Generation
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    int current_idx1 = 0;

    for (int t = 0; t < num_threads; ++t) {
        int start_idx1 = current_idx1;
        int end_idx1 = start_idx1 + idx1_per_thread;
        if (t < remainder) {
            end_idx1 += 1;
        }
        current_idx1 = end_idx1;

        threads.push_back(std::thread(PotentialCalculation, answer_array, start_idx1, end_idx1, n_y, np0, np1,
            potential_array, gate_level_array, x_list_array, y_list_array,
            xp_list_array, yp_list_array, args));
    }

    for (auto& th : threads) {
        if (th.joinable()) {
            th.join();
        }
    }
}

PYBIND11_MODULE(_cxx_potcalc, m) {
    m.def("potcalc", &ParallelLookupCompute, "C++ BOUND CODE");
}
