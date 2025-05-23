#include <thread>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
//Namespace declaration
namespace py = pybind11;
//A function to be used by an individual threads
//IStrayFieldCalculatorThreads is a multithreaded function that calculates
//the I-th component of the stray field generated by the magnetiztion density
//input
const double mu_0 = 1.25637e-6;
const double pi = 3.14159265358979323846;
const double prop_const = mu_0/pi/4.0;
void StrayFieldCalculatorThreads(double * answer_array,
    const std::vector<double*>& r,
    const std::vector<double*>& mag,
    const std::vector<double*>& ri,
    int start_idx,
    int end_idx,
    int ny,
    int * ni_array,
    int component_idx){
    int nxi = ni_array[0], nyi = ni_array[1], nzi = ni_array[2];
    double *x_array = r[0], *y_array=r[1];
    double z = *r[2];
    double *mx_array = mag[0], *my_array = mag[1], *mz_array = mag[2];
    double *xi_array = ri[0],  *yi_array = ri[1], *zi_array = ri[2];
    for (int i = start_idx; i < end_idx; ++i){
        double x = x_array[i];
        for (int j = 0; j < ny; ++j){
            double y = y_array[j];
            double sum = 0.0;
            for (int k = 0; k < nxi*nyi*nzi; ++k){
                int xidx = k / (nyi*nzi);
                int yidx = (k / nzi) % nyi;
                int zidx = k % nzi;
                int midx = xidx*nyi*nzi+yidx*nzi+zidx;
                double diffx = x - xi_array[xidx];
                double diffy = y - yi_array[yidx];
                double diffz = z - zi_array[zidx];
                double mx = mx_array[midx];
                double my = my_array[midx];
                double mz = mz_array[midx];
                double r_length_doubled = diffx*diffx+diffy*diffy+diffz*diffz;
                double r_length = std::sqrt(r_length_doubled);
                double r_length_cubed = r_length_doubled * r_length;
                double m_dot_r = mx*diffx+my*diffy+mz*diffz;
                if (r_length == 0){
                    continue;
                }
                switch (component_idx){
                    case 0:
                        sum += prop_const / r_length_doubled / r_length_cubed * (3 * m_dot_r*diffx-mx*r_length_doubled);
                        break;
                    case 1:
                        sum += prop_const / r_length_doubled / r_length_cubed * (3 * m_dot_r*diffy-my*r_length_doubled);
                        break;
                    case 2:
                        sum += prop_const / r_length_doubled / r_length_cubed * (3 * m_dot_r*diffz-mz*r_length_doubled);
                        break;
                    default:
                        break;
                }
            }
            answer_array[i * ny + j] = sum;
        }
    }

}

//A function that is bound to a python program
//mi_array_p is expected to be the dimension of [nxi, nyi, nzi]
//ansewr_array_p is expected to be the dimension of [nx, ny]
//calculation idx; 0: x component, 1: y component, 2: z component
void StrayFieldCalculator(
    py::array_t<double> answer_array_p,
    py::array_t<double> xi_array_p,
    py::array_t<double> yi_array_p,
    py::array_t<double> zi_array_p,
    py::array_t<double> x_array_p,
    py::array_t<double> y_array_p,
    py::array_t<double> mx_array_p,
    py::array_t<double> my_array_p,
    py::array_t<double> mz_array_p,
    double z_p,
    int calculation_idx,
    int num_threads = 4
){
    // Generating a thread pool
    std::vector<std::thread> threads;
    int num_threads_max = std::thread::hardware_concurrency();
    num_threads = (num_threads_max > num_threads) ? num_threads : num_threads_max;
    threads.reserve(num_threads);
    //Fetching buffer information
    py::buffer_info answer_buffer = answer_array_p.request();
    py::buffer_info xi_buffer = xi_array_p.request();
    py::buffer_info yi_buffer = yi_array_p.request();
    py::buffer_info zi_buffer = zi_array_p.request();
    py::buffer_info x_buffer = x_array_p.request();
    py::buffer_info y_buffer = y_array_p.request();
    py::buffer_info mx_buffer = mx_array_p.request();
    py::buffer_info my_buffer = my_array_p.request();
    py::buffer_info mz_buffer = mz_array_p.request();
    //Pointer retrival
    double * answer_array = static_cast<double*>(answer_buffer.ptr);
    double * x_array = static_cast<double*>(x_buffer.ptr);
    double * y_array = static_cast<double*>(y_buffer.ptr);
    double * xi_array = static_cast<double*>(xi_buffer.ptr);
    double * yi_array = static_cast<double*>(yi_buffer.ptr);
    double * zi_array = static_cast<double*>(zi_buffer.ptr);
    double * mx_array = static_cast<double*>(mx_buffer.ptr);
    double * my_array = static_cast<double*>(my_buffer.ptr);
    double * mz_array = static_cast<double*>(mz_buffer.ptr);
    double * zp_ptr = &z_p;
    //Number setup
    int nx = static_cast<int>(x_buffer.shape[0]);
    int ny = static_cast<int>(y_buffer.shape[0]);
    int nxi = static_cast<int>(xi_buffer.shape[0]);
    int nyi = static_cast<int>(yi_buffer.shape[0]);
    int nzi = static_cast<int>(zi_buffer.shape[0]);
    //Input Params Setup, pointer vector is used instead of pointer array
    //for the pybind11 compatibility issue
    int ni_array[3] = {nxi, nyi, nzi};
    std::vector<double*> r = {x_array, y_array, zp_ptr};
    std::vector<double*> ri = {xi_array, yi_array, zi_array};
    std::vector<double*> mag = {mx_array, my_array, mz_array};
    //Job allocation
    int idx_per_thread = nx/num_threads;
    int remainder = nx % num_threads;
    int current_idx = 0;
    //Switching the thread allocation upon the value of calculation index.
    //0: x component, 1: y component, 2: z component

    // Releasing Global Interpreter Lock. Released right before the multithreaded operation to avoid segfault.
    py::gil_scoped_release release;
    for (int t = 0; t < num_threads; ++t){
        int start_idx = current_idx;
        int end_idx = current_idx + idx_per_thread;
        if (t < remainder) {
            end_idx++;
        }
        current_idx = end_idx;
        threads.push_back(std::thread(StrayFieldCalculatorThreads, 
        answer_array,
        r,
        mag,
        ri,
        start_idx,
        end_idx,
        ny,
        ni_array,
        calculation_idx));
        
    }
    //Joining Each Threads
    for (auto& th: threads){
        if (th.joinable()) {th.join();}
    }
}
//Pybind11 Macro
PYBIND11_MODULE(_cxx_magcalc, m){
    m.def("straycalc", &StrayFieldCalculator, "Calculates stray field upon request. The specific component calculated is determined by an input integer 0 = x, 1 = y, 2 = z");
}