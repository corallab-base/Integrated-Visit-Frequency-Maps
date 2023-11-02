#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

inline int ravel(int i, int j, int num_cols) {
    return i * num_cols + j;
}

py::tuple spfa_dense_source(py::array_t<bool> input_map, py::array_t<bool> source) {
    // Input and source maps must be equivalently sized 
    py::buffer_info map_buf = input_map.request();
    int num_rows = map_buf.shape[0];
    int num_cols = map_buf.shape[1];
    bool* map_ptr = (bool *) map_buf.ptr;

    py::buffer_info source_buf = source.request();
    int source_rows = source_buf.shape[0];
    int source_cols = source_buf.shape[1];
    bool* source_ptr = (bool *) source_buf.ptr;

    if (num_rows != source_rows || num_cols != source_cols)
        throw std::invalid_argument("'input_map' and 'source' must share same shape");

    const float eps = 1e-6;
    const int num_dirs = 8;
    const int dirs[num_dirs][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}};
    const float dir_lengths[num_dirs] = {std::sqrt(2.0f), 1, std::sqrt(2.0f), 1, std::sqrt(2.0f), 1, std::sqrt(2.0f), 1};

    // Process input map
    int max_num_verts = num_rows * num_cols;
    int max_edges_per_vert = num_dirs;
    const float inf = 2 * max_num_verts;
    int queue_size = num_dirs * num_rows * num_cols;

    // Initialize arrays
    int* edges = new int[max_num_verts * max_edges_per_vert]();
    int* edge_counts = new int[max_num_verts]();
    int* queue = new int[queue_size]();
    bool* in_queue = new bool[max_num_verts]();
    float* weights = new float[max_num_verts * max_edges_per_vert]();
    float* dists = new float[max_num_verts];
    for (int i = 0; i < max_num_verts; ++i)
        dists[i] = inf;
    int* parents = new int[max_num_verts]();
    for (int i = 0; i < max_num_verts; ++i)
        parents[i] = -1;

    // Build graph
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            int v = ravel(i, j, num_cols);
            if (!map_ptr[v])
                continue;
            for (int k = 0; k < num_dirs; ++k) {
                int ip = i + dirs[k][0], jp = j + dirs[k][1];
                if (ip < 0 || jp < 0 || ip >= num_rows || jp >= num_cols)
                    continue;
                int vp = ravel(ip, jp, num_cols);
                if (!map_ptr[vp])
                    continue;
                int e = ravel(v, edge_counts[v], max_edges_per_vert);
                edges[e] = vp;
                weights[e] = dir_lengths[k];
                edge_counts[v]++;
            }
        }
    }

    int head = 0, tail = 0;

    // Copy sources to dist map  
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            int v = ravel(i, j, num_cols);
            if (source_ptr[v]) {
                dists[v] = 0;
                queue[++tail] = v;
                in_queue[v] = true;
            }
        }
    }

    // SPFA
    while (head < tail) {
        int u = queue[++head];
        in_queue[u] = false;
        for (int j = 0; j < edge_counts[u]; ++j) {
            int e = ravel(u, j, max_edges_per_vert);
            int v = edges[e];
            float new_dist = dists[u] + weights[e];
            if (new_dist < dists[v]) {
                parents[v] = u;
                dists[v] = new_dist;
                if (!in_queue[v]) {
                    assert(tail < queue_size);
                    queue[++tail] = v;
                    in_queue[v] = true;
                    if (dists[queue[tail]] < dists[queue[head + 1]])
                        std::swap(queue[tail], queue[head + 1]);
                }
            }
        }
    }

    // Copy output into numpy array
    auto output_dists = py::array_t<float>({num_rows, num_cols});
    auto output_parents = py::array_t<int>({num_rows, num_cols});
    py::buffer_info dists_buf = output_dists.request(), parents_buf = output_parents.request();
    float* dists_ptr = (float *) dists_buf.ptr;
    int* parents_ptr = (int *) parents_buf.ptr;
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            int u = ravel(i, j, num_cols);
            dists_ptr[u] = (dists[u] < inf - eps) * dists[u];
            parents_ptr[u] = parents[u];
        }
    }

    // Free memory
    delete[] edges;
    delete[] edge_counts;
    delete[] queue;
    delete[] in_queue;
    delete[] weights;
    delete[] dists;
    delete[] parents;

    return py::make_tuple(output_dists, output_parents);
}

PYBIND11_MODULE(spfa, m) {
    m.doc() = R"pbdoc(
        SPFA implemented in C++
    )pbdoc";

    m.def("spfa_dense_source", &spfa_dense_source, R"pbdoc(
        spfa_dense_source
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
