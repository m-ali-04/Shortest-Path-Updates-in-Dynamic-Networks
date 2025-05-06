#include <bits/stdc++.h>
#include <chrono>
#include <mpi.h>
#include <metis.h>
using namespace std;

const double INF = 1e18;

struct Edge {
    int to;
    vector<double> weights;
};

struct IncomingEdge {
    int u;
    int v;
    vector<double> weights;
};

void read_graph(const string& filename, int& n, vector<vector<Edge>>& graph, int k, int& edge_count, vector<idx_t>& xadj, vector<idx_t>& adjncy) {
    ifstream fin(filename);
    if (!fin.is_open()) {
        cout << "Error: Cannot open file " << filename << "\n";
        exit(1);
    }
    string line;
    bool is_mtx = filename.substr(filename.find_last_of(".") + 1) == "mtx";
    int offset = is_mtx ? 1 : 0;
    map<pair<int,int>, vector<double>> edge_weights;
    default_random_engine generator;
    uniform_real_distribution<double> distribution(1.0, 10.0);

    n = 0;
    edge_count = 0;
    if (!is_mtx) {
        while (getline(fin, line)) {
            if (line.find("# Nodes:") != string::npos) {
                size_t pos1 = line.find("Nodes:") + 6;
                size_t pos2 = line.find("Edges:", pos1);
                if (pos2 != string::npos) {
                    string nodes_str = line.substr(pos1, pos2 - pos1);
                    string edges_str = line.substr(pos2 + 6);
                    stringstream ss_nodes(nodes_str);
                    ss_nodes >> n;
                    stringstream ss_edges(edges_str);
                    ss_edges >> edge_count;
                }
                break;
            }
        }
    } else {
        while (getline(fin, line)) {
            if (line[0] != '%') {
                stringstream ss(line);
                int rows, cols, nonzeros;
                ss >> rows >> cols >> nonzeros;
                n = rows;
                edge_count = nonzeros;
                break;
            }
        }
    }

    if (n == 0 || edge_count == 0) {
        cout << "Error: Failed to parse node or edge count from " << filename << "\n";
        exit(1);
    }

    graph.resize(n);
    vector<vector<int>> adj_list(n);
    int actual_edges = 0;
    while (getline(fin, line)) {
        if ((!is_mtx && line[0] == '#') || (is_mtx && line[0] == '%')) continue;
        stringstream ss(line);
        int u, v;
        if (ss >> u >> v) {
            u -= offset;
            v -= offset;
            if (u >= 0 && u < n && v >= 0 && v < n) {
                pair<int,int> key = {min(u,v), max(u,v)};
                if (edge_weights.find(key) == edge_weights.end()) {
                    vector<double> weights(k);
                    for (int i = 0; i < k; i++) weights[i] = distribution(generator);
                    edge_weights[key] = weights;
                }
                const auto& weights = edge_weights[key];
                graph[u].push_back({v, weights});
                adj_list[u].push_back(v);
                if (is_mtx && u != v) {
                    graph[v].push_back({u, weights});
                    adj_list[v].push_back(u);
                }
                actual_edges++;
            }
        }
    }

    xadj.resize(n + 1);
    xadj[0] = 0;
    for (int u = 0; u < n; u++) {
        xadj[u + 1] = xadj[u] + adj_list[u].size();
        for (int v : adj_list[u]) {
            adjncy.push_back(v);
        }
    }

    if (actual_edges != edge_count) {
        cout << "Warning: Edge count mismatch! Expected: " << edge_count << ", Added: " << actual_edges << "\n";
    }
}

void save_graph(const vector<vector<Edge>>& graph, const string& filename, int edge_count) {
    ofstream fout(filename);
    fout << "# Processed graph with weights for objectives: distance, time, energy\n";
    fout << "# Nodes: " << graph.size() << "\n";
    fout << "# Directed Edges: " << edge_count << "\n";
    fout << "# FromNodeId ToNodeId w_distance w_time w_energy\n";
    for (int u = 0; u < graph.size(); u++) {
        for (const auto& edge : graph[u]) {
            int v = edge.to;
            const auto& weights = edge.weights;
            fout << u << " " << v << " ";
            for (double w : weights) fout << w << " ";
            fout << "\n";
        }
    }
    fout.close();
}

void partition_graph(int n, const vector<idx_t>& xadj, const vector<idx_t>& adjncy, vector<idx_t>& part, int size) {
    idx_t ncon = 1;
    idx_t nparts = size;
    idx_t objval;
    int ret = METIS_PartGraphKway(&n, &ncon, const_cast<idx_t*>(xadj.data()), const_cast<idx_t*>(adjncy.data()),
                                  NULL, NULL, NULL, &nparts, NULL, NULL, NULL, &objval, part.data());
    if (ret != METIS_OK) {
        cout << "METIS partitioning failed\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void distribute_graph(const vector<vector<Edge>>& graph, vector<vector<IncomingEdge>>& incoming_edges, const vector<idx_t>& part, int rank, int size, int n, int k) {
    incoming_edges.resize(n);
    if (rank == 0) {
        vector<vector<IncomingEdge>> incoming_lists(size);
        for (int u = 0; u < n; u++) {
            for (const auto& edge : graph[u]) {
                int v = edge.to;
                int P = part[v];
                IncomingEdge ie = {u, v, edge.weights};
                incoming_lists[P].push_back(ie);
            }
        }
        for (int P = 0; P < size; P++) {
            int num_edges = incoming_lists[P].size();
            MPI_Send(&num_edges, 1, MPI_INT, P, 0, MPI_COMM_WORLD);
            for (const auto& ie : incoming_lists[P]) {
                MPI_Send(&ie.u, 1, MPI_INT, P, 1, MPI_COMM_WORLD);
                MPI_Send(&ie.v, 1, MPI_INT, P, 2, MPI_COMM_WORLD);
                MPI_Send(ie.weights.data(), k, MPI_DOUBLE, P, 3, MPI_COMM_WORLD);
            }
        }
    }
    int num_edges;
    MPI_Recv(&num_edges, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int i = 0; i < num_edges; i++) {
        IncomingEdge ie;
        MPI_Recv(&ie.u, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&ie.v, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        ie.weights.resize(k);
        MPI_Recv(ie.weights.data(), k, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        incoming_edges[ie.v].push_back(ie);
    }
}

void dijkstra_parallel(const vector<vector<Edge>>& graph, const vector<vector<IncomingEdge>>& incoming_edges, int source, int obj_index, vector<double>& dist, vector<int>& parent, const vector<idx_t>& part, int rank, int size) {
    int n = graph.size();
    dist.assign(n, INF);
    parent.assign(n, -1);
    dist[source] = 0;
    vector<char> settled(n, 0);

    while (true) {
        double local_min_dist = INF;
        int local_min_vertex = -1;
        for (int v = 0; v < n; v++) {
            if (!settled[v] && dist[v] < local_min_dist) {
                local_min_dist = dist[v];
                local_min_vertex = v;
            }
        }

        struct MinData { double dist; int vertex; int rank; } local_data = {local_min_dist, local_min_vertex, rank};
        vector<MinData> all_data(size);
        MPI_Allgather(&local_data, sizeof(MinData), MPI_BYTE, all_data.data(), sizeof(MinData), MPI_BYTE, MPI_COMM_WORLD);

        double global_min_dist = INF;
        int global_min_vertex = -1, min_rank = -1;
        for (int i = 0; i < size; i++) {
            if (all_data[i].dist < global_min_dist) {
                global_min_dist = all_data[i].dist;
                global_min_vertex = all_data[i].vertex;
                min_rank = all_data[i].rank;
            }
        }
        if (global_min_dist == INF) break;

        settled[global_min_vertex] = 1;

        for (int v = 0; v < n; v++) {
            if (part[v] == rank) {
                for (const auto& ie : incoming_edges[v]) {
                    int u = ie.u;
                    if (u == global_min_vertex && !settled[v]) {
                        double w = ie.weights[obj_index];
                        if (dist[v] > dist[u] + w) {
                            dist[v] = dist[u] + w;
                            parent[v] = u;
                        }
                    }
                }
            }
        }

        MPI_Allreduce(MPI_IN_PLACE, dist.data(), n, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, parent.data(), n, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    }
}

vector<int> get_path(const vector<int>& parent, int target) {
    vector<int> path;
    int current = target;
    while (current != -1) {
        path.push_back(current);
        current = parent[current];
    }
    reverse(path.begin(), path.end());
    return path.size() > 1 ? path : vector<int>();
}

void compute_sosp(const vector<vector<Edge>>& graph, const vector<vector<IncomingEdge>>& incoming_edges, int source, int target, int k, const vector<string>& objectives, vector<vector<int>>& parents, vector<vector<double>>& dists, const vector<idx_t>& part, int rank, int size) {
    for (int i = 0; i < k; i++) {
        dijkstra_parallel(graph, incoming_edges, source, i, dists[i], parents[i], part, rank, size);
        if (rank == 0 && dists[i][target] < INF) {
            vector<int> path = get_path(parents[i], target);
            if (!path.empty() && path[0] == source) {
                cout << "SOSP path for " << objectives[i] << ": ";
                for (int node : path) cout << node << " ";
                cout << "\nCost for " << objectives[i] << ": " << dists[i][target] << "\n";
            } else {
                cout << "No valid SOSP path for " << objectives[i] << "\n";
            }
        }
    }
}

void compute_mosp(const vector<vector<Edge>>& graph, int source, int target, int k, const vector<string>& objectives, const vector<vector<int>>& parents, int rank) {
    if (rank != 0) return;

    set<pair<int,int>> B_edges;
    map<pair<int,int>, int> edge_count;
    for (int i = 0; i < k; i++) {
        for (int v = 0; v < graph.size(); v++) {
            if (parents[i][v] != -1) {
                int u = parents[i][v];
                B_edges.insert({u, v});
                edge_count[{u, v}]++;
            }
        }
    }
    vector<vector<pair<int,double>>> B(graph.size());
    for (const auto& edge : B_edges) {
        int u = edge.first, v = edge.second;
        double weight = k - edge_count[{u, v}] + 1;
        B[u].push_back({v, weight});
    }

    vector<double> dist_B(graph.size(), INF);
    vector<int> parent_B(graph.size(), -1);
    priority_queue<pair<double,int>, vector<pair<double,int>>, greater<>> pq;
    dist_B[source] = 0;
    pq.push({0, source});
    while (!pq.empty()) {
        auto [cost, u] = pq.top();
        pq.pop();
        if (cost > dist_B[u]) continue;
        for (const auto& neighbor : B[u]) {
            int v = neighbor.first;
            double w = neighbor.second;
            if (dist_B[v] > dist_B[u] + w) {
                dist_B[v] = dist_B[u] + w;
                parent_B[v] = u;
                pq.push({dist_B[v], v});
            }
        }
    }

    if (dist_B[target] < INF) {
        vector<int> path = get_path(parent_B, target);
        if (!path.empty() && path[0] == source) {
            cout << "MOSP path: ";
            for (int node : path) cout << node << " ";
            cout << "\n";
            for (int i = 0; i < k; i++) {
                double cost = 0;
                for (int j = 0; j < path.size() - 1; j++) {
                    int u = path[j], v = path[j + 1];
                    auto it = find_if(graph[u].begin(), graph[u].end(), [v](const Edge& e) { return e.to == v; });
                    if (it != graph[u].end()) cost += it->weights[i];
                }
                cout << "Cost for " << objectives[i] << ": " << cost << "\n";
            }
        } else {
            cout << "No valid MOSP path\n";
        }
    } else {
        cout << "No MOSP path from " << source << " to " << target << "\n";
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    string filename = "/mirror/proj/data/roadNet-PA.txt";
    int k = 3;
    vector<string> objectives = {"distance", "time", "energy"};
    int source = 0;
    int target = 100;

    auto start_total = chrono::high_resolution_clock::now();
    int n = 0, edge_count = 0;
    vector<vector<Edge>> graph;
    vector<idx_t> xadj, adjncy, part;

    auto start_read = chrono::high_resolution_clock::now();
    if (rank == 0) {
        read_graph(filename, n, graph, k, edge_count, xadj, adjncy);
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&edge_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) graph.resize(n);

    if (rank == 0) {
        partition_graph(n, xadj, adjncy, part, size);
    }
    part.resize(n);
    MPI_Bcast(part.data(), n, MPI_INT, 0, MPI_COMM_WORLD);

    vector<vector<IncomingEdge>> incoming_edges;
    distribute_graph(graph, incoming_edges, part, rank, size, n, k);

    auto end_read = chrono::high_resolution_clock::now();

    auto start_save = chrono::high_resolution_clock::now();
    if (rank == 0) {
        save_graph(graph, "processed_graph.txt", edge_count);
    }
    auto end_save = chrono::high_resolution_clock::now();

    auto start_compute = chrono::high_resolution_clock::now();
    vector<vector<int>> parents(k, vector<int>(n, -1));
    vector<vector<double>> dists(k, vector<double>(n, INF));
    compute_sosp(graph, incoming_edges, source, target, k, objectives, parents, dists, part, rank, size);
    compute_mosp(graph, source, target, k, objectives, parents, rank);
    auto end_compute = chrono::high_resolution_clock::now();

    auto end_total = chrono::high_resolution_clock::now();

    if (rank == 0) {
        cout << "Reading time: " << chrono::duration_cast<chrono::milliseconds>(end_read - start_read).count() << " ms\n";
        cout << "Saving time: " << chrono::duration_cast<chrono::milliseconds>(end_save - start_save).count() << " ms\n";
        cout << "Computation time: " << chrono::duration_cast<chrono::milliseconds>(end_compute - start_compute).count() << " ms\n";
        cout << "Total execution time: " << chrono::duration_cast<chrono::milliseconds>(end_total - start_total).count() << " ms\n";
    }

    MPI_Finalize();
    return 0;
}
