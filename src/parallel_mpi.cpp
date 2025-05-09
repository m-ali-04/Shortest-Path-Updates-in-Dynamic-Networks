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
    int u, v;
    vector<double> weights;
};

void log_message(int rank, const string& message) {
    cout << "[Rank " << rank << "] " << message << endl;
}

void read_graph(const string& filename, int& n, vector<vector<Edge>>& graph, int k, int& edge_count, vector<idx_t>& xadj, vector<idx_t>& adjncy, int rank) {
    log_message(rank, "Starting graph reading...");
    ifstream fin(filename);
    if (!fin.is_open() && rank == 0) {
        cout << "Error: Cannot open file " << filename << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    string line;
    bool is_mtx = filename.substr(filename.find_last_of(".") + 1) == "mtx";
    int offset = is_mtx ? 1 : 0;
    map<pair<int,int>, vector<double>> edge_weights;
    default_random_engine generator;
    uniform_real_distribution<double> distribution(1.0, 10.0);

    n = 0;
    edge_count = 0;
    if (rank == 0) {
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
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    log_message(rank, "Before broadcasting n and edge_count...");
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&edge_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    log_message(rank, "After broadcasting, n: " + to_string(n) + ", edge_count: " + to_string(edge_count));

    graph.resize(n);
    vector<set<int>> adj_set(n); // Use set to avoid duplicate edges
    int actual_edges = 0;
    if (rank == 0) {
        while (getline(fin, line)) {
            if (line.empty() || (!is_mtx && line[0] == '#') || (is_mtx && line[0] == '%')) continue;
            stringstream ss(line);
            int u, v;
            if (!(ss >> u >> v)) continue; // Skip invalid lines
            u -= offset;
            v -= offset;
            if (u >= 0 && u < n && v >= 0 && v < n && u != v) { // Skip self-loops
                pair<int,int> key = {min(u,v), max(u,v)};
                if (edge_weights.find(key) == edge_weights.end()) {
                    vector<double> weights(k);
                    for (int i = 0; i < k; i++) weights[i] = distribution(generator);
                    edge_weights[key] = weights;
                }
                const auto& weights = edge_weights[key];
                if (adj_set[u].insert(v).second) { // Only add if not already present
                    graph[u].push_back({v, weights});
                    actual_edges++;
                }
                if (is_mtx && adj_set[v].insert(u).second) {
                    graph[v].push_back({u, weights});
                    actual_edges++;
                }
            }
        }
        xadj.resize(n + 1);
        xadj[0] = 0;
        for (int u = 0; u < n; u++) {
            xadj[u + 1] = xadj[u] + adj_set[u].size();
            for (int v : adj_set[u]) adjncy.push_back(v);
        }
        if (actual_edges != edge_count) {
            cout << "Warning: Edge count mismatch! Expected: " << edge_count << ", Added: " << actual_edges << "\n";
        }
    }
    log_message(rank, "Graph reading completed, actual_edges: " + to_string(actual_edges));
    MPI_Barrier(MPI_COMM_WORLD);
    log_message(rank, "Passed barrier after graph reading");
}

void save_graph(const vector<vector<Edge>>& graph, const string& filename, int edge_count, int rank) {
    log_message(rank, "Starting graph saving...");
    if (rank == 0) {
        ofstream fout(filename);
        fout << "# Processed graph with weights for objectives: distance, time, energy\n";
        fout << "# Nodes: " << graph.size() << "\n";
        fout << "# Directed Edges: " << edge_count << "\n";
        fout << "# FromNodeId ToNodeId w_distance w_time w_energy\n";
        for (int u = 0; u < graph.size(); u++) {
            for (const auto& edge : graph[u]) {
                fout << u << " " << edge.to << " ";
                for (double w : edge.weights) fout << w << " ";
                fout << "\n";
            }
        }
        fout.close();
    }
    log_message(rank, "Graph saving completed.");
}

void partition_graph(int n, const vector<idx_t>& xadj, const vector<idx_t>& adjncy, vector<idx_t>& part, int size, int rank) {
    log_message(rank, "Starting graph partitioning...");
    if (rank == 0) {
        idx_t ncon = 1, nparts = size, objval;
        int ret = METIS_PartGraphKway(&n, &ncon, const_cast<idx_t*>(xadj.data()), const_cast<idx_t*>(adjncy.data()),
                                      NULL, NULL, NULL, &nparts, NULL, NULL, NULL, &objval, part.data());
        if (ret != METIS_OK) {
            cout << "METIS partitioning failed\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    log_message(rank, "Graph partitioning completed.");
}

void distribute_graph(const vector<vector<Edge>>& graph, vector<vector<IncomingEdge>>& incoming_edges, const vector<idx_t>& part, int rank, int size, int n, int k) {
    log_message(rank, "Starting graph distribution...");
    incoming_edges.resize(n);
    if (rank == 0) {
        vector<vector<IncomingEdge>> incoming_lists(size);
        for (int u = 0; u < n; u++) {
            for (const auto& edge : graph[u]) {
                int v = edge.to;
                int P = part[v];
                incoming_lists[P].push_back({u, v, edge.weights});
            }
        }
        vector<MPI_Request> requests;
        for (int P = 0; P < size; P++) {
            int num_edges = incoming_lists[P].size();
            log_message(rank, "Sending " + to_string(num_edges) + " edges to rank " + to_string(P));
            MPI_Request req;
            MPI_Isend(&num_edges, 1, MPI_INT, P, 0, MPI_COMM_WORLD, &req);
            requests.push_back(req);
            if (num_edges > 0) {
                vector<int> u_list(num_edges), v_list(num_edges);
                vector<double> weights_list(num_edges * k);
                for (int i = 0; i < num_edges; i++) {
                    u_list[i] = incoming_lists[P][i].u;
                    v_list[i] = incoming_lists[P][i].v;
                    for (int j = 0; j < k; j++) {
                        weights_list[i * k + j] = incoming_lists[P][i].weights[j];
                    }
                }
                MPI_Isend(u_list.data(), num_edges, MPI_INT, P, 1, MPI_COMM_WORLD, &req);
                requests.push_back(req);
                MPI_Isend(v_list.data(), num_edges, MPI_INT, P, 2, MPI_COMM_WORLD, &req);
                requests.push_back(req);
                MPI_Isend(weights_list.data(), num_edges * k, MPI_DOUBLE, P, 3, MPI_COMM_WORLD, &req);
                requests.push_back(req);
            }
        }
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
        log_message(rank, "All sends completed.");
    }
    int num_edges;
    log_message(rank, "Waiting to receive edge count...");
    MPI_Recv(&num_edges, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    log_message(rank, "Received " + to_string(num_edges) + " edges.");
    if (num_edges > 0) {
        vector<int> u_list(num_edges), v_list(num_edges);
        vector<double> weights_list(num_edges * k);
        MPI_Recv(u_list.data(), num_edges, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(v_list.data(), num_edges, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(weights_list.data(), num_edges * k, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < num_edges; i++) {
            IncomingEdge ie;
            ie.u = u_list[i];
            ie.v = v_list[i];
            ie.weights.resize(k);
            for (int j = 0; j < k; j++) {
                ie.weights[j] = weights_list[i * k + j];
            }
            incoming_edges[ie.v].push_back(ie);
        }
    }
    log_message(rank, "Graph distribution completed.");
    MPI_Barrier(MPI_COMM_WORLD);
    log_message(rank, "Passed barrier after graph distribution");
}

void dijkstra_parallel(const vector<vector<IncomingEdge>>& incoming_edges, int source, int obj_index, vector<double>& dist, vector<int>& parent, const vector<idx_t>& part, int rank, int size, int n) {
    log_message(rank, "Starting parallel Dijkstra for objective " + to_string(obj_index));
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
        int global_min_vertex = -1;
        for (int i = 0; i < size; i++) {
            if (all_data[i].dist < global_min_dist) {
                global_min_dist = all_data[i].dist;
                global_min_vertex = all_data[i].vertex;
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
    log_message(rank, "Parallel Dijkstra for objective " + to_string(obj_index) + " completed.");
}

vector<int> get_path(const vector<int>& parent, int target) {
    vector<int> path;
    for (int v = target; v != -1; v = parent[v]) path.push_back(v);
    reverse(path.begin(), path.end());
    return path.size() > 1 ? path : vector<int>();
}

void compute_sosp(const vector<vector<IncomingEdge>>& incoming_edges, int source, int target, int k, const vector<string>& objectives, vector<vector<int>>& parents, vector<vector<double>>& dists, const vector<idx_t>& part, int rank, int size, int n) {
    log_message(rank, "Starting SOSP computation...");
    for (int i = 0; i < k; i++) {
        dijkstra_parallel(incoming_edges, source, i, dists[i], parents[i], part, rank, size, n);
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
    log_message(rank, "SOSP computation completed.");
}

void compute_mosp(const vector<vector<Edge>>& graph, int source, int target, int k, const vector<string>& objectives, const vector<vector<int>>& parents, int rank, int n) {
    log_message(rank, "Starting MOSP computation...");
    if (rank != 0) return;
    set<pair<int,int>> B_edges;
    map<pair<int,int>, int> edge_count;
    for (int i = 0; i < k; i++) {
        for (int v = 0; v < n; v++) {
            if (parents[i][v] != -1) {
                int u = parents[i][v];
                B_edges.insert({u, v});
                edge_count[{u, v}]++;
            }
        }
    }
    vector<vector<pair<int,double>>> B(n);
    for (const auto& edge : B_edges) {
        int u = edge.first, v = edge.second;
        double weight = k - edge_count[{u, v}] + 1;
        B[u].push_back({v, weight});
    }

    vector<double> dist_B(n, INF);
    vector<int> parent_B(n, -1);
    priority_queue<pair<double,int>, vector<pair<double,int>>, greater<>> pq;
    dist_B[source] = 0;
    pq.push({0, source});
    while (!pq.empty()) {
        auto [cost, u] = pq.top(); pq.pop();
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
                    for (const auto& e : graph[u]) {
                        if (e.to == v) { cost += e.weights[i]; break; }
                    }
                }
                cout << "Cost for " << objectives[i] << ": " << cost << "\n";
            }
        } else {
            cout << "No valid MOSP path\n";
        }
    } else {
        cout << "No MOSP path from " << source << " to " << target << "\n";
    }
    log_message(rank, "MOSP computation completed.");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    string filename = "/mirror/proj/data/roadNet-PA.txt";
    int k = 3;
    vector<string> objectives = {"distance", "time", "energy"};
    int source = 0, target = 100;

    auto start = chrono::high_resolution_clock::now();
    int n = 0, edge_count = 0;
    vector<vector<Edge>> graph;
    vector<idx_t> xadj, adjncy, part(n);

    read_graph(filename, n, graph, k, edge_count, xadj, adjncy, rank);
    part.resize(n);
    partition_graph(n, xadj, adjncy, part, size, rank);
    log_message(rank, "Before broadcasting part array...");
    MPI_Bcast(part.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
    log_message(rank, "After broadcasting part array");

    vector<vector<IncomingEdge>> incoming_edges;
    distribute_graph(graph, incoming_edges, part, rank, size, n, k);
    save_graph(graph, "processed_graph.txt", edge_count, rank);

    vector<vector<int>> parents(k, vector<int>(n, -1));
    vector<vector<double>> dists(k, vector<double>(n, INF));
    compute_sosp(incoming_edges, source, target, k, objectives, parents, dists, part, rank, size, n);
    compute_mosp(graph, source, target, k, objectives, parents, rank, n);

    auto end = chrono::high_resolution_clock::now();
    if (rank == 0) {
        cout << "Total execution time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms\n";
    }

    MPI_Finalize();
    return 0;
}
