#include <bits/stdc++.h>
#include <chrono>
using namespace std;

const double INF = 1e18;

struct Edge {
    int to;
    vector<double> weights;
};

void read_graph(const string& filename, int& n, vector<vector<Edge>>& graph, int k, int& edge_count) {
    ifstream fin(filename);
    if (!fin.is_open()) {
        cout << "Error: Cannot open file " << filename << "\n";
        exit(1);
    }
    string line;
    bool is_mtx = filename.substr(filename.find_last_of(".") + 1) == "mtx";
    int offset = is_mtx ? 1 : 0; // 1-based for .mtx, 0-based for .txt
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
    int actual_edges = 0;
    while (getline(fin, line)) {
        if ((!is_mtx && line[0] == '#') || (is_mtx && line[0] == '%')) continue;
        stringstream ss(line);
        int u, v;
        if (ss >> u >> v) {
            u -= offset; // Convert to 0-based indexing
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
                if (is_mtx && u != v) graph[v].push_back({u, weights}); // Ensure undirected for .mtx
                actual_edges++;
            }
        }
    }

    cout << "Parsed nodes: " << n << ", Expected edges: " << edge_count << ", Actual edges added: " << actual_edges << "\n";
    if (actual_edges == 0) {
        cout << "Error: No edges were added to the graph.\n";
        exit(1);
    }
    if (actual_edges != edge_count) {
        cout << "Warning: Mismatch in edge counts! Check dataset format.\n";
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

void dijkstra(const vector<vector<Edge>>& graph, int source, int obj_index, vector<double>& dist, vector<int>& parent) {
    int n = graph.size();
    dist.assign(n, INF);
    parent.assign(n, -1);
    dist[source] = 0;
    priority_queue<pair<double,int>, vector<pair<double,int>>, greater<>> pq;
    pq.push({0, source});
    while (!pq.empty()) {
        auto [cost, u] = pq.top();
        pq.pop();
        if (cost > dist[u]) continue;
        for (const auto& edge : graph[u]) {
            int v = edge.to;
            double w = edge.weights[obj_index];
            if (dist[v] > dist[u] + w) {
                dist[v] = dist[u] + w;
                parent[v] = u;
                pq.push({dist[v], v});
            }
        }
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

void compute_sosp(const vector<vector<Edge>>& graph, int source, int target, int k, const vector<string>& objectives, vector<vector<int>>& parents, vector<vector<double>>& dists) {
    for (int i = 0; i < k; i++) {
        dijkstra(graph, source, i, dists[i], parents[i]);
        if (dists[i][target] < INF) {
            vector<int> path = get_path(parents[i], target);
            if (!path.empty() && path[0] == source) {
                cout << "SOSP path for " << objectives[i] << ": ";
                for (int node : path) cout << node << " ";
                cout << "\nCost for " << objectives[i] << ": " << dists[i][target] << "\n";
            } else {
                cout << "No valid SOSP path for " << objectives[i] << "\n";
            }
        } else {
            cout << "No SOSP path for " << objectives[i] << " from " << source << " to " << target << "\n";
        }
    }
}

void compute_mosp(const vector<vector<Edge>>& graph, int source, int target, int k, const vector<string>& objectives, const vector<vector<int>>& parents) {
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

int main() {
    string filename = "/home/hussain-ali/pdc-project/data/rgg_n_2_20_s0.mtx";
    int k = 3;
    vector<string> objectives = {"distance", "time", "energy"};
    int source = 0;
    int target = 100;

    auto start_total = chrono::high_resolution_clock::now();
    int n, edge_count;
    vector<vector<Edge>> graph;

    auto start_read = chrono::high_resolution_clock::now();
    read_graph(filename, n, graph, k, edge_count);
    auto end_read = chrono::high_resolution_clock::now();

    auto start_save = chrono::high_resolution_clock::now();
    save_graph(graph, "processed_graph.txt", edge_count);
    auto end_save = chrono::high_resolution_clock::now();

    auto start_compute = chrono::high_resolution_clock::now();
    vector<vector<int>> parents(k, vector<int>(n, -1));
    vector<vector<double>> dists(k, vector<double>(n, INF));
    compute_sosp(graph, source, target, k, objectives, parents, dists);
    compute_mosp(graph, source, target, k, objectives, parents);
    auto end_compute = chrono::high_resolution_clock::now();

    auto end_total = chrono::high_resolution_clock::now();

    cout << "Reading time: " << chrono::duration_cast<chrono::milliseconds>(end_read - start_read).count() << " ms\n";
    cout << "Saving time: " << chrono::duration_cast<chrono::milliseconds>(end_save - start_save).count() << " ms\n";
    cout << "Computation time: " << chrono::duration_cast<chrono::milliseconds>(end_compute - start_compute).count() << " ms\n";
    cout << "Total execution time: " << chrono::duration_cast<chrono::milliseconds>(end_total - start_total).count() << " ms\n";

    return 0;
}