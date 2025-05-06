#include <iostream>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <queue>
#include <stack>

const double INF = std::numeric_limits<double>::infinity();

// Define a structure to represent each edge
struct Edge {
    int toNode;
    std::vector<double> weights;
};

class Graph {
public:
    std::unordered_map<int, std::vector<Edge>> adjList;
    std::unordered_map<int, std::vector<double>> dist;
    std::unordered_map<int, std::vector<int>> parent;

    void initializeFromWeightedFile(const std::string &filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return;
        }

        int fromNode, toNode;
        double w1, w2, w3;
        std::unordered_set<int> nodeSet;
        int edgeCount = 0;

        while (file >> fromNode >> toNode >> w1 >> w2 >> w3) {
            std::vector<double> objectiveWeights = {w1, w2, w3};

            adjList[fromNode].push_back({toNode, objectiveWeights});
            adjList[toNode].push_back({fromNode, objectiveWeights});  // assuming undirected

            nodeSet.insert(fromNode);
            nodeSet.insert(toNode);
            edgeCount++;

            if (dist.find(fromNode) == dist.end()) {
                dist[fromNode] = std::vector<double>(3, INF);
                parent[fromNode] = std::vector<int>(3, -1);
            }
            if (dist.find(toNode) == dist.end()) {
                dist[toNode] = std::vector<double>(3, INF);
                parent[toNode] = std::vector<int>(3, -1);
            }
        }
        file.close();

        std::cout << "Total nodes read: " << nodeSet.size() << std::endl;
        std::cout << "Total edges read: " << edgeCount << std::endl;
    }

    void printFirst10Nodes() const {
        int count = 0;
        for (const auto &node : adjList) {
            if (count == 10) break;
            std::cout << "Node " << node.first << " -> ";
            for (const auto &edge : node.second) {
                std::cout << "(" << edge.toNode << ": ";
                for (const auto &weight : edge.weights) {
                    std::cout << weight << " ";
                }
                std::cout << ") ";
            }
            std::cout << std::endl;
            count++;
        }
    }

    void runSOSP(int source, int objectiveIndex) {
        for (auto &d : dist) {
            d.second[objectiveIndex] = INF;
            parent[d.first][objectiveIndex] = -1;
        }
        dist[source][objectiveIndex] = 0;

        std::queue<int> q;
        q.push(source);

        while (!q.empty()) {
            int u = q.front();
            q.pop();

            for (const auto &edge : adjList[u]) {
                int v = edge.toNode;
                double weight = edge.weights[objectiveIndex];

                if (dist[u][objectiveIndex] + weight < dist[v][objectiveIndex]) {
                    dist[v][objectiveIndex] = dist[u][objectiveIndex] + weight;
                    parent[v][objectiveIndex] = u;
                    q.push(v);
                }
            }
        }
    }

    void runMOSP(int source) {
        for (int i = 0; i < 3; ++i) {
            runSOSP(source, i);
        }
    }

    void printDistances(int objectiveIndex, int target) const {
        std::cout << "Objective " << objectiveIndex << " → ";
        if (dist.find(target) != dist.end()) {
            std::cout << "Distance to node " << target << ": " << dist.at(target)[objectiveIndex] << std::endl;
        } else {
            std::cout << "Target node not found.\n";
        }
    }

    void printPath(int objectiveIndex, int source, int target) const {
        std::stack<int> path;
        int current = target;

        while (current != -1) {
            path.push(current);
            current = parent.at(current)[objectiveIndex];
        }

        if (path.top() != source) {
            std::cout << "No path found from " << source << " to " << target << " for objective " << objectiveIndex << ".\n";
            return;
        }

        std::cout << "Path (objective " << objectiveIndex << "): ";
        while (!path.empty()) {
            std::cout << path.top();
            path.pop();
            if (!path.empty()) std::cout << " -> ";
        }
        std::cout << std::endl;
    }
};

int main() {
    std::cout << "Running on weighted graph...\n";

    Graph graph;
    graph.initializeFromWeightedFile("data/roadNet-CA-weighted.txt");

    graph.printFirst10Nodes();

    int sourceNode = 0;
    int targetNode = 100;

    std::cout << "\nRunning MOSP (all objectives)...\n";
    graph.runMOSP(sourceNode);

    for (int i = 0; i < 3; ++i) {
        graph.printDistances(i, targetNode);
        graph.printPath(i, sourceNode, targetNode);
    }

    return 0;
}
