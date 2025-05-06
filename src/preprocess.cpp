#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>

int main() {
    srand(time(0));

    std::string inputFile = "data/roadNet-CA.txt";
    std::string outputFile = "data/roadNet-CA-weighted.txt";

    std::ifstream inFile(inputFile);
    if (!inFile.is_open()) {
        std::cerr << "Error opening input file: " << inputFile << std::endl;
        return 1;
    }

    std::ofstream outFile(outputFile);
    if (!outFile.is_open()) {
        std::cerr << "Error opening output file: " << outputFile << std::endl;
        return 1;
    }

    int fromNode, toNode;
    while (inFile >> fromNode >> toNode) {
        double w1 = rand() % 100 / 100.0;
        double w2 = rand() % 100 / 100.0;
        double w3 = rand() % 100 / 100.0;

        outFile << fromNode << " " << toNode << " "
                << w1 << " " << w2 << " " << w3 << std::endl;
    }

    inFile.close();
    outFile.close();

    std::cout << "Preprocessing complete. Weighted graph saved to: " << outputFile << std::endl;
    return 0;
}
