#pragma once

#include "argparse/argparse.hpp"
#include "commons.hpp"

// parameters define
struct args_params_t : public argparse::Args {
    bool& results = kwarg("results", "print generated results (default: false)").set_default(true);
    bool& help = flag("h, help", "print help");
    bool& time = kwarg("t, time", "print time").set_default(true);
    std::string& input_file = kwarg("input_file", "path to input file").set_default("");
};

// Function to read data from a text file and store it in a vector
template <typename T>
std::vector<T> readDataFromFile(const std::string& filename) {
    std::vector<T> data;

    // Open the file
    std::ifstream file(filename);

    // Check if the file is open successfully
    if (!file.is_open()) {
        fmt::print("Failed to open the file: {}\n", filename);
        return data;  // Return an empty vector in case of failure
    }

    std::string line;
    while (std::getline(file, line)) {
        // to parse each line into doubles and store them in the vector
        T value;
        std::istringstream iss(line);
        while (iss >> value) {
            data.push_back(value);
        }
    }

    // Close the file
    file.close();

    return data;
}
