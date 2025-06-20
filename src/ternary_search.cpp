#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

//Command for energy, example: "MLIPcalc -mlip=[MLIP] -model=[model]" or "MLIPmd -mlip=[MLIP]= -model=[model]"
std::string command;


// Function to call Python script and get energy
double md(double x) {
    std::ifstream fin("POSCAR");
    if (!fin) {
        std::cerr << "Error opening POSCAR for reading\n";
        exit(1);
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(fin, line)) {
        lines.push_back(line);
    }
    fin.close();

    if (lines.size() < 2) {
        std::cerr << "POSCAR file empty!\n";
        exit(1);
    }

    // Replace the 2nd line with x
    lines[1] = std::to_string(x);

    std::ofstream fout("POSCAR");
    if (!fout) {
        std::cerr << "Error opening POSCAR for writing\n";
        exit(1);
    }

    for (const auto& l : lines) {
        fout << l << "\n";
    }
    fout.close();

    int status = system(command.c_str());
    if (status != 0) {
        std::cerr << "Command run failed\n";
        exit(1);
    }

    std::ifstream fin2("energy");
    if (!fin2) {
        std::cerr << "Error reading  energy\n";
        exit(1);
    }

    double energy;
    fin2 >> energy;
    fin2.close();
    return energy;
}

// Ternary search: finds minimum of unimodal function in [left, right]
double ternary_search(double left, double right, double eps) {
    int iter = 0;
    while (right - left > eps) {
        double m1 = left + (right - left) / 3.0;
        double m2 = right - (right - left) / 3.0;
        double f1 = md(m1);
        double f2 = md(m2);

        if (f1 < f2)
            right = m2;
        else
            left = m1;

    	std::cout << "Iter " << iter
                  << ": left = " << left
                  << ", m1 = " << m1 << ", energy(m1) = " << f1
                  << ", m2 = " << m2 << ", energy(m2) = " << f2
                  << ", right = " << right << std::endl;
	iter++;
    }
    return (left + right) / 2.0;
}

int main(int argc, char* argv[]){

    
    double lower_limit = 0.7;
    double upper_limit = 1.3;
    double epsilon = 1e-2;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);

        if (arg.find("-ll=") == 0) {
            lower_limit = std::stod(arg.substr(4));
        } else if (arg.find("-ul=") == 0) {
            upper_limit = std::stod(arg.substr(4));
        } else if (arg.find("-eps=") == 0) {
            epsilon = std::stod(arg.substr(5));
        } else if (arg.find("-c=") == 0) {
            command = arg.substr(3);
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return 1;
        }
    }

    // Show parsed values
    std::cout << "Lower limit: " << lower_limit << std::endl;
    std::cout << "Upper limit: " << upper_limit << std::endl;
    std::cout << "Epsilon    : " << epsilon << std::endl;
    std::cout << "Command    : " << command << std::endl;	
	
	
	
    double min_x = ternary_search(lower_limit, upper_limit,epsilon);
    double min_energy = md(min_x);

    std::cout << "Minimum found at x = " << min_x << std::endl;
    std::cout << "Minimum energy = " << min_energy << std::endl;

    // Also write to file if needed
    std::ofstream fout("energy");
    fout << min_energy << std::endl;
    fout.close();

    return 0;
}
