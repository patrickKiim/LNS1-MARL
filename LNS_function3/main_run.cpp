//
// Created by marmot on 10/04/23.
//
#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp> // transfer character to token
#include "check_collision.h"
#include "string"
#include "vector"
#include <sstream>

int main(int argc, char** argv)
{
    namespace po = boost::program_options;
    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
            ("help", "produce help message")

//             params for the input instance and experiment settings
            ("map,m", po::value<int>()->required(), "map size")
            ("paths,p", po::value<string>()->required(), "input file for paths")
            ("outputPaths,o", po::value<string>()->required(), "output file for statistics")
            ("number,n", po::value<int>()->required(), "number of agents");
//            ("map,m", po::value<int>()->default_value(40), "map size")
//            ("paths,p", po::value<string>()->default_value("/home/marmot/Yutong/MAPF/replace_path_easy/record_files/temp_paths1.txt"), "input file for paths")
//            ("outputPaths,o", po::value<string>()->default_value("/home/marmot/Yutong/MAPF/replace_path_easy/record_files/check_output.txt"), "output file for statistics")
//            ("number,n", po::value<int>()->default_value(400), "number of agents");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    po::notify(vm);

    Check Check(vm["paths"].as<string>(),vm["map"].as<int>(),vm["number"].as<int>());
    int num_collision=Check.calculate_collision(); //succ if the input path already done

    ofstream stats(vm["outputPaths"].as<string>(), std::ios::out | std::ios::trunc);
    stats << std::to_string(num_collision)<< endl;
    stats.close();

    return 0;
}
