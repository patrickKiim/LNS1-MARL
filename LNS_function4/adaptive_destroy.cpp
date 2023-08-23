//
// Created by marmot on 10/04/23.
//
#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp> // transfer character to token
#include "neighbor_selection.h"
#include "Instance.h"
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
            ("map,m", po::value<string>()->required(), "input file for map")
            ("agents,a", po::value<string>()->required(), "input file for agents")
            ("paths,p", po::value<string>()->required(), "input file for paths")
            ("outputPaths,o", po::value<string>()->required(), "output file for statistics")
            ("number,n", po::value<int>()->required(), "number of agents")
            ("numberagents,g", po::value<int>()->required(), "id of the selection method");
//            ("map,m", po::value<string>()->default_value("/home/marmot/Yutong/MAPF/training/cl_step_block_individual_ifbetter/record_files/world0.txt"), "input file for map")
//            ("agents,a", po::value<string>()->default_value("/home/marmot/Yutong/MAPF/training/cl_step_block_individual_ifbetter/record_files/pos0.txt"), "input file for agents")
//            ("paths,p", po::value<string>()->default_value("/home/marmot/Yutong/MAPF/training/cl_step_block_individual_ifbetter/record_files/cl_paths0.txt"), "input file for paths")
//            ("outputPaths,o", po::value<string>()->default_value("/home/marmot/Yutong/MAPF/training/cl_step_block_team_ifbetter/record_files/f4_output.txt"), "output file for statistics")
//            ("number,n", po::value<int>()->default_value(4), "number of agents")
//            ("numberagents,g", po::value<int>()->default_value(30), "id of the selection method");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
//
    po::notify(vm);

    Instance instance(vm["map"].as<string>(), vm["agents"].as<string>(),
                      vm["numberagents"].as<int>()); //map and relative operation, similar to environment
    Selection selection(instance,vm["paths"].as<string>(),vm["number"].as<int>());
    vector<vector<pair<int,int>>> sipp_paths(vm["number"].as<int>(),vector<pair<int,int>>(10, make_pair(-1,-1)));
    sipp_paths=selection.runPP();
    ofstream stats(vm["outputPaths"].as<string>(), std::ios::out | std::ios::trunc);
    stats << std::to_string(selection.makespan)<< endl;
    for (int i = 0; i < vm["number"].as<int>(); i++)
        stats << std::to_string(selection.num_collision[i])<< " ";
    stats << endl;
    for  (int i = 0; i < vm["number"].as<int>(); i++)
    {
        vector<pair<int,int>> temp= sipp_paths[i];
        for (vector<pair<int,int>>::iterator it =temp.begin(); it != temp.end();  it++)
        {   pair<int,int> temp_pair=*it;
            stats << std::to_string(temp_pair.first)<<" "<<std::to_string(temp_pair.second)<<"-";
        }
        stats <<endl;
    }
    stats.close();

    return 0;
}
