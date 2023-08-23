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
            ("start,s", po::value<int>()->required(), "agent start poss")
            ("goal,g", po::value<int>()->required(), "agent goal poss")
            ("self,e", po::value<int>()->required(), "self poss");
//            ("map,m", po::value<string>()->default_value("/home/marmot/Yutong/MAPF/training/cl_step_block_team_succ/record_files/world0.txt"), "input file for map")
//            ("agents,a", po::value<string>()->default_value("/home/marmot/Yutong/MAPF/training/cl_step_block_team_succ/record_files/part_pos0.txt"), "input file for agents")
//            ("paths,p", po::value<string>()->default_value("/home/marmot/Yutong/MAPF/training/cl_step_block_team_succ/record_files/part_path0.txt"), "input file for path")
//            ("outputPaths,o", po::value<string>()->default_value("/home/marmot/Yutong/MAPF/training/cl_step_block_team_succ/record_files/f5_output.txt"), "output file for statistics")
//            ("number,n", po::value<int>()->default_value(28), "number of agents")
//            ("start,s", po::value<int>()->default_value(20), "agent start poss")
//            ("goal,g", po::value<int>()->default_value(332), "agent goal poss")
//            ("self,e", po::value<int>()->default_value(10), "self poss");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    po::notify(vm);

    Instance instance(vm["map"].as<string>(), vm["agents"].as<string>(),
                      vm["number"].as<int>(),vm["start"].as<int>(),vm["goal"].as<int>(),vm["self"].as<int>()); //map and relative operation, similar to environment
    Selection selection(instance,vm["paths"].as<string>(),vm["self"].as<int>());

    vector<pair<int,int>> sipp_path(10, make_pair(-1,-1));
    sipp_path=selection.runPP();
    ofstream stats(vm["outputPaths"].as<string>(), std::ios::out | std::ios::trunc);

    for (vector<pair<int,int>>::iterator it =sipp_path.begin(); it != sipp_path.end();  it++)
    {   pair<int,int> temp_pair=*it;
        stats << std::to_string(temp_pair.first)<<" "<<std::to_string(temp_pair.second)<<"-";
    }
    stats.close();

    return 0;
}
