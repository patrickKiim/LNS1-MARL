//
// Created by marmot on 10/04/23.
//
#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp> // transfer character to token
#include "PP.h"
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

                // params for the input instance and experiment settings
                ("map,m", po::value<string>()->required(), "input file for map")
                ("agents,a", po::value<string>()->required(), "input file for agents")
                ("outputPaths,o", po::value<string>()->required(), "output file for statistics")
                ("number,n", po::value<int>()->required(), "number of agents");
//                ("map,m", po::value<string>()->default_value("/home/marmot/Yutong/MAPF/LNS2+RL_allow_collide/record_files/world1.txt"), "input file for map")
//                ("agents,a", po::value<string>()->default_value("/home/marmot/Yutong/MAPF/LNS2+RL_allow_collide/record_files/pos1.txt"), "input file for agents")
//                ("outputPaths,o", po::value<string>()->default_value("/home/marmot/Yutong/MAPF/LNS2+RL_allow_collide/record_files/pp_output.txt"), "output file for statistics")
//                ("number,n", po::value<int>()->default_value(460), "number of agents");
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        Instance instance(vm["map"].as<string>(), vm["agents"].as<string>(),
                          vm["number"].as<int>()); //map and relative operation, similar to environment
        PP pp(instance);
        bool succ=pp.run();
        vector<vector<pair<int,int>>> vector_path=pp.get_all_paths();  //makespan =size-1
        int makespan=pp.all_path_makespan();
        int num_of_colliding_pairs=pp.num_of_colliding_pairs;

        ofstream stats(vm["outputPaths"].as<string>(), std::ios::out | std::ios::trunc);
        std::ostringstream converter;
        converter << succ;
        stats << string(converter.str())<< endl;
        stats << std::to_string(makespan)<< endl;
        stats << std::to_string(num_of_colliding_pairs)<< endl;
        for  (int i = 0; i < vm["number"].as<int>(); i++)
        {
            vector<pair<int,int>> temp= vector_path[i];
            for (vector<pair<int,int>>::iterator it =temp.begin(); it != temp.end();  it++)
            {   pair<int,int> temp_pair=*it;
                stats << std::to_string(temp_pair.first)<<" "<<std::to_string(temp_pair.second)<<"-";
            }
            stats <<endl;
        }
        stats.close();

        return 0;
    }
