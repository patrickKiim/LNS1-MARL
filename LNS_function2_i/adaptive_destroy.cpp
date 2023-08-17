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
            ("ALNS,l", po::value<bool>()->required(), "if use adaptive selection")
            ("collisions,c", po::value<int>()->required(), "number of collisions")
            ("update,u", po::value<bool>()->required(), "if update weight")
            ("selection,s", po::value<int>()->required(), "id of the selection method")
            ("numberagents,g", po::value<int>()->required(), "id of the selection method");
//            ("map,m", po::value<string>()->default_value("/home/marmot/Yutong/sipps_window/record_files/world1.txt"), "input file for map")
//            ("agents,a", po::value<string>()->default_value("/home/marmot/Yutong/sipps_window/record_files/pos1.txt"), "input file for agents")
//            ("paths,p", po::value<string>()->default_value("/home/marmot/Yutong/sipps_window/record_files/paths1.txt"), "input file for paths")
//            ("outputPaths,o", po::value<string>()->default_value("/home/marmot/Yutong/sipps_window/record_files/selection_output1.txt"), "output file for statistics")
//            ("number,n", po::value<int>()->default_value(8), "number of agents")
//            ("ALNS,l", po::value<bool>()->default_value(true), "if use adaptive selection")
//            ("collisions,c", po::value<int>()->default_value(41), "number of collisions")
//            ("update,u", po::value<bool>()->default_value(false), "if update weight")
//            ("selection,s", po::value<int>()->default_value(0), "id of the selection method")
//            ("numberagents,g", po::value<int>()->default_value(400), "id of the selection method");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
        cout << desc << endl;
        return 1;
    }

    po::notify(vm);

    srand((int)time(0));

    Instance instance(vm["map"].as<string>(), vm["agents"].as<string>(),
                      vm["numberagents"].as<int>()); //map and relative operation, similar to environment
    Selection selection(instance,vm["paths"].as<string>(), vm["ALNS"].as<bool>() ,vm["number"].as<int>(),vm["collisions"].as<int>(),"Collision",vm["update"].as<bool>(),vm["selection"].as<int>());
    bool succ =selection.run(); //succ if the input path already done
    vector<int> choosed_agent;
    vector<vector<pair<int,int>>> sipp_paths(vm["number"].as<int>(),vector<pair<int,int>>(10, make_pair(-1,-1)));
    if (succ)
    {   choosed_agent.assign(vm["number"].as<int>(),-1);
        }
    else
    {   choosed_agent=selection.neighbor.agents;
        sipp_paths=selection.runPP();}

    int global_num_collison=selection.curr_num_of_colliding_pairs;
    vector<double> curr_destroy_weights=selection.destroy_weights;
    int neighbor_method=selection.selected_neighbor;
    int makespan=selection.all_path_makespan();
    ofstream stats(vm["outputPaths"].as<string>(), std::ios::out | std::ios::trunc);
    stats << std::to_string(global_num_collison)<< endl;
    for(int i=0;i<3;i++)
    {
        stats<<std::to_string(curr_destroy_weights[i])<<" ";
    }
    stats << endl;

    for (vector<int>::iterator it = choosed_agent.begin(); it != choosed_agent.end();  it++)
    {
        stats<<std::to_string(*it)<<" ";
    }
    stats << endl;
    std::ostringstream converter;
    converter << succ;
    stats << string(converter.str())<< endl;
    stats << std::to_string(neighbor_method)<< endl;
    stats << std::to_string(makespan)<< endl;
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