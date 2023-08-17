#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp>
// #include "LNS.h"
// #include "AnytimeBCBS.h"
#include "AnytimeEECBS.h"
// #include "PIBT/pibt.h"


/* Main function */
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
        ("outputPaths,o", po::value<string>(), "output file name (no extension)")
		("agentNum,n", po::value<int>()->default_value(8), "number of agents");

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);

	if (vm.count("help")) {
		cout << desc << endl;
		return 1;
	}

    po::notify(vm);

	srand((int)time(0));

	Instance instance(vm["map"].as<string>(), vm["agents"].as<string>(),
		vm["agentNum"].as<int>());
        
    double time_limit = (double)7200;//vm["cutoffTime"].as<double>();
    int screen = 0;//vm["screen"].as<int>();
	srand(0);

    AnytimeEECBS eecbs(instance, time_limit, screen);
    vector<vector<pair<int,int>>> paths(vm["agentNum"].as<int>(),vector<pair<int,int>>(10, make_pair(-1,-1)));
    paths = eecbs.run();   // return path from here !
    eecbs.validateSolution();
    ofstream stats(vm["outputPaths"].as<string>(), std::ios::out | std::ios::trunc);
    // if (vm.count("output"))
    //         eecbs.writeResultToFile(vm["output"].as<string>() + ".csv");
    // if (vm.count("stats"))
    //         eecbs.writeIterStatsToFile(vm["stats"].as<string>());

    for  (int i = 0; i < vm["agentNum"].as<int>(); i++)
    {
        vector<pair<int,int>> temp= paths[i];
        for (vector<pair<int,int>>::iterator it =temp.begin(); it != temp.end();  it++)
        {   pair<int,int> temp_pair=*it;
            stats << std::to_string(temp_pair.first)<<" "<<std::to_string(temp_pair.second)<<"-";
        }
        stats <<endl;
    }
    stats.close();
	return 0;

}
