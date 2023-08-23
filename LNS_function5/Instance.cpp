#include<boost/tokenizer.hpp>
#include <algorithm>    // std::shuffle
#include <random>      // std::default_random_engine
#include <chrono>       // std::chrono::system_clock
#include"Instance.h"


Instance::Instance( const string& map_fname, const string& agent_fname,
                    int num_of_agents,int start, int goal, int self):
        map_fname(map_fname), agent_fname(agent_fname), num_of_agents(num_of_agents),start(start),goal(goal),self(self)
{
    bool succ = loadMap();
    if (!succ)
    {
        cerr << "loading map failed"  << endl;
        exit(-1);
    }

    succ = loadAgents();
    if (!succ)
    {
        cerr << "loading agents failed"  << endl;
        exit(-1);
    }

}

bool Instance::loadMap()
{
    using namespace boost;
    using namespace std;
    ifstream myfile(map_fname.c_str());
    if (!myfile.is_open())
        return false;
    string line;
    tokenizer< char_separator<char> >::iterator beg; // iterator pointer
    char_separator<char> sep(" ");
    getline(myfile, line);  // read one line, end untile meet /n
    tokenizer< char_separator<char> > tok(line, sep);
    beg = tok.begin();  // content is height
    beg++;
    num_of_rows = atoi((*beg).c_str());
    getline(myfile, line);
    tokenizer< char_separator<char> > tok2(line, sep);
    beg = tok2.begin();
    beg++;  // weight
    num_of_cols = atoi((*beg).c_str()); // read number of cols
    map_size = num_of_cols * num_of_rows;  // linearized
    my_map.resize(map_size, false);  // release rest space
    // read map (and start/goal locations)
    for (int i = 0; i < num_of_rows; i++) {
        getline(myfile, line);
        for (int j = 0; j < num_of_cols; j++) {
            my_map[linearizeCoordinate(i, j)] = (line[j] != '.'); // @=1 obstacle, .=0 empty, trasfer form 2 D to 1 D
        }
    }
    myfile.close();
    return true;
}

bool Instance::loadAgents()
{
    using namespace std;
    using namespace boost;

    string line;
    ifstream myfile (agent_fname.c_str());
    if (!myfile.is_open())
        return false;
    start_locations.resize(num_of_agents);
    goal_locations.resize(num_of_agents);
    int load_agent=num_of_agents-2;
    if (self==-1)
        load_agent=num_of_agents-1;

    char_separator<char> sep(" ");
    for (int i = 0; i < load_agent; i++)
    {
        getline(myfile, line);
        tokenizer< char_separator<char> > tok(line, sep);
        tokenizer< char_separator<char> >::iterator beg = tok.begin();
        int row = atoi((*beg).c_str());  // start col
        beg++;
        int col = atoi((*beg).c_str());  // start row
        start_locations[i] = linearizeCoordinate(row, col);
        // read goal [row,col] for agent i
        beg++;
        row = atoi((*beg).c_str());  // goal col
        beg++;
        col = atoi((*beg).c_str());  // goal row
        goal_locations[i] = linearizeCoordinate(row, col);
    }
    if (self!=-1)
    {   start_locations[num_of_agents-2]=self;
        goal_locations[num_of_agents-2]=self;}
    start_locations[num_of_agents-1]=start;
    goal_locations[num_of_agents-1]=goal;
    myfile.close();
    return true;
}

list<int> Instance::getNeighbors(int curr) const  // get truely moveable agent
{
	list<int> neighbors;
	int candidates[4] = {curr + 1, curr - 1, curr + num_of_cols, curr - num_of_cols};  // right, left, up,down
	for (int next : candidates)  // for next in candidates
	{
		if (validMove(curr, next))
			neighbors.emplace_back(next);
	}
	return neighbors;
}

