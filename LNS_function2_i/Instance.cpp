#include<boost/tokenizer.hpp>
#include <algorithm>    // std::shuffle
#include <random>      // std::default_random_engine
#include <chrono>       // std::chrono::system_clock
#include"Instance.h"


Instance::Instance( const string& map_fname, const string& agent_fname,
                    int num_of_agents):
        map_fname(map_fname), agent_fname(agent_fname), num_of_agents(num_of_agents)
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
    char_separator<char> sep(" ");
    for (int i = 0; i < num_of_agents; i++)
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

bool Instance::validateSolution(const vector<Path*>& paths, int sum_of_costs, int num_of_colliding_pairs) const
{
    if (paths.size() != start_locations.size())  // number of paths=number of agents
    {
        cerr << "We have " << paths.size() << " for " << start_locations.size() << " agents." << endl;
        exit(-1);
    }
    int sum = 0;
    for (auto i = 0; i < start_locations.size(); i++)
    {
        if (paths[i] == nullptr or paths[i]->empty())
        {
            cerr << "No path for agent " << i << endl;
            exit(-1);
        }
        else if (start_locations[i] != paths[i]->front().location)
        {
            cerr << "The path of agent " << i << " starts from location " << paths[i]->front().location
                 << ", which is different from its start location " << start_locations[i] << endl;
            exit(-1);
        }
        else if (goal_locations[i] != paths[i]->back().location)
        {
//            return false;
            cerr << "The path of agent " << i << " ends at location " << paths[i]->back().location
                 << ", which is different from its goal location " << goal_locations[i] << endl;
            exit(-1);
        }
        for (int t = 1; t < (int) paths[i]->size(); t++ )  // in one path
        {
            if (!validMove(paths[i]->at(t - 1).location, paths[i]->at(t).location))  //invalid: not in one step distacle, on obstacles, exceed map
            {
                cerr << "The path of agent " << i << " jumps from "
                     << paths[i]->at(t - 1).location << " to " << paths[i]->at(t).location
                     << " between timesteps " << t - 1 << " and " << t << endl;
                exit(-1);
            }
        }
        sum += (int) paths[i]->size() - 1;
    }
    if (sum_of_costs != sum)
    {
        cerr << "The computed sum of costs " << sum_of_costs <<
             " is different from that of the solution " << sum << endl;
        exit(-1);
    }
    // check for colliions
    int collisions = 0;
    for (auto i = 0; i < start_locations.size(); i++)  //for every agent
    {
        for (auto j = i + 1; j < start_locations.size(); j++)  // check if the number of colllison equal to the true number of collision
        {
            bool found_collision = false;
            const auto a1 = paths[i]->size() <= paths[j]->size()? i : j; //smaller
            const auto a2 = paths[i]->size() <= paths[j]->size()? j : i;  //higher
            int t = 1;
            for (; t < (int) paths[a1]->size(); t++)
            {
                if (paths[a1]->at(t).location == paths[a2]->at(t).location) // vertex conflict
                {
                    if (num_of_colliding_pairs == 0)
                    {
                        cerr << "Find a vertex conflict between agents " << a1 << " and " << a2 <<
                             " at location " << paths[a1]->at(t).location << " at timestep " << t << endl;
                        exit(-1);
                    }
                    collisions++;
                    found_collision = true;
                    return false;
                }
                else if (paths[a1]->at(t).location == paths[a2]->at(t-1).location &&
                         paths[a1]->at(t-1).location == paths[a2]->at(t).location) // edge conflict
                {
                    if (num_of_colliding_pairs == 0)
                    {
                        cerr << "Find an edge conflict between agents " << a1 << " and " << a2 <<
                             " at edge (" << paths[a1]->at(t-1).location << "," << paths[a1]->at(t).location <<
                             ") at timestep " << t << endl;
                        exit(-1);
                    }
                    collisions++;
                    found_collision = true;
                    return false;
                }
            }
            if (!found_collision)  // target collision, passby the target of agent whose timestep already done
            {
                auto target = paths[a1]->back().location;
                for (; t < (int) paths[a2]->size(); t++)
                {
                    if (paths[a2]->at(t).location == target)  // target conflict
                    {
                        if (num_of_colliding_pairs == 0)
                        {
                            cerr << "Find a target conflict where agent " << a2 << " (of length " << paths[a2]->size() - 1 <<
                                 ") traverses agent " << a1 << " (of length " << paths[a1]->size() - 1<<
                                 ")'s target location " << target << " at timestep " << t << endl;
                            exit(-1);
                        }
                        collisions++;
                        return false;
                    }
                }
            }
        }
    }

    return true;
}