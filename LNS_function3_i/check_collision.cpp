//marmot specific code
//2023 09 05
#include "check_collision.h"
#include <queue>
#include <algorithm>
#include<boost/tokenizer.hpp>


Check::Check(const string& pathfile, int map_size,int num_agents) :
pathfile(pathfile),map_size(map_size),num_agents(num_agents),path_table(map_size*map_size, num_agents)
{
    bool succ =loadpath();
    if (!succ) {
        cerr << "loading path failed" << endl;
        exit(-1);
    }
}

bool Check::loadpath()
{
    using namespace boost;
    using namespace std;
    ifstream myfile(pathfile.c_str());
    if (!myfile.is_open())
        return false;
    string line;
    char_separator<char> sep("-");
    char_separator<char> sep2(" ");

    vector_paths.resize(num_agents);
    for (int i = 0; i < num_agents; i++) {
        getline(myfile, line);
        tokenizer< char_separator<char> > tok(line, sep);
        for (tokenizer< char_separator<char> >::iterator beg=tok.begin(); beg!=tok.end(); beg++) {
            string temp=*beg;
            tokenizer< char_separator<char> > tok2(temp, sep2);
            tokenizer< char_separator<char> >::iterator temp_beg=tok2.begin();
            temp_beg++;
            vector_paths[i].emplace_back(make_pair(atoi((*tok2.begin()).c_str()),atoi((*temp_beg).c_str()))); // @=1 obstacle, .=0 empty, trasfer form 2 D to 1 D
        }
    }

    myfile.close();
    return true;}

int Check::calculate_collision()
{
    agents.reserve(num_agents); //vector.reserve: adjust capacity
    set<pair<int, int>> colliding_pairs;
    for(int id=0;id<num_agents;id++)
    {
        agents.emplace_back(id);
        agents[id].path.resize(vector_paths[id].size());
        for (int t=0;t<(int)vector_paths[id].size();t++)
        {
            agents[id].path[t].location=linearizeCoordinate(vector_paths[id][t].first, vector_paths[id][t].second, map_size);
        }
        updateCollidingPairs(colliding_pairs, id, agents[id].path);
        path_table.insertPath(id, agents[id].path);
    }
    num_of_colliding_pairs = colliding_pairs.size();
    return num_of_colliding_pairs;

}

void Check::updateCollidingPairs(set<pair<int, int>>& colliding_pairs, int agent_id, const Path& path) const
{
    if (path.size() < 2)
        return;
    for (int t = 1; t < (int)path.size(); t++)
    {
        int from = path[t - 1].location;
        int to = path[t].location;
        if ((int)path_table.table[to].size() > t) // vertex conflicts
        {
            for (auto id : path_table.table[to][t])
            {
                colliding_pairs.emplace(min(agent_id, id), max(agent_id, id));// emplace: insert new element into set
            }
        }
        if (from != to && path_table.table[to].size() >= t && path_table.table[from].size() > t) // edge conflicts(swapping conflicts)
        {
            for (auto a1 : path_table.table[to][t - 1])
            {
                for (auto a2: path_table.table[from][t])
                {
                    if (a1 == a2)
                    {
                        colliding_pairs.emplace(min(agent_id, a1), max(agent_id, a1));
                        break;
                    }
                }
            }
        }
        //auto id = getAgentWithTarget(to, t);
        //if (id >= 0) // this agent traverses the target of another agent
        //    colliding_pairs.emplace(min(agent_id, id), max(agent_id, id));
        if (!path_table.goals.empty() && path_table.goals[to] < t) // target conflicts, already has agent in its goal, so the new agent can not tarverse it
        { // this agent traverses the target of another agent
            for (auto id : path_table.table[to][path_table.goals[to]]) // look at all agents at the goal time
            {
                if (agents[id].path.back().location == to) // if agent id's goal is to, then this is the agent we want
                {
                    colliding_pairs.emplace(min(agent_id, id), max(agent_id, id));
                    break;
                }
            }
        }
    }
    int goal = path.back().location; // target conflicts - some other agent traverses the target of this agent
    for (int t = (int)path.size(); t < path_table.table[goal].size(); t++)
    {
        for (auto id : path_table.table[goal][t])
        {
            colliding_pairs.emplace(min(agent_id, id), max(agent_id, id));
        }
    }
}

