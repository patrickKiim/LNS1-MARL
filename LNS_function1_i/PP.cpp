//coded by Patrick Kim @ MARMOT
//2023 09 03

#include "PP.h"
#include "common.h"
#include <iostream>


PP::PP(const Instance& instance) : instance(instance),
path_table(instance.map_size, instance.num_of_agents), goal_table(instance.map_size, -1)
{
    int N = instance.num_of_agents;
    agents.reserve(N); //vector.reserve: adjust capacity
    for (int i = 0; i < N; i++)
        agents.emplace_back(instance, i);  //  add element to the last place
    for (auto& i:agents) {
        goal_table[i.path_planner->goal_location] = i.id;
    }
}

bool PP::run()
{   //neighbor.agents.resize(agents.size());
    neighbor.agents.reserve(agents.size());
    sum_of_costs = 0;
    for (int i = 0; i < (int)agents.size(); i++)
    {
        if (!agents[i].path.empty())
        {
            cerr << "agent already has path"  << endl;
            exit(-1);
        }
        else
            neighbor.agents.push_back(i);
    }
    int remaining_agents = (int)neighbor.agents.size();
    std::random_shuffle(neighbor.agents.begin(), neighbor.agents.end());
    ConstraintTable constraint_table(instance.num_of_cols, instance.map_size, &path_table);
    set<pair<int, int>> colliding_pairs;
    for (auto id : neighbor.agents)
    {
        agents[id].path = agents[id].path_planner->findPath(constraint_table);
        assert(!agents[id].path.empty() && agents[id].path.back().location == agents[id].path_planner->goal_location);  // no hard obstacle, thus must find path
        if (agents[id].path_planner->num_collisions > 0)
            updateCollidingPairs(colliding_pairs, agents[id].id, agents[id].path);
        sum_of_costs += (int)agents[id].path.size() - 1;
        remaining_agents--;
        path_table.insertPath(agents[id].id, agents[id].path);

    }
    num_of_colliding_pairs = colliding_pairs.size();
    return remaining_agents == 0 && num_of_colliding_pairs==0;
}

vector<vector<pair<int,int>>> PP::get_all_paths()
{   vector<vector<pair<int,int>>>  vector_path(instance.num_of_agents,vector<pair<int,int>>(path_table.makespan+1, make_pair(-1,-1)));
    int a=0;

    for (const auto &agent : agents)
    {   int t=0;
        for (const auto &state : agent.path)
        {   vector_path[a][t]=instance.getCoordinate(state.location);
            t++;
        }
        vector_path[a].resize(t);
        a++;
    }
     return vector_path;
}

bool PP::updateCollidingPairs(set<pair<int, int>>& colliding_pairs, int agent_id, const Path& path) const
{
    bool succ = false;
    if (path.size() < 2)
        return succ;
    for (int t = 1; t < (int)path.size(); t++)
    {
        int from = path[t - 1].location;
        int to = path[t].location;
        if ((int)path_table.table[to].size() > t) // vertex conflicts
        {
            for (auto id : path_table.table[to][t])
            {
                succ = true;
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
                        succ = true;
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
                    succ = true;
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
            succ = true;
            colliding_pairs.emplace(min(agent_id, id), max(agent_id, id));
        }
    }
    return succ;
}




