//MARMOT specific code
// 09/12/24
#include "neighbor_selection.h"
#include <queue>
#include <algorithm>
#include<boost/tokenizer.hpp>


Selection::Selection(const Instance& instance,const string& pathfile,int neighbor_size) :
instance(instance),neighbor_size(neighbor_size),num_agents(instance.num_of_agents),
path_table(instance.map_size, num_agents),pathfile(pathfile)
{
    bool succ =loadpath();
    if (!succ) {
        cerr << "loading path failed" << endl;
        exit(-1);
    }
}


bool Selection::loadpath()
{
    using namespace boost;
    using namespace std;
    ifstream myfile(pathfile.c_str());
    if (!myfile.is_open())
        return false;
    string line;
    tokenizer< char_separator<char> >::iterator beg; // iterator pointer
    char_separator<char> sep(" ");
    getline(myfile, line);  // read one line, end untile meet /n
    tokenizer< char_separator<char> > tok(line, sep);
    beg=tok.begin();
    neighbor.agents.resize(neighbor_size);
    for (int i = 0; i < neighbor_size; i++) {
        neighbor.agents[i]=atoi((*beg).c_str());
        beg++;
    }
//    for (tokenizer< char_separator<char> >::iterator beg=tok.begin(); beg!=tok.end(); beg++) {
//        neighbor.agents[stod((*beg).c_str())];
//    }
    vector_paths.resize(num_agents);

    char_separator<char> sep2("-");
    for (int i = 0; i < num_agents; i++) {
        getline(myfile, line);
        tokenizer< char_separator<char> > tok2(line, sep2);
        for (tokenizer< char_separator<char> >::iterator beg=tok2.begin(); beg!=tok2.end(); beg++) {
            string temp=*beg;
            tokenizer< char_separator<char> > tok3(temp, sep);
            tokenizer< char_separator<char> >::iterator temp_beg=tok3.begin();
            temp_beg++;
            vector_paths[i].emplace_back(make_pair(atoi((*tok3.begin()).c_str()),atoi((*temp_beg).c_str()))); // @=1 obstacle, .=0 empty, trasfer form 2 D to 1 D
        }
    }
    myfile.close();
    return true;}

void Selection::build_element()
{
    agents.reserve(num_agents); //vector.reserve: adjust capacity
    for(int id=0;id<num_agents;id++)
    {
        agents.emplace_back(instance, id);
        agents[id].path.resize(vector_paths[id].size());
        for (int t=0;t<(int)vector_paths[id].size();t++)
        {
            agents[id].path[t].location=linearizeCoordinate(vector_paths[id][t].first, vector_paths[id][t].second, instance.num_of_rows);
        }
        path_table.insertPath(id, agents[id].path);
    }
}


vector<vector<pair<int,int>>> Selection::runPP()
{
    build_element();
    for (int i = 0; i < (int)neighbor.agents.size(); i++)
    {
        path_table.deletePath(neighbor.agents[i]);
    }
    makespan=path_table.makespan;
    auto shuffled_agents = neighbor.agents;
    auto p = shuffled_agents.begin();
    vector<vector<pair<int,int>>> temp(neighbor.agents.size(),vector<pair<int,int>>(10000, make_pair(-1,-1)));
    ConstraintTable constraint_table(instance.num_of_cols, instance.map_size, &path_table);
    int a=0;
    while (p != shuffled_agents.end())
    {
        int id = *p;
        agents[id].path = agents[id].path_planner->findPath(constraint_table);
        assert(!agents[id].path.empty() && agents[id].path.back().location == agents[id].path_planner->goal_location);
        path_table.insertPath(agents[id].id, agents[id].path);
        int t=0;
        for (const auto &state : agents[id].path)
        {   temp[a][t]=instance.getCoordinate(state.location);
            t++;
        }
        temp[a].resize(t);
        a++;
        ++p;
    }
    int i =0;
    num_collision.assign(neighbor_size,0);
    auto local_p = shuffled_agents.begin();
    while (local_p != shuffled_agents.end())
    {   int local_id = *local_p;
        calculateColliding(local_id,i,agents[local_id].path);
        ++local_p;
        i++;
    }
    return temp;
}


void Selection::calculateColliding(int agent_id,int local_agent_id, const Path& path)
{
    for (int t = 1; t < (int)path.size(); t++)
    {
        int from = path[t - 1].location;
        int to = path[t].location;
        if ((int)path_table.table[to].size() > t) // vertex conflicts
        {
            for (auto id : path_table.table[to][t])
            {   if (id!= agent_id)
                    num_collision[local_agent_id]+=1;// emplace: insert new element into set
            }
        }
        if (from != to && path_table.table[to].size() >= t && path_table.table[from].size() > t) // edge conflicts(swapping conflicts)
        {
            for (auto a1 : path_table.table[to][t - 1])
            {
                for (auto a2: path_table.table[from][t])
                {
                    if (a1 == a2 and a1!=agent_id)
                    {
                        num_collision[local_agent_id]+=1;
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
                {   if (id!= agent_id)
                    {   num_collision[local_agent_id]+=1;
                        break;}
                }
            }
        }
    }
    int goal = path.back().location; // target conflicts - some other agent traverses the target of this agent
    for (int t = (int)path.size(); t < path_table.table[goal].size(); t++)
    {
        for (auto id : path_table.table[goal][t])
        {
            if (id!= agent_id)
                num_collision[local_agent_id]+=1;
        }
    }
}


