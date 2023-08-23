#include "neighbor_selection.h"
#include <queue>
#include <algorithm>
#include<boost/tokenizer.hpp>


Selection::Selection(const Instance& instance,const string& pathfile, int self) :
instance(instance),num_agents(instance.num_of_agents),
path_table(instance.map_size, num_agents),pathfile(pathfile),self(self),load_agent(num_agents-2)
{
    if (self==-1)
        load_agent=num_agents-1;
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
    vector_paths.resize(load_agent);
    char_separator<char> sep2("-");
    for (int i = 0; i < load_agent; i++) {
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
    for(int id=0;id<load_agent;id++)
    {
        agents.emplace_back(instance, id);
        agents[id].path.resize(vector_paths[id].size());
        for (int t=0;t<(int)vector_paths[id].size();t++)
        {
            agents[id].path[t].location=linearizeCoordinate(vector_paths[id][t].first, vector_paths[id][t].second, instance.num_of_rows);
        }
        path_table.insertPath(id, agents[id].path);
    }
    if (self!=-1)
    {
        agents.emplace_back(instance, num_agents-2);
        agents[num_agents-2].path.resize(1);
        agents[num_agents-2].path[0].location=self;
        path_table.insertPath(num_agents-2, agents[num_agents-2].path);
    }
    agents.emplace_back(instance, num_agents-1);
}


vector<pair<int,int>> Selection::runPP()
{
    build_element();
    vector<pair<int,int>> temp(10000, make_pair(-1,-1));
    ConstraintTable constraint_table(instance.num_of_cols, instance.map_size, &path_table);
    int id =num_agents-1;
    agents[id].path = agents[id].path_planner->findPath(constraint_table);
    assert(!agents[id].path.empty() && agents[id].path.back().location == agents[id].path_planner->goal_location);
    int t=0;
    for (const auto &state : agents[id].path)
    {   temp[t]=instance.getCoordinate(state.location);
        t++;
    }
    temp.resize(t);
    return temp;
}



