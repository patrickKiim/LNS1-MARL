//MARMOT specific code
//2023 09 04
#pragma once
#include "LNS.h"
#include "PathTable.h"

class Selection
{
public:
    vector<Agent> agents;
    Neighbor neighbor;
    int num_agents;
    int makespan=0;
    vector<int> num_collision;
    Selection(const Instance& instance,const string& pathfile,int neighbor_size);

    vector<vector<pair<int,int>>> runPP();

private:
    const int neighbor_size;
    vector<vector<pair<int,int>>> vector_paths;
    const Instance& instance;
    string pathfile;
    void calculateColliding(int agent_id,int local_agent_id, const Path& path);

    PathTableWC path_table; // 1. stores the paths of all agents in a time-space table;
    // 2. avoid making copies of this variable as much as possible.
    inline int linearizeCoordinate(int row, int col,int row_size) const { return (row_size * row + col); }
    void build_element();
    bool loadpath();
};
