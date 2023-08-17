#pragma once
#include "BasicLNS.h"
#include "PathTable.h"


class Check
{
public:
    vector<Agent> agents;
    int num_of_colliding_pairs = 0;  //calculate at this step
    int num_agents;
    int map_size;

    Check(const string& pathfile, int map_size,int num_agents);

    int calculate_collision();

private:
    vector<vector<pair<int,int>>> vector_paths;
    string pathfile;

    PathTableWC path_table; // 1. stores the paths of all agents in a time-space table;
    // 2. avoid making copies of this variable as much as possible.
    inline int linearizeCoordinate(int row, int col,int row_size) const { return (row_size * row + col); }
    void updateCollidingPairs(set<pair<int, int>>& colliding_pairs, int agent_id, const Path& path) const;
    bool loadpath();
};
