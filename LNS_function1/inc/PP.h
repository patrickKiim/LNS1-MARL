//coded by Patrick Kim @ MARMOT
//2023 09 03

#pragma once
#include "common.h"
#include "Instance.h"
#include "LNS.h"
#include "string"

class PP
{
public:
    int sum_of_costs = 0;
    int num_of_colliding_pairs = 0;

    PP(const Instance& instance);
    bool run();
    vector<vector<pair<int,int>>> get_all_paths();
    int all_path_makespan() const {return path_table.makespan;};

private:
    const Instance& instance;
    vector<Agent> agents;
    vector<int> goal_table;  // location-> agent id
    Neighbor neighbor;
    PathTableWC path_table;
    bool updateCollidingPairs(set<pair<int, int>>& colliding_pairs, int agent_id, const Path& path) const;
};
