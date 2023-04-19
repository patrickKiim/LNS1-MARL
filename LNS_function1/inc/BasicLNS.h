#pragma once
#include "common.h"
#include "SIPP.h"

struct Agent
{
    int id;
    SingleAgentSolver* path_planner = nullptr; // start, goal, and heuristics are stored in the path planner
    Path path;

    Agent(const Instance& instance, int id) : id(id)
    {
        path_planner = new SIPP(instance, id);
    }
    ~Agent(){ delete path_planner; }

};

struct Neighbor
{
    vector<int> agents;
};
