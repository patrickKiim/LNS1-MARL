#pragma once
#include "common.h"

#define NO_AGENT -1

class PathTableWC // with collisions
{
public:
    int makespan = 0;
    vector< vector< list<int> > > table; // this stores the paths, the value is the id of the agent
    vector<int> goals; // this stores the goal locatons of the paths: key is the location, while value is the timestep when the agent reaches the goal
    void insertPath(int agent_id, const Path& path);
    explicit PathTableWC(int map_size = 0, int num_of_agents = 0) : table(map_size), goals(map_size, MAX_COST),
        paths(num_of_agents, nullptr) {}
private:
    vector<const Path*> paths;
};