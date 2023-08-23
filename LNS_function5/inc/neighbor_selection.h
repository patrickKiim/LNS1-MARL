#pragma once
#include "BasicLNS.h"
#include "PathTable.h"


class Selection
{
public:
    vector<Agent> agents;
    int num_agents;
    int self;
    int load_agent;

    Selection(const Instance& instance,const string& pathfile,int self);

    vector<pair<int,int>> runPP();

private:
    vector<vector<pair<int,int>>> vector_paths;
    const Instance& instance;
    string pathfile;

    PathTableWC path_table; // 1. stores the paths of all agents in a time-space table;
    // 2. avoid making copies of this variable as much as possible.
    inline int linearizeCoordinate(int row, int col,int row_size) const { return (row_size * row + col); }
    void build_element();
    bool loadpath();
};
