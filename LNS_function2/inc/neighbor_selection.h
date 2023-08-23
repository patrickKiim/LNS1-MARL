#pragma once
#include "BasicLNS.h"
#include "PathTable.h"

enum init_destroy_heuristic { TARGET_BASED, COLLISION_BASED, RANDOM_BASED, INIT_COUNT };

class Selection
{
public:
    vector<Agent> agents;
    int curr_num_of_colliding_pairs = 0;  //calculate at this step
    int curr_sum_of_costs=0;
    const int old_num_collision_pairs;
    vector<double> destroy_weights;
    Neighbor neighbor;
    int num_agents;
    int selected_neighbor=0;
    vector<int> num_collision;

    Selection(const Instance& instance,const string& pathfile ,bool ALNS ,
                         int neighbor_size,int old_num_collision,const string & init_destory_name,
                         bool update,int old_selected_neighbor);

    bool run();
    int all_path_makespan() const {return path_table.makespan;}
    void clear();
    vector<vector<pair<int,int>>> runPP();

private:
    const int neighbor_size;
    vector<vector<pair<int,int>>> vector_paths;
    const bool ALNS = false;
    const Instance& instance;
    bool succ= false;
    const bool update;
    string pathfile;

    double decay_factor = -1;
    double reaction_factor = -1;
    int old_selected_neighbor;
    init_destroy_heuristic init_destroy_strategy = COLLISION_BASED;

    PathTableWC path_table; // 1. stores the paths of all agents in a time-space table;
    // 2. avoid making copies of this variable as much as possible.

    vector<set<int>> collision_graph;
    vector<int> goal_table;  // location-> agent id
    inline int linearizeCoordinate(int row, int col,int row_size) const { return (row_size * row + col); }
    void updateCollidingPairs(set<pair<int, int>>& colliding_pairs, int agent_id, const Path& path) const;
    void calculateColliding(int agent_id, int local_agent_id,const Path& path);
    void build_element();
    void update_weight();
    void chooseDestroyHeuristicbyALNS();

    bool generateNeighborByCollisionGraph();
    bool generateNeighborByTarget();
    bool generateNeighborRandomly();
    bool loadpath();

    // int findRandomAgent() const;
    int randomWalk(int agent_id);

    static boost::unordered_map<int, set<int>>& findConnectedComponent(const vector<set<int>>& graph, int vertex,
                                                                       boost::unordered_map<int, set<int>>& sub_graph);

    void rouletteWheel();
};
