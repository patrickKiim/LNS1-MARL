#include "neighbor_selection.h"
#include <queue>
#include <algorithm>
#include<boost/tokenizer.hpp>


Selection::Selection(const Instance& instance,const string& pathfile ,bool ALNS ,
int neighbor_size,int old_num_collision,const string & init_destory_name,
bool update,int old_selected_neighbor) :
instance(instance),ALNS(ALNS),neighbor_size(neighbor_size),
old_num_collision_pairs(old_num_collision),goal_table(instance.map_size, -1),\
update(update),num_agents(instance.num_of_agents),collision_graph(num_agents),
path_table(instance.map_size, num_agents),old_selected_neighbor(old_selected_neighbor),pathfile(pathfile)
{
    bool succ =loadpath();
    if (!succ)
    {
        cerr << "loading path failed"  << endl;
        exit(-1);
    }
    if (ALNS)
    {
        decay_factor = 0.05;
        reaction_factor = 0.05;
    }
    else if (init_destory_name == "Target")
        init_destroy_strategy = TARGET_BASED;
    else if (init_destory_name == "Collision")
        init_destroy_strategy = COLLISION_BASED;
    else if (init_destory_name == "Random")
        init_destroy_strategy = RANDOM_BASED;
    else
    {
        cerr << "Init Destroy heuristic " << init_destory_name << " does not exists. " << endl;
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
    destroy_weights.resize(3);
    beg = tok.begin();  // content is height
    destroy_weights[0]=stod((*beg).c_str());
    beg++;
    destroy_weights[1]=stod((*beg).c_str());
    beg++;
    destroy_weights[2]=stod((*beg).c_str());
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
    set<pair<int, int>> colliding_pairs;

    for(int id=0;id<num_agents;id++)
    {
        agents.emplace_back(instance, id);
        goal_table[instance.goal_locations[id]] = id;
        agents[id].path.resize(vector_paths[id].size());
        for (int t=0;t<(int)vector_paths[id].size();t++)
        {
            agents[id].path[t].location=linearizeCoordinate(vector_paths[id][t].first, vector_paths[id][t].second, instance.num_of_rows);
        }
        updateCollidingPairs(colliding_pairs, id, agents[id].path);
        path_table.insertPath(id, agents[id].path);
        curr_sum_of_costs += (int)agents[id].path.size() - 1;
    }
    curr_num_of_colliding_pairs = colliding_pairs.size();
    for(const auto& agent_pair : colliding_pairs)
    {
        collision_graph[agent_pair.first].emplace(agent_pair.second);
        collision_graph[agent_pair.second].emplace(agent_pair.first);
    }
}

void Selection::update_weight()
{
    if (old_num_collision_pairs==0) // no need to replan
    {
        if (ALNS) // update destroy heuristics
        {
            destroy_weights[old_selected_neighbor] = (1 - decay_factor) * destroy_weights[old_selected_neighbor];
            cerr << "already no collision " << endl;
        }
        return;
    }
    if (ALNS) // update destroy heuristics
    {
        if (curr_num_of_colliding_pairs < old_num_collision_pairs)
            destroy_weights[old_selected_neighbor] =
                    reaction_factor * (double)(old_num_collision_pairs -
                            curr_num_of_colliding_pairs) // / neighbor.agents.size()
                    + (1 - reaction_factor) * destroy_weights[old_selected_neighbor];// success
        else
            destroy_weights[old_selected_neighbor] =
                    (1 - decay_factor) * destroy_weights[old_selected_neighbor]; // failure
    }
}

bool Selection::run()
{
    build_element();
    if (update)
        {update_weight();}
    vector<Path*> paths(num_agents);  // all agents paths
    for (auto i = 0; i < num_agents; i++)
        paths[i] = &agents[i].path;
    if (instance.validateSolution(paths, curr_sum_of_costs, curr_num_of_colliding_pairs))
        return true;
    while (true)
    {
        if (ALNS)
            chooseDestroyHeuristicbyALNS();

        switch (init_destroy_strategy)
        {
            case TARGET_BASED:
                succ = generateNeighborByTarget();
                break;
            case COLLISION_BASED:
                succ = generateNeighborByCollisionGraph();
                break;
            case RANDOM_BASED:
                succ = generateNeighborRandomly();
                break;
            default:
                cerr << "Wrong neighbor generation strategy" << endl;
                exit(-1);
        }
        if(succ && neighbor.agents.size()==neighbor_size)
            break;
    }
    for (int i = 0; i < (int)neighbor.agents.size(); i++)
    {
        path_table.deletePath(neighbor.agents[i]);
    }
    return false;
}

vector<vector<pair<int,int>>> Selection::runPP()
{
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
    return temp;
}


void Selection::clear()
{
    path_table.clear();
    collision_graph.clear();
    goal_table.clear();
}

// return true if the new path has collisions;
void Selection::updateCollidingPairs(set<pair<int, int>>& colliding_pairs, int agent_id, const Path& path) const
{
    if (path.size() < 2)
        return;
    for (int t = 1; t < (int)path.size(); t++)
    {
        int from = path[t - 1].location;
        int to = path[t].location;
        if ((int)path_table.table[to].size() > t) // vertex conflicts
        {
            for (auto id : path_table.table[to][t])
            {
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
            colliding_pairs.emplace(min(agent_id, id), max(agent_id, id));
        }
    }
}

void Selection::chooseDestroyHeuristicbyALNS()
{
    rouletteWheel();
    switch (selected_neighbor)
    {
        case 0 : init_destroy_strategy = TARGET_BASED; break;
        case 1 : init_destroy_strategy = COLLISION_BASED; break;
        case 2 : init_destroy_strategy = RANDOM_BASED; break;
        default : cerr << "ERROR" << endl; exit(-1);
    }
}

bool Selection::generateNeighborByCollisionGraph()
{

    vector<int> all_vertices;  //all collide agent
    all_vertices.reserve(collision_graph.size());   // number of all agents
    for (int i = 0; i < (int)collision_graph.size(); i++)
    {
        if (!collision_graph[i].empty())  // except did not collide with other agents
            all_vertices.push_back(i);
    }
    boost::unordered_map<int, set<int>> G;
    auto v = all_vertices[rand() % all_vertices.size()]; // pick a random vertex
    findConnectedComponent(collision_graph, v, G);
    assert(G.size() > 1);

    assert(neighbor_size <= (int)agents.size());
    set<int> neighbors_set;
    if ((int)G.size() <= neighbor_size)
    {
        for (const auto& node : G)
            neighbors_set.insert(node.first);
        int count = 0;
        while ((int)neighbors_set.size() < neighbor_size && count < 10)  // chosen agents size smaller than requirement and tried time<10
        {
            int a1 = *std::next(neighbors_set.begin(), rand() % neighbors_set.size());// next return the nth sucessor of neighbors_set.begin()
            int a2 = randomWalk(a1);
            if (a2 != NO_AGENT)
                neighbors_set.insert(a2);
            else
                count++;
        }
    }
    else
    {
        int a = std::next(G.begin(), rand() % G.size())->first; //G:[key(int),value(set)]->[agent id, agent set that collide with agent id]
        neighbors_set.insert(a);
        while ((int)neighbors_set.size() < neighbor_size)
        {
            a = *std::next(G[a].begin(), rand() % G[a].size());
            neighbors_set.insert(a);
        }
    }
    neighbor.agents.assign(neighbors_set.begin(), neighbors_set.end());  //assign element from beginning to end
    return true;

}
bool Selection::generateNeighborByTarget()
{
    int a = -1;
    auto r = rand() % (curr_num_of_colliding_pairs * 2);
    int sum = 0;
    for (int i = 0 ; i < (int)collision_graph.size(); i++)
    {
        sum += (int)collision_graph[i].size();
        if (r <= sum and !collision_graph[i].empty())
        {
            a = i;
            break;
        }
    }//according collision degree to chose agent
    assert(a != -1 and !collision_graph[a].empty());
    set<pair<int,int>> A_start; // an ordered set of (time, id) pair.  early step
    set<int> A_target;  // goal block


    for(int t = 0 ;t< path_table.table[agents[a].path_planner->start_location].size();t++){
        for(auto id : path_table.table[agents[a].path_planner->start_location][t]){
            if (id!=a)
                A_start.insert(make_pair(t,id));  // agent id visit agent a' start position at time t
        }
    }

    agents[a].path_planner->findMinimumSetofColldingTargets(goal_table,A_target);// generate non-wait path and collect A_target, agent a's A* path has visit the goal of agent j(add to A_t)

    set<int> neighbors_set;

    neighbors_set.insert(a);

    if(A_start.size() + A_target.size() >= neighbor_size-1){
        if (A_start.empty()){
            vector<int> shuffled_agents;
            shuffled_agents.assign(A_target.begin(),A_target.end());
            std::random_shuffle(shuffled_agents.begin(), shuffled_agents.end());
            neighbors_set.insert(shuffled_agents.begin(), shuffled_agents.begin() + neighbor_size-1);
        }  // case(a)  n-1 A_t
        else if (A_target.size() >= neighbor_size){
            vector<int> shuffled_agents;
            shuffled_agents.assign(A_target.begin(),A_target.end());
            std::random_shuffle(shuffled_agents.begin(), shuffled_agents.end());
            neighbors_set.insert(shuffled_agents.begin(), shuffled_agents.begin() + neighbor_size-2);

            neighbors_set.insert(A_start.begin()->second);
        }  //case(b) n-2 A_t+1 A_s
        else{
            neighbors_set.insert(A_target.begin(), A_target.end());
            for(auto e : A_start){
                //A_start is ordered by time.
                if (neighbors_set.size()>= neighbor_size)
                    break;
                neighbors_set.insert(e.second);

            }  // case(c) all A-t+ rest agents come from A_s
        }
    }
    else if (!A_start.empty() || !A_target.empty()){
        neighbors_set.insert(A_target.begin(), A_target.end());
        for(auto e : A_start){
            neighbors_set.insert(e.second);
        }  // case 2. add all agents in A_s and A_t

        set<int> tabu_set; // add additional agentss whose goal are visitied by the agents that already in neighbors(a)
        while(neighbors_set.size()<neighbor_size){
            int rand_int = rand() % neighbors_set.size();
            auto it = neighbors_set.begin();
            std::advance(it, rand_int);  //advance:
            a = *it;
            tabu_set.insert(a);

            if(tabu_set.size() == neighbors_set.size())
                break;

            vector<int> targets;
            for(auto p: agents[a].path){
                if(goal_table[p.location]>-1){
                    targets.push_back(goal_table[p.location]);
                }  // goal table: location-> agent id
            }

            if(targets.empty())
                continue;
            rand_int = rand() %targets.size();
            neighbors_set.insert(*(targets.begin()+rand_int));
        }
    }

    neighbor.agents.assign(neighbors_set.begin(), neighbors_set.end());
    return true;
}
bool Selection::generateNeighborRandomly()  // select neighbor by probability deg(collision)+1
{
    if (neighbor_size >= agents.size())
    {
        neighbor.agents.resize(agents.size());
        for (int i = 0; i < (int)agents.size(); i++)
            neighbor.agents[i] = i;  //add all agents
        return true;
    }
    set<int> neighbors_set;
    auto total = curr_num_of_colliding_pairs * 2 + agents.size();
    while(neighbors_set.size() < neighbor_size)
    {
        vector<int> r(neighbor_size - neighbors_set.size());
        for (auto i = 0; i < neighbor_size - neighbors_set.size(); i++)
            r[i] = rand() % total;
        std::sort(r.begin(), r.end());
        int sum = 0;
        for (int i = 0, j = 0; i < agents.size() and j < r.size(); i++)
        {
            sum += (int)collision_graph[i].size() + 1;  // deg(collision)+1
            if (sum >= r[j])
            {
                neighbors_set.insert(i);
                while (j < r.size() and sum >= r[j])
                    j++;
            }
        }
    }
    neighbor.agents.assign(neighbors_set.begin(), neighbors_set.end());
    return true;
}

// Random walk; return the first agent that the agent collides with
int Selection::randomWalk(int agent_id)
{
    int t = rand() % agents[agent_id].path.size();
    int loc = agents[agent_id].path[t].location;  // start from a random position at the agent's path
    while (t <= path_table.makespan and
           (path_table.table[loc].size() <= t or
           path_table.table[loc][t].empty() or
           (path_table.table[loc][t].size() == 1 and path_table.table[loc][t].front() == agent_id)))  // no one at this position except the agent self
    {
        auto next_locs = instance.getNeighbors(loc);
        next_locs.push_back(loc);
        int step = rand() % next_locs.size();
        auto it = next_locs.begin();
        loc = *std::next(next_locs.begin(), rand() % next_locs.size()); // move to next connected random position
        t = t + 1;
    }
    if (t > path_table.makespan)
        return NO_AGENT;
    else
        return *std::next(path_table.table[loc][t].begin(), rand() % path_table.table[loc][t].size());  // find collision agents at this step, choosen a random agent from them
}

boost::unordered_map<int, set<int>>& Selection::findConnectedComponent(const vector<set<int>>& graph, int vertex,
                                                                       boost::unordered_map<int, set<int>>& sub_graph)
{
    std::queue<int> Q;
    Q.push(vertex);
    sub_graph.emplace(vertex, graph[vertex]);  // insert value:graph[vertex] at key:vertex
    while (!Q.empty())
    {
        auto v = Q.front(); Q.pop();
        for (const auto & u : graph[v])
        {
            auto ret = sub_graph.emplace(u, graph[u]);
            if (ret.second) // insert successfully
                Q.push(u);
        }
    }
    return sub_graph;
}


void Selection::rouletteWheel()
{
    double sum = 0;
    for (const auto& h : destroy_weights)
        sum += h;

    double r = (double) rand() / RAND_MAX;// probability
    double threshold = destroy_weights[0];
    selected_neighbor = 0;
    while (threshold < r * sum)
    {
        selected_neighbor++;
        threshold += destroy_weights[selected_neighbor];
    }
}
