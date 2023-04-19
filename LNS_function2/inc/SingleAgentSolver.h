#pragma once
#include "Instance.h"


class SingleAgentSolver
{
public:
	int start_location;
	int goal_location;
	vector<int> my_heuristic;  // this is the precomputed heuristic for this agent
	const Instance& instance;

    void findMinimumSetofColldingTargets(vector<int>& goal_table,set<int>& A_target);  // used for neighbor selection

	SingleAgentSolver(const Instance& instance, int agent) :
		instance(instance), //agent(agent), 
		start_location(instance.start_locations[agent]),
		goal_location(instance.goal_locations[agent]) // initial give value
	{
		compute_heuristics();
	}
	virtual ~SingleAgentSolver()= default;

protected:
	void compute_heuristics();
};

