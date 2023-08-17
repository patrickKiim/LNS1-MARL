#pragma once
#include "common.h"

struct Agent
{
    int id;
    Path path;
    Agent(int id) : id(id){}
};
