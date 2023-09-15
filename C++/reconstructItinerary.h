#ifndef LC_RECONSTRUCT_ITINERARY_H_
#define LC_RECONSTRUCT_ITINERARY_H_

#include <unordered_map>
#include <vector>
#include <string>
#include <stack>
#include <queue>
#include <algorithm>
#include <iostream>

using std::unordered_map;
using std::vector;
using std::string;
using std::priority_queue;
using std::stack;

class SolutionRI {
    public:
        vector<string> findItinerary(vector<vector<string>>& tickets);
};

#endif // LC_RECONSTRUCT_ITINERARY_H_