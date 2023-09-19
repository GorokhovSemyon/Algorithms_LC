#ifndef LC_PATH_WITH_MIN_EFFORT_H_
#define LC_PATH_WITH_MIN_EFFORT_H_

#include <unordered_map>
#include <vector>
#include <string>
#include <stack>
#include <queue>
#include <algorithm>
#include <iostream>
#include <limits.h>

using std::unordered_map;
using std::vector;
using std::string;
using std::priority_queue;
using std::stack;

class SolutionME {
    public:
        int dijkstra(vector<vector<int>>& heights);
        int minimumEffortPath(vector<vector<int>>& heights);
    private:
        int effort[100][100];    // Сохранм effort для каждой ячейки
        int dx[4] = {0, 1, -1, 0};  // Все возможные изменения координаты x
        int dy[4] = {1, 0, 0, -1};  // Все возможные изменения координаты y (в соответствии с x)

};

#endif // LC_PATH_WITH_MIN_EFFORT_H_