#include "pathWithMinEffort.h"

// Алгоритм Дейкстры
int SolutionME::dijkstra(vector<vector<int>>& heights) {
    int rows = heights.size();
    int cols = heights[0].size();

    // Очередь с приоритетом для {effort, {x, y}}
    std::priority_queue<std::pair<int, std::pair<int, int>>> pq;
    pq.push({0, {0, 0}});  // Начало в верхнем левом углу
    effort[0][0] = 0;  // Инициализируем effort в начале

    while (!pq.empty()) {
        auto current = pq.top().second;
        int cost = -pq.top().first;  // Effort для текущей ячейки
        pq.pop();

        int x = current.first;
        int y = current.second;

        // Пропускаем, если уже есть более короткий путь
        if (cost > effort[x][y])
            continue;

        // Останавливаемся в правом нижнем углу
        if (x == rows - 1 && y == cols - 1)
            return cost;

        // Проверяем все возможные пути из этой ячейки (up, down, left, right)
        for (int i = 0; i < 4; i++) {
            int new_x = x + dx[i];
            int new_y = y + dy[i];

            // Проверка на выход за границы поля
            if (new_x < 0 || new_x >= rows || new_y < 0 || new_y >= cols)
                continue;

            // Вычисление нового effort для ячейки-соседа
            int new_effort = std::max(effort[x][y], std::abs(heights[x][y] - heights[new_x][new_y]));

            // Обновляем effort если меньший effort был найден в ячейке-соседе
            if (new_effort < effort[new_x][new_y]) {
                effort[new_x][new_y] = new_effort;
                pq.push({-new_effort, {new_x, new_y}});
            }
        }
    }
    return effort[rows - 1][cols - 1];  // Минимальный effort 
}

int SolutionME::minimumEffortPath(vector<vector<int>>& heights) {
    // Искусственно выставляем максимально возможное значение в каждую ячейку
    for (int i = 0; i < heights.size(); i++) {
        for (int j = 0; j < heights[i].size(); j++) {
            effort[i][j] = INT_MAX;
        }
    }
    return dijkstra(heights);
}