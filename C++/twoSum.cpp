#include "twoSum.h"

vector<int> SolutionTwoSum::twoSum(vector<int>& nums, int target) {
    unordered_map<int, int> ind_map;

    for (int i = 0; i < nums.size(); i++) {
        int n = nums.at(i);
        bool is_in = ind_map.find(target - n) != ind_map.end();
        if (is_in) {
            return {ind_map[target - n], i};
        }
        ind_map.insert({n, i});
    }

    throw invalid_argument("Solution doesn't exist");
}

void SolutionTwoSum::out_vector(vector<int> result) {
    bool is_first = true;
    for (auto el : result) 
    {
        if (is_first)
        {
            std::cout << el;
            is_first = false;
        }
        else std::cout << ", " << el;
    }
    std::cout << std::endl;
}