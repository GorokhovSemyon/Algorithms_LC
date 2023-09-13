#ifndef LC_TWO_SUM_H_
#define LC_TWO_SUM_H_

#include <vector>
#include <unordered_map>
#include <iostream>

using namespace std;

class SolutionTwoSum {
public:
    vector<int> twoSum(vector<int>& nums, int target);
    void out_vector(vector<int> result);
};

#endif // LC_TWO_SUM_H_