#include "romanToInteger.h"

int SolutionRtoI::romanToInt(string s) {
    // Dict to save all possible Roman numbers
    unordered_map<char, int> dict;
    dict.insert({{'I', 1}, {'V', 5}, {'X', 10}, {'L', 50}, 
    {'C', 100}, {'D', 500}, {'M', 1000}});
    
    int res = 0;

    for (int i = 0; i < s.length() - 1; i++) {
        if (dict.find(s[i+1])->second > dict.find(s[i])->second) {
            res -= dict.find(s[i])->second;
        } else {
            res += dict.find(s[i])->second;
        }
    }

    res += dict.find(s[s.length() - 1])->second;

    return res;

}