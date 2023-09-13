#include "longestCommonPrefix.h"

string SolutionLCP::longestCommonPrefix(vector<string>& strs) {
    string str="";
    string first = strs[0]; 
    
    for (int i = 0; i < first.length(); i++) {
        int j = 1;
        for (; j<strs.size(); j++){
            if (first[i] != strs[j][i]) {
                break;
            }
        }

        if (j == strs.size()) {
            str += first[i];
        } else
            break;
    }

    return str;
}