#include "candy.h"
    
int SolutionCND::candy(vector<int>& ratings) {
    int i = 1;
    int right = 1;
    int n = ratings.size();
    vector<int> left(n, 1);
    for (; i < n; i++) {
        if (ratings[i] > ratings[i-1]) {
            left[i] = left[i-1] + 1;
        }
    }

    i = n - 2;
    for (; i >= 0; i--) {
        if (ratings[i] > ratings[i + 1]) {
            right++;
            left[i] = max(left[i], right);
        } else {
            right = 1;
        }
    }

    i = 0;
    right = 0;
    for (; i < n; i++) {
        right += left[i]; 
    }

    return right;
}
