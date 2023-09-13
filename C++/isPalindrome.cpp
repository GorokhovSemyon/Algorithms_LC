#include "isPalindrome.h"

bool SolutionIsPalindrome::isPalindrome(int x) {
    std::string x_str = std::to_string(x);
    cout << x_str << std::endl;
    int left = 0;
    int right = x_str.length() - 1;
    while (left < right) {
        if (x_str[left] != x_str[right]) {
            return false;
        }
        left++;
        right--;
    }
    
    return true;
}