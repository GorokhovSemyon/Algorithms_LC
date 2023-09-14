// To use different files, change tesks.json

// #include "Two_Sum.h"
// #include "isPalindrome.h"
// #include "Roman_to_integer.h"
// #include "longestCommonPrefix.h"
// #include "validParentheses.h"
#include "candy.h"

int main() {
    // For Two_Sum

    // vector<int> inp = {3,2,4};
    // vector<int> result;
    // SolutionTwoSum res;
    // result = res.twoSum(inp, 6);
    // res.out_vector(result);
   
    // For isPalindrome
   
    // SolutionIsPalindrome res;
    // int input;
    // cin >> input;
    // if(res.isPalindrome(input)){
    //     cout << "Yes";
    // } else {
    //     cout << "No";
    // }

     // For Roman_to_integer

    // SolutionRtoI tmp;
    // string input;
    // cin >> input;
    // int res = tmp.romanToInt(input);
    // cout << res << std::endl;

     // For Roman_to_integer

    // SolutionLCP tmp;
    // vector<string> strs = {"flower", "flow", "flight"};
    // cout << tmp.longestCommonPrefix(strs);

    // For validParentheses

    // SolutionVP res;
    // string input;
    // cin >> input;
    // if(res.isValid(input)){
    //     cout << "Yes\n";
    // } else {
    //     cout << "No\n";
    // }

    // For candy

    SolutionCND res;
    int n;
    std::cout << "Введите длину вектора: ";
    std::cin >> n;
    int* elements = new int[n];
    std::cout << "Введите элементы вектора через пробел: ";
    for (int i = 0; i < n; ++i) {
        std::cin >> elements[i];
    }
    std::vector<int> inp(elements, elements + n);
    delete[] elements;
    cout << res.candy(inp) << std::endl;
}