#include "validParentheses.h"

bool SolutionVP::isValid(string s) {
    // unordered_map<char, int> freq = {{'{', 0}, {'}', 0}, 
    // {'[', 0}, {']', 0}, {'(', 0}, {')', 0}};

    // for (char c : s) {
    //     freq[c]++;
    // }

    // cout << freq['('];
    // cout << freq[')'];
    // cout << freq['{'];
    // cout << freq['}'];
    // cout << freq['['];
    // cout << freq[']'];
    // cout << '\n';

    // if (freq['('] == freq[')'] 
    // && freq['{'] == freq['}']
    // && freq['['] == freq[']']) {
    //     return true;
    // } else 
    //     return false;

    // if(s.length() <= 1) {
    //         return false;
    //     }

    
    // Stack + unordered_map
    // unordered_map<char, char> oc;
    // stack<char> chrStck;
    // oc.insert({'}', '{'});
    // oc.insert({')', '('});
    // oc.insert({']', '['});

    // if (oc.count(s[0])) {
    //     return false;
    // }

    // for (char c : s) {
    //     if (!oc.count(c)) {
    //         chrStck.push(c);
    //     } else if (!chrStck.empty() && chrStck.top() == oc[c]) {
    //         chrStck.pop();
    //     } else {
    //         return false;
    //     }
    // }

    // return chrStck.empty();

    // Stack only
    stack<char> st;
        int n=s.size();

        if(n==1){
            return false;
        }

        for (int i=0; i < n; i++){

            if (st.empty() && s[i]==')'|| st.empty() && s[i]=='}' || st.empty() && s[i]==']'){
                return false;
            }

            if (s[i]=='{' || s[i]=='(' || s[i]=='['){
                st.push(s[i]);
            } else {
                char t = st.top();
                if ((s[i]=='}' && t=='{' && (!st.empty()))
                || (s[i]==')' && t=='(' && (!st.empty()))
                || (s[i]==']' && t=='[' && (!st.empty()))) {
                    st.pop();
                } else {
                    return false;
                }
            }
        } 
        
        return st.empty();
}