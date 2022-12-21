/*
 * Generate a testcase to stress test
 */

#include <bits/stdc++.h>

using namespace std;

mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());

// random value in range [l, r] with discrete uniform distribution   
template<typename T = int>
T rand(T l, T r) {
    return l + rng()%(r - l + 1);
}

// create a array which have number of elements is `n` and  have value in range [l, r] with discrete uniform distribution  

template<typename T = int>
void arr(int n, T l, T r, bool ascending = false) {
    vector<T> a(n);
    for(auto &v : a) v = rand(l, r);
    if(ascending) sort(a.begin(), a.end());
    for(auto &v : a) cout << v << " "; 
    cout << '\n';
}

// create a tree which have number of nodes is n  
// (optional) : with weight in range [l, r] 

template<typename T = int>
void tree(int n, bool weight = false, T l = 0, T r = 100) {
    vector<int> par(n), permutation(n); vector<pair<int, int>> edges;
    iota(permutation.begin(), permutation.end(), 0);
    shuffle(permutation.begin(), permutation.end(), rng);
    for(int i = 1; i < n; i++) {
        par[i] = rand(0, i - 1);
        edges.push_back({permutation[i], permutation[par[i]]});
    }
    shuffle(edges.begin(), edges.end(), rng);
    for(auto [u, v] : edges) {
        cout << u + 1 << " " << v + 1; 
        if(weight) {
            cout << " " << rand(l, r);
        }
        cout << '\n';
    }
}

// create a string which have character from `l` to `r` and this length is `n`
void str(int n, char l, char r) {
    for(int i = 0; i < n; i++) {
        cout << char(rand(l, r));
    }
    cout << '\n';
}

int32_t main() {
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    arr(30, 1, 100);
    arr(30, 1, 100);
    return 0;
}
