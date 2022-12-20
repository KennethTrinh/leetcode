from itertools import accumulate

transpose = lambda arr: [list(row) for row in list(zip(*arr))]

pprint = lambda arr: [print(row) for row in arr]
isPowerofTwo = lambda x: (x & (x-1)) == 0
prefixSum = lambda arr: [0] + list(accumulate(arr))
countTotal = lambda P, x, y: P[y+1] - P[x]


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    def __repr__(self):
        arr = []
        self.form_list(arr, self)
        return str(arr)
    def form_list(self, arr, root):
        #preorder traverse: root, left, right
        if root != None:
            arr.append(root.val)
            self.form_list(arr, root.left)
            self.form_list(arr, root.right)
        # else:
        #     arr.append('null')
    @staticmethod
    def list_to_node(arr):
        def helper(arr, index):
            value = arr[index-1] if index <= len(arr) else None
            if value and value!='null':
                t = TreeNode(value)
                t.left = helper(arr, index*2)
                t.right = helper(arr, index*2 + 1)
                return t
            return None
        root = helper(arr, 1)
        return root



root =  TreeNode.list_to_node([4,2,7,1,3,6,9])
root = TreeNode.list_to_node([2,1,3])
root  = TreeNode.list_to_node([-10,9,20,'null','null',15,7])
root = TreeNode.list_to_node([-3])
root = TreeNode.list_to_node([1,2,3,'null','null',4,5])

root = TreeNode.list_to_node([3,4,5,1,2, 'null', 'null']) 
# root = TreeNode.list_to_node([3,4,5,1,2,'null','null','null','null',0])
subRoot = TreeNode.list_to_node([4,1,2])



class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return all([other.val == self.val, self.neighbors == other.neighbors])
        return False
    def __repr__(self):
        return f"val:{self.val}, neighbors:{[n.val for n in self.neighbors]}"
    def adj_list(self):
        adj = dict()
        def dfs(node):
            if node is None:
                return
            if node.val not in adj:
                adj[node.val] = set()
            for neighbor in node.neighbors:
                if neighbor.val not in adj[node.val]:
                    adj[node.val].add( neighbor.val)
                    dfs(neighbor)
        dfs(self)
        return adj




def buildAdjacencyList(n, edgesList):
        adjList = [[] for _ in range(n)]
        # c2 (course 2) is a prerequisite of c1 (course 1)
        # i.e c2c1 is a directed edge in the graph
        for c1, c2 in edgesList:
            adjList[c2].append(c1)
        return adjList



class Node:
    def __init__(self, val):
        self.val = val
        self.parent = self
        self.size = 1
    
class UnionFind:
    def find(self, node):
        if node.parent != node:
            node.parent = self.find(node.parent)
        return node.parent
    def union(self, node1, node2):
        parent_1 = self.find(node1)
        parent_2 = self.find(node2)
        if parent_1 != parent_2:
            parent_2.parent = parent_1
            parent_1.size += parent_2.size
        return parent_1.size

def sieve(n):
    primes = [True] * (n+1)
    p = 2
    while p * p <= n:
        if primes[p] == True:
            for i in range(p * p, n+1, p):
                primes[i] = False
        p += 1
    return primes

from collections import defaultdict

def _trie():
    return defaultdict(_trie)

class Trie:
    def __init__(self):
        self.trie = _trie()
    def insert(self, word):
        t = self.trie
        for c in word:
            t = t[c]
        t['end'] = True
    def search(self, word):
        t = self.trie
        n = len(word)
        for i, c in enumerate(word): 
            if i == n-1 and i != 0 and (c not in t) and any( 'end' in t[j] for j in t ):
                # the maximmum number of 'j' keys in t is 26, so the any is constant time
                return True
            if c not in t:
                return False
            t = t[c]
        return 'end' in t




def subsets(arr):
    result = [[]]
    for num in arr:
        result += [curr + [num] for curr in result]
    return result







"""
Allocator loc = new Allocator(10); // Initialize a memory array of size 10. All memory units are initially free.

loc.allocate(1, 1); // The leftmost block's first index is 0. The memory array becomes [1,_,_,_,_,_,_,_,_,_]. We return 0.
loc.allocate(1, 2); // The leftmost block's first index is 1. The memory array becomes [1,2,_,_,_,_,_,_,_,_]. We return 1.
loc.allocate(1, 3); // The leftmost block's first index is 2. The memory array becomes [1,2,3,_,_,_,_,_,_,_]. We return 2.
loc.free(2); // Free all memory units with mID 2. The memory array becomes [1,_, 3,_,_,_,_,_,_,_]. We return 1 since there is only 1 unit with mID 2.


"""

class Allocator:
    
        def __init__(self, n: int):
            self.n = n
            self.avail = {0: n}
            self.alloc = {}
            

        def allocate(self, size: int, mID: int):
            for i, i_sz in self.avail.items():
                if i_sz >= size:
                    self.alloc[mID] = self.alloc.get(mID, []) + [(i, size)]
                    self.avail.pop(i)
                    if i_sz > size:
                        self.avail[i + size] = i_sz - size
                    return i
            return -1
    
        def free(self, mID: int):
            if not mID in self.alloc:
                return 0
            count = 0
            for i, sz in self.alloc[mID]:
                self.avail[i] = sz
                if i in self.avail and i - 1 in self.avail:
                    self.avail[i - 1] += sz
                    self.avail.pop(i)
                    i = i - 1
                if i + sz in self.avail:
                    self.avail[i] += self.avail[i + sz]
                    self.avail.pop(i + sz)
                count += sz
            self.alloc.pop(mID)
            return count

        

loc = Allocator(10)
loc.allocate(1, 1)

loc.allocate(1, 2)

loc.allocate(1, 3)

loc.free(2)

print('avail: ', loc.avail)
print('alloc: ', loc.alloc)

print(loc.allocate(3, 4))
print(loc.allocate(1, 1))
print(loc.allocate(1, 1))
print(loc.free(1))
print(loc.allocate(10, 2))
print(loc.free(7))

# grid = [[1,2,3],[2,5,7],[3,5,1]]
# queries = [5,6,2]

# grid = [[5,2,1],[1,1,2]]
# queries = [3]
        

def maxPoints(grid, queries):
    class edge:
        def __init__(self, u, v, maxVal):
            self.u = u
            self.v = v
            self.maxVal = maxVal
        def __repr__(self):
            return f"({self.u}, {self.v}, {self.maxVal})"
    class DSU:
        def __init__(self, n):
            self.parent = [i for i in range(n)]
            self.size = [1 for i in range(n)]
        def find(self, x):
            if self.parent[x] == x:
                return x
            self.parent[x] = self.find(self.parent[x])
            return self.parent[x]
        def unions(self, x, y):
            x_rep = self.find(x)
            y_rep = self.find(y)
            if x_rep == y_rep:
                return
            self.parent[y_rep] = x_rep
            self.size[x_rep] += self.size[y_rep]
        def getSize(self, x):
            return self.size[self.find(x)]
    isWithinGrid =  lambda x, y, r, c: x >= 0 and y >= 0 and x < r and y < c
    def processEdges(grid):
        allEdges = []
        n = len(grid)
        m = len(grid[0])
        u , v = 0, 0
        for i in range(n):
            for j in range(m):
                u = i * m + j
                if isWithinGrid(i, j + 1, n, m):
                    v = i * m + j + 1
                    temp = edge(u, v, max(grid[i][j], grid[i][j + 1]))
                    allEdges.append(temp)
                if isWithinGrid(i + 1, j, n, m):
                    v = (i + 1) * m + j
                    temp = edge(u, v, max(grid[i][j], grid[i + 1][j]))
                    allEdges.append(temp)
        return allEdges
    allEdges = processEdges(grid)
    allEdges.sort(key = lambda x: x.maxVal)
    ufobj = DSU(len(grid) * len(grid[0]))
    q = sorted( [ (query, i) for i, query in enumerate(queries) ]) 
    ans = [0 for i in range(len(queries))]
    edPointer = 0
    for currQ in q:
        while edPointer < len(allEdges) and allEdges[edPointer].maxVal < currQ[0]:
            ufobj.unions(allEdges[edPointer].u, allEdges[edPointer].v)
            edPointer += 1
        if grid[0][0] < currQ[0]:
            ans[currQ[1]] = ufobj.getSize(0)
    return ans

# print( maxPoints(grid, queries) )