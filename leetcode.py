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


def getPrimeFactors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1 
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors

from collections import deque
n = 6
edges = [[1,2],[1,4],[1,5],[2,6],[2,3],[4,6]]

class UF:
    def __init__(self):
        self.uf = {}
    def find(self, x):
        if x not in self.uf:
            self.uf[x] = x
        if x != self.uf[x]:
            self.uf[x] = self.find(self.uf[x])
        return self.uf[x]
    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        self.uf[ rootX ] = rootY
def magnificentSets(n, edges):

        uf = UF()
        

        # check if graph has odd number edge cycles by checking if neighbor node has been visited before
        # and if the neighbor was on the same level as the current node when we were visiting the neighbor
        def BFS(node):
            q = deque([(node,1)])
            seen = {node:1}
            level = 1
            while q:
                cur,level = q.popleft()
                for nei in graph[cur]:
                    if nei not in seen:
                        seen[nei] = level+1
                        q.append((nei,level+1))
                    ### check if there is a odd number edges cycle
                    elif seen[nei]==level: 
                        return -1
            return level
        
        graph = defaultdict(list)
        for s,e in edges:
            graph[s].append(e)
            graph[e].append(s)
            uf.union(s,e)
        
        ### Store the largest group in each connected component
        maxGroup = defaultdict(int)
        ### Start a BFS on each node, and update the maxGroup for each connected component
        for i in range(1,n+1):
            groups = BFS(i)
            ### There is a odd number edges cycle, so return -1
            if groups==-1:
                return -1
            ### Find the root of the current connected component
            root = uf.find(i)
            ### Update it.
            maxGroup[root] = max(maxGroup[root],groups)

        return sum(maxGroup.values())
    

# print(magnificentSets(n, edges))

n = 4
roads = [[1,2,9],[2,3,6],[2,4,5],[1,4,7]]

n = 4
roads = [[1,2,2],[1,3,4],[3,4,7]]
def solution(n, roads):
    adjList = {}
    for v1,v2,w in roads:
        if v1 not in adjList:
            adjList[v1] = []
        if v2 not in adjList:
            adjList[v2] = []
        adjList[v1].append((v2,w))
        adjList[v2].append((v1,w))

    def BFS(start):
        q = deque([start])
        visited = set()
        minimum = float('inf')
        while q:
            node = q.popleft()
            visited.add(node)
            print(q)
            for nei, w in adjList[node]:
                minimum = min(minimum, w)
                if nei not in visited:
                    q.append(nei)
        return minimum

skill = [3,2,5,1,3,4]
skill = [3,4]
skill = [1,1,2,3]
def dividePlayers(skill):
    skill.sort()
    n = len(skill)
    k = skill[0] + skill[-1]
    res = skill[0] * skill[-1]
    for i in range(1, n//2):
        if skill[i] + skill[n-i-1] != k:
            return -1
        res += skill[i] * skill[n-i-1]
    return res
        

print(
    dividePlayers(skill)
)




