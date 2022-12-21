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





n = 3
queries = [[5,3], [4,7], [2,3]]
# n = 2
# queries = [[1,2]]
queries = [[17,21],[23,5],[15,7],[3,21],[31,9],[5,15],[11,2],[19,7]]
def helper(a,b):
    if a == b:
        return 1
    elif a > b:
        return 1 + helper(a//2, b)
    else:
        return 1 + helper(a, b//2)
def cycleLengthQueries(n, queries):
    result = []
    for a, b in queries:
        result.append(
            helper(a,b)
        )
    return result

# print(cycleLengthQueries(n, queries))


"""
# plot a complete binary tree with 2^n -1 nodes
import matplotlib.pyplot as plt
import networkx as nx

def add_nodes_and_edges(G, root, level, max_level):
    if level > max_level:
        return
    
    left_child = 2 * root
    right_child = 2 * root + 1
    
    G.add_node(left_child)
    G.add_edge(root, left_child)
    add_nodes_and_edges(G, left_child, level+1, max_level)
    
    G.add_node(right_child)
    G.add_edge(root, right_child)
    add_nodes_and_edges(G, right_child, level+1, max_level)

def create_complete_binary_tree(n):
    G = nx.DiGraph()
    G.add_node(1)
    add_nodes_and_edges(G, 1, 1, n)
    return G

def plot_complete_binary_tree(G):
    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
    nx.draw(G, pos, with_labels=True, node_size=1000, width=2)
    plt.show()

G = create_complete_binary_tree(4)

plot_complete_binary_tree(G)


"""
n = 5
edges = [[1,2],[2,3],[3,4],[4,2],[1,4],[2,5]]

n = 4
edges = [[1,2], [3,4]]

n = 4
edges = [[1,2],[1,3],[1,4]]

n = 5
edges = [[4,3],[4,5],[5,3],[3,1],[5,2]]
from collections import defaultdict
from itertools import product
def isPossible(n, edges):
    """
    Zero is trivial.

    For 2 vertices, we check that both of them can be connected to a third one (or to each other) with 
    which neither of them share an edge.

    For 4 vertices, we can add only internal edges, becuase we are allowed to use not more than 2 of them. 
    We add edges based on the number of existing edges within the group of 4 odd vertices:
        0 - connect any combination of two pairs
        1 - add 2 more to make an X̲ / Π pattern
        2 - add 2 more to make a loop / hourglass pattern
        4 - add 2 more to make a loop + hourglass pattern
        3 or 5 edges won't solve the problem. 6 is the maximum possible number of edges between 4 vertices.
    """
    G = defaultdict(set)
    for i,j in edges:
        G[i].add(j)
        G[j].add(i)
    
    odd = [i for i in range(1, n+1) if len(G[i]) % 2]
    f = lambda a, b: a not in G[b]

    if len(odd) == 2:
        a, b = odd
        return any( f(a,i ) and f(b,i) for i in range(1, n+1) )
    
    if len(odd) == 4:
        a,b,c,d = odd
        return f(a,b) and f(c,d) or \
                f(a,c) and f(b,d) or \
                f(a,d) and f(c,b)  
    return len(odd) == 0

# print(isPossible(n, edges))

n = 15
# n = 3
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

def smallestValue(n):
    factors = getPrimeFactors(n)
    while sum(factors) != n:
        n = sum(factors)
        factors = getPrimeFactors(n)
    return n


print(getPrimeFactors(n))
