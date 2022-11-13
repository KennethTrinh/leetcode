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



A = 6
B = 20
# A = 21
# B = 29

def getFactors(n):
    sqrt = int(n**0.5)
    arr = []
    for i in range(sqrt, 0, -1):
        if n%i==0:
            m = n//i #m bigger, i smaller
            if m!=i:
                arr.insert(0,i) #insert from front
            arr.append(m) #insert to back
    return arr

def checkConsecutiveFactorsMutiplyToN(n):
    factors = getFactors(n)
    for i in range(1, len(factors)):
        if factors[i] * factors[i-1] == n and factors[i] - factors[i-1] == 1:
            return True
    return False


def solution(A, B):
    result = 0
    for i in range(A, B+1):
        if checkConsecutiveFactorsMutiplyToN(i):
            result += 1
    return result

# print(solution(A, B))



# # check if array contains all numbers from 1 to K
# def solution(A, K):
#     if len(A) < K:
#         return False
#     if len(A) == K:
#         return 1 if len(set(A)) == K else 0
#     m = { i:False for i in range(1, K+1) }
#     for i in range(len(A)):
#         m[A[i]] = True
#     return all (m.values())

#A is sorted in non-decreasing order

def solution(A, K):
    n = len(A)
    for i in range(n - 1):
        if A[i] + 1  < A[i+1]:
            return False
    if A[0] != 1 and A[n-1] < K:
        return False
    return True



print(solution([1,1,2,3,3], 3))
print(solution([1,1,3], 2))
print(solution([1,1,2, 3, 4], 4))


# print(solution([1,1,2, 3, 4, 5], 4))





class Tree():
    def __init__(self, x):
        self.val = x
        self.l = None
        self.r = None


# T = Tree(1)
# T.l = Tree(2)
# T.r = Tree(2)
# T.l.l = Tree(1)
# T.l.r = Tree(2)
# T.r.l = Tree(4)
# T.r.r = Tree(1)

T = Tree(1)
T.r = Tree(2)
T.r.r = Tree(1)
T.r.l = Tree(1)
T.r.r.l = Tree(4)



def solution(T):
    def traverse(root, mSet):
        if root is None or root.val in mSet:
            return len(mSet)
        mSet.add(root.val)
        l = traverse(root.l, mSet)
        r = traverse(root.r, mSet)
        mSet.remove(root.val)
        return max(l, r)
    return traverse(T, set())

# print(solution(T))



