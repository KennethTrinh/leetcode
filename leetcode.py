from itertools import accumulate

transpose = lambda arr: [list(row) for row in list(zip(*arr))]

pprint = lambda arr: [print(row) for row in arr]
isPowerofTwo = lambda x: (x & (x-1)) == 0
prefixSum = lambda arr: [0] + list(accumulate(arr))
countTotal = lambda P, x, y: P[y+1] - P[x]
MOD = 1_000_000_007

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
    def levelOrder(self, root):
        if not root:
            return []
        res = []
        queue = [root]
        while queue:
            res.append([node.val for node in queue])
            queue = [child for node in queue for child in (node.left, node.right) if child]
        return res



# root =  TreeNode.list_to_node([4,2,7,1,3,6,9])
# root = TreeNode.list_to_node([2,1,3])
# root  = TreeNode.list_to_node([-10,9,20,'null','null',15,7])
# root = TreeNode.list_to_node([-3])
# root = TreeNode.list_to_node([1,2,3,'null','null',4,5])

root = TreeNode.list_to_node([3,4,5,1,2, 'null', 'null']) 
# root = TreeNode.list_to_node([3,4,5,1,2,'null','null','null','null',0])
# subRoot = TreeNode.list_to_node([4,1,2])



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


def subarrays(arr):
    """
    A subarray is a contiguous part of array and maintains relative ordering of elements. 
    For an array/string of size n, there are n*(n+1)/2 non-empty subarrays/substrings.
    """
    return [arr[i:j] for i in range(len(arr)) 
            for j in range(i+1, len(arr)+1)]




def subsets(arr):
    """
    A subset MAY NOT maintain relative ordering of elements and can or cannot be a contiguous part of an array. 
    For a set of size n, we can have (2^n) sub-sets in total.  Same as subsequences but with the empty set included.
    """
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

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
    @staticmethod
    def generateList(arr):
        head = ListNode(arr[0])
        ptr = head
        for i in range(1, len(arr)):
            ptr.next = ListNode(arr[i])
            ptr = ptr.next
        return head
    def printList(self):
        ptr = self
        while ptr:
            print(ptr.val, end=" ")
            ptr = ptr.next
        print()



primes = set('2357')
s = "23542185131"
k = 3
minLength = 2

def partition(s: str, k: int):
    """
    all possible non-overlapping partitions of k groups:
    """
    if k == 1:# If there is only one group, the only partition is the entire string
        return [(s,)]
    if not s:# If the string is empty, there are no partitions
        return []

    partitions = []
    # Consider all possible partitions of the string into the first group and the remaining groups
    for i in range(1, len(s)):
        for p in partition(s[i:], k - 1): # Partition the remaining groups recursively
            partitions.append((s[:i],) + p)
    return partitions

# print(
#     partition(s, k)
# )


s = "23542185131"
k = 3 # number of substrings
minLength = 2
isPrime = lambda x: x in {'2', '3', '5', '7'}
def beautifulPartitions(s, k, minLength):
    prime = "2357"
    dp = [[0]*(len(s)+1) for _ in range(k)]
    # dp[i][j] = number of ways to partition s[j:] into i groups
    if s[0] in prime and s[-1] not in prime: 
        for j in range(len(s)+1): 
            dp[0][j] = 1 # for 
        for i in range(1, k): 
            for j in range(len(s)-1, -1, -1): 
                dp[i][j] = dp[i][j+1]
                if minLength <= j <= len(s)-minLength and s[j-1] not in prime and s[j] in prime: 
                    dp[i][j] = (dp[i][j] + dp[i-1][j+minLength]) % 1_000_000_007
                    if dp[i-1][j + minLength] > 0:
                        print( s[j+minLength:], s[j:] , '-->', i+1 , j)
    print(dp[-1][0])


    import pandas as pd
    df = pd.DataFrame(dp, index = [f'{i+1} group' for i in range(k)], columns = [' '] + list(s))
    print(df)

beautifulPartitions(s, k, minLength)

roads = [[3,1],[3,2],[1,0],[0,4],[0,5],[4,6]] 
seats = 2




f = lambda n, s: n if n <= s else f(n-s,s) + n



roads = [[0,1],[0,2],[0,3]]
seats = 5

seats = 2
roads = [[0,2], [2,1], [2,3], [3,4], [3,5], [0,6]]
from collections import defaultdict
from math import ceil
def minimumFuelCost(roads, seats):
    graph = defaultdict(list)
    for u, v in roads:
        graph[u].append(v)
        graph[v].append(u)

    result = 0
    def dfs(node, parent):
        nonlocal result
        cnt = 1
        for neighbor in graph[node]:
            if neighbor != parent:
                cnt += dfs(neighbor, node)
        print(node, cnt)
        if node != 0:
            result += ceil(cnt / seats)
        return cnt
    dfs(0, -1)
    return result

# print(minimumFuelCost(roads, seats))

root = [6,2,13,1,4,9,15,'null','null','null','null','null','null',14]
queries = [2,5,16]
root  = TreeNode().list_to_node(root)




from bisect import bisect_left 
def closestNodes(root, queries):
    def dfs(n, v):                                                 
        if n: 
            dfs(n.left, v)
            v.append(n.val)
            dfs(n.right, v)     # inorder traversal of BST
    nums = []                                                     
    dfs(root, nums)   # nums will be sorted
    print(nums)                                              

    results, n = [], len(nums)
    
    for q in queries:                                              # [2] make queries using the binary
        i = bisect_left(nums, q)                                   #     search, then consider several
        if i < n and nums[i] == q : 
            results.append([q,q])         
        else:                                                      
            if i == 0: # if the query is smaller than the smallest element in the tree 
                results.append([-1,nums[0]])  
            elif i == n:  # if the query is larger than the largest element in the tree
                results.append([nums[-1],-1])
            else: 
                results.append([nums[i-1], nums[i]])
                
    return results

# print(closestNodes(root, queries))
"""
what is the difference between bisect_left and bisect_right?
bisect_left returns the index of the first element that is greater than or equal to x.
bisect_right returns the index of the first element that is greater than x.
"""
