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


nums1 = [1,2,3,4,5]
nums2 = [1,2,3,4,5]

# nums1 = [1,2,3]
# nums2 = [1,2,3]
# minimum cost is 3:
# swap (0,1) to make nums1 = [2,1,3] --> cost = 0 + 1
# then swap (0,2) to make nums1 = [3,1,2] --> cost = 0 + 2
# end result is nums1 = [3,1,2] != nums2 = [1,2,3] for all i

# for an array of length n, you need a maximum of n-1 swaps so that nums1 != nums2
# it is impossible when we have m identical elements that must go in m diifferent slots, 
# and n-m < m --> n < 2m

# nums1 = [1,2,2]
# nums2 = [1,2,2]
nums1 = [2,2,2,1,3]
nums2 = [1,2,2,3,3]

def minimumTotalCost(nums1, nums2):
    n = len(nums1)
    ans = 0
    freq = {}
    maxFrequency, maxFrequencyValue, toSwap = 0,0,0
    for i in range(n):
        if nums1[i] == nums2[i]:
            freq[nums1[i]] = freq.get(nums1[i], 0) + 1
            if freq[nums1[i]] > maxFrequency:
                maxFrequency = freq[nums1[i]]
                maxFrequencyValue = nums1[i]
            toSwap += 1
            ans += i
    print(freq, ans, maxFrequency)
    for i in range(n):
        if maxFrequency > toSwap//2 and nums1[i] != nums2[i] and  nums1[i]!=maxFrequencyValue and nums2[i]!=maxFrequencyValue:
            ans += i
            toSwap += 1

    if maxFrequency > toSwap//2: return -1

    return ans                    



# print( minimumTotalCost( nums1, nums2) )

stones = [0,2,5,6,7]
stones = [0,5,13,14]
stones = [0,5,12,25,28,35]
diff = lambda arr: [abs(arr[i+1]-arr[i]) for i in range(len(arr)-1)]
def maxJump(stones):
    res = stones[1] - stones[0]
    for i in range(2, len(stones)):
        res = max(res, stones[i] - stones[i - 2])
    return res

# print( maxJump(stones) )

from collections import defaultdict
import heapq
vals = [1,2,3,4,10,-10,-20]
edges = [[0,1],[1,2],[1,3],[3,4],[3,5],[3,6]]
k = 2

vals = [-5]
edges = []
k = 0
neighbors  = defaultdict(set)
for i, j in edges:
    neighbors[i].add(j)
    neighbors[j].add(i)

best = min(vals)
for i in range(len(vals)):
    heap = []
    for j in neighbors[i]:
        if vals[j] > 0 and len(heap) < k:
            heapq.heappush(heap, vals[j])
        elif vals[j] > 0 and len(heap) == k:
            heapq.heappushpop(heap, vals[j])
    best = max(best, sum(heap) + vals[i])
print(best)
        



