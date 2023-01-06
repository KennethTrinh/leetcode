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
    @staticmethod
    def levelOrder(root):
        if not root:
            return []
        res = []
        queue = [root]
        while queue:
            res.append([node.val for node in queue])
            queue = [child for node in queue for child in (node.left, node.right) if child]
        return res



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


def minSwapsForSorted(arr):                               # this function simulates cycle sort,
    pos = {m:j for j,m in enumerate(sorted(arr))}         # namely, traverses every cycle in
    vis, tot = [0] * len(arr), 0                          # the permutation of elements and 
    for i in range(len(arr)):                             # counts the number of swaps 
        cnt = 0
        while not vis[i] and i != pos[arr[i]]:            # it is known that cycle sort is the
            vis[i], i = 1, pos[arr[i]]                    # sorting algorithm with the minmal
            cnt += 1                                      # number of memory operations (swaps)
        tot += max(0, cnt-1)                              # needed to sort an array
    return tot

robot = [0,4,6]
factory = [[2,2],[6,2]]

# robot = [9, 11, 99, 101]
# factory = [7, 10, 14, 96, 100, 103]
# factory = list(map(lambda x: [x], factory)) 
# robot = [1,-1]
# factory = [[-2,1],[2,1]]

robot = [0,4,6]
factory = [[2,2],[6,2]]

def minimumTotalDistance2(robot,factory):
    R, F = robot, factory
    R.sort()
    F.sort()
    dp = [0] + [float('inf')] * len(R)
    expand = lambda arr: [arr[i][0] for i in range(len(arr)) for j in range(arr[i][1])]
    F = expand(F)
    for f in F:
        for i in range(len(R), 0, -1):
            dp[i] = min(dp[i], dp[i-1] + abs(R[i-1]-f))
    print(dp)


# minimumTotalDistance2(robot, factory)



costs = [1,2,4,1]
k = 3
candidates = 3

costs = [17,12,10,2,7,2,11,20,8]
k = 3
candidates = 3

from heapq import heappush, heappop, heapify
def totalCost(costs, k, candidates):
    n, res = len(costs), 0                           # to use just one heap, we have to be 
    pairs  = [(t, i) for i, t in enumerate(costs)]   # able to extract the origin of a number
    l, r   = min(candidates,n//2), max(n-candidates,n//2)            # (left/right) when popping from the heap,
    pq     = pairs[:l] + pairs[r:]                 # thus, we save index for each number
    heapify(pq)
    for _ in range(k):                     # the smallest number comes first; if there
        cost, i = heappop(pq)                      # are several such numbers then the one with
        print(pq, i, l, r)        
        if i < l  : i, l = l, l+1                  # the smallest index (from the left) is taken
        if i >= r : i, r = r-1, r-1                
        if l <= r : heappush(pq, pairs[i])         # push numbers until they are depleted
        res += cost
    
    return res

    
# print(
# totalCost(costs, k, candidates)

# )

nums = [1,5,4,2,9,9,9]
k = 3

# nums = [1,1,1,7,8,9]
# k = 3

def maximumSubarraySum(nums, k):
    res, cur, pos, dup = 0, 0, [-1] * 100001, -1
    
    for i in range(0,len(nums)):
        cur += nums[i]                      # compute running sum for
        if i >= k: cur -= nums[i-k]         # the window of length k
        
        dup = max(dup, pos[nums[i]])        # update LAST seen duplicate
        
        if i - dup >= k:                    # if no duplicates were found
            res = max(res, cur)             # update max window sum

        pos[nums[i]] = i

    return res

print(res)
