from itertools import accumulate

transpose = lambda arr: [list(row) for row in list(zip(*arr))]

isPowerofTwo = lambda x: (x & (x-1)) == 0
prefixSum = lambda arr: [0] + list(accumulate(arr))
countTotal = lambda P, x, y: P[y+1] - P[x]
MOD = 1_000_000_007

def fill(*args, num=0) -> list:
    if len(args) == 0:
        raise ValueError("fill() takes at least 1 argument (0 given)")
    
    if len(args) == 1 and isinstance(args[0], int):
        return [num for _ in range(args[0])]

    return [fill(*args[1:]) for _ in range(args[0])]

def pp(arr) -> None:
    def get_dims(arr):
        try:
            return 1 + get_dims(arr[0])
        except:
            return 0
    if get_dims(arr) == 1:
        print(arr)
    elif get_dims(arr) == 2:
        for row in arr:
            print(row)
    elif get_dims(arr) == 3: # 3D array :D
        import pandas as pd
        import numpy as np
        df = [ [ np.array(arr[i][j]) for j in range(len(arr[0])) ] for i in range(len(arr)) ]
        df = pd.DataFrame(df)
        print(df)

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
        queue = []
        root = TreeNode(arr[0])
        queue.append(root)
        i = 1
        while queue:
            node = queue.pop(0)
            if i < len(arr) and arr[i] != 'null':
                node.left = TreeNode(arr[i])
                queue.append(node.left)
            i += 1
            if i < len(arr) and arr[i] != 'null':
                node.right = TreeNode(arr[i])
                queue.append(node.right)
            i += 1
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
    @staticmethod
    def inOrder(root):
        if not root:
            return []
        return TreeNode.inOrder(root.left) + [root.val] + TreeNode.inOrder(root.right)
    @staticmethod
    def preOrder(root):
        if not root:
            return []
        return [root.val] + TreeNode.preOrder(root.left) + TreeNode.preOrder(root.right)
    @staticmethod
    def postOrder(root):
        if not root:
            return []
        return TreeNode.postOrder(root.left) + TreeNode.postOrder(root.right) + [root.val]



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




def goodness(arr):
    dp = {0: 0}
    for x in arr:
        for target, limit in list(dp.items()):
            if limit >= x: # no way
                continue
            new_target = target | x
            if new_target not in dp or dp[new_target] > x:
                dp[new_target] = x
    return sorted(dp.keys())

from collections import Counter
def maxLengthOfConsistentLogs(nums):
    min_freq = min(Counter(nums).values())
    
    ans, start = 0, -1
    frequency = Counter()
    for end in range(len(nums)):
        frequency[nums[end]] += 1
        while frequency[nums[end]] > min_freq:
            start += 1
            frequency[nums[start]] -= 1
        ans = max(ans, end - start)
    return ans


from math import ceil

def feas(execution, x, y, operations):
    # diff = extra time major job gets
    diff = x - y
    needed_major_ops = 0
    
    for el in execution:
        # Subtract base progress (y seconds per operation)
        remaining = max(0, el - operations*y)
        # How many times this job needs to be major
        needed_major_ops += ceil(remaining / diff)
        
    # Check if required major operations fits within total operations
    return needed_major_ops <= operations

def minTimeToExecute(execution, x, y):
    l, r = 0, sum(execution)
    ret = -1
    while l <= r:
        m = (l + r) // 2
        if feas(execution, x, y, m):
            ret = m
            r = m - 1
        else:
            l = m + 1
            
    return ret

def isAchievable(target, throughput, scaling_cost, budget):
    total_cost = 0
    for i in range(len(throughput)):
        if throughput[i] >= target:
            continue
        # Calculate required scaling
        scales_needed = ((target - throughput[i]) + throughput[i] - 1) // throughput[i]
        cost = scales_needed * scaling_cost[i]
        total_cost += cost
        if total_cost > budget:
            return False
    return True

def maxThroughput(throughput, scaling_cost, budget):
    left = min(throughput)
    right = left + budget * max(throughput) // min(scaling_cost)
    
    result = left
    while left <= right:
        mid = (left + right) // 2
        if isAchievable(mid, throughput, scaling_cost, budget):
            result = mid
            left = mid + 1
        else:
            right = mid - 1
            
    return result


def countStableSegmentsOptimal(n, capacity):
    # Create prefix sum array
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + capacity[i]
    
    count = 0
    # For each possible left endpoint
    for i in range(n-2):
        # For each possible right endpoint
        for j in range(i+2, n):
            # Get interior sum using prefix array
            interior_sum = prefix[j] - prefix[i+1]
            # Check if segment is stable
            if capacity[i] == capacity[j] and capacity[i] == interior_sum:
                count += 1
    
    return count

# Test cases
n1 = 5
capacity1 = [9,3,3,3,9]
print(countStableSegmentsOptimal(n1, capacity1))  # Should print 2

n2 = 7
capacity2 = [9,3,1,2,3,9,10]
print(countStableSegmentsOptimal(n2, capacity2))  # Should print 2

# Test case
# throughput = [4, 2, 7]
# scaling_cost = [3, 5, 6]
# budget = 32
# print(maxThroughput(throughput, scaling_cost, budget))  # Should output 10

# print(minTimeToExecute([3, 4, 1, 7, 6] , x = 4, y = 2 ))

# print(goodness([4, 2, 4, 1]))
# print(maxLengthOfConsistentLogs([1,2,1,3,4,2,4,3,3,4]))