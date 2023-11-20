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




def random_grid_generator(m, n):
    import random
    random.seed(1)
    grid = zeros(m, n)
    P = {
        -1: 0.1,
        0: 0.4,
        1: 0.5
    }
    for i in range(m):
        for j in range(n):
            grid[i][j] = random.choices(list(P.keys()), list(P.values()))[0]
    return grid

# print(random_grid_generator(5, 5))

grid = [
    [0, 1, 1, 0, 0], 
    [0, 1, 1, -1, -1], 
    [1, 0, 1, -1, 0], 
    [1, 0, 1, 1, -1], 
    [-1, 1, 1, 0, 0]
]

grid = [
    [0, 1, -1],
    [1, 0, -1],
    [1, 1, 1]
]


isValid = lambda i, j, grid: 0 <= i < len(grid) and 0 <= j < len(grid[0]) and grid[i][j] != -1

def brute_force(grid):
    from itertools import combinations
    m, n = len(grid), len(grid[0])
    # generate all paths from (0, 0) to (m-1, n-1)
    def dfs(grid):
        def backtrack(i, j, path):
            if not isValid(i, j, grid):
                return
            if i == m-1 and j == n-1:
                paths.append(path)
                return
            for di, dj in [(1, 0), (0, 1)]:
                backtrack(i+di, j+dj, path + [(i+di, j+dj)])
        paths = []
        backtrack(0, 0, [(0, 0)])
        return paths

    paths = dfs(grid)
    best = 0
    for path1, path2 in combinations(paths, 2):
        result = 0
        unique = set(path1 + path2)
        for i, j in unique:
            if grid[i][j] == 1:
                result += 1
        best = max(best, result)
    return best

# print(brute_force(grid))



# Fact: in an m x n grid, there are comb(m+n-2, m-1) paths from (0, 0) to (m-1, n-1) --> (do the stars and bars thing)

# assuming the grid is n x n -> there are <= comb(n + n - 2, n-1) = comb(2n-2, n-1) paths from (0, 0) to (n-1, n-1) 
# --> (and therefore comb(2n-2, n-1) paths from (n-1, n-1) to (0, 0))

# So to solve this problem, we need to consider all paths to (n-1, n-1) , and pick any other path back to (0, 0),
# so there are theoretically comb( comb(2n-2, n-1), 2) paths to consider, which is too many

# if you just want to go from (0, 0) to (n-1, n-1), the solution is as follows:
def one_way(grid):
    n = len(grid)
    dp = fill(n, n)
    for i in range(n):
        for j in range(n):
            if grid[i][j] == -1:
                dp[i][j] = 0
            elif i == 0 and j == 0:
                dp[i][j] = grid[i][j]
            elif i == 0:
                dp[i][j] = dp[i][j-1] + grid[i][j]
            elif j == 0:
                dp[i][j] = dp[i-1][j] + grid[i][j]
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]) + grid[i][j]

# pp(dp)

# however, if you want to go from (0, 0) to (n-1, n-1) and back to (0, 0), we need a different approach:
n = len(grid)
dp = fill(n, n)
dp[0][0] = grid[0][0]

n = len(grid)

# 2d DP array where dp[x1][x2] contains the maximum number of cherries
# that can be attained from a 2-legged path starting from (x1, y1) for leg 1
# (x2, y2) for leg 2. This DP only stores the answers for a given path 
# length. That's why we don't store the y coordinates, they can be derived
# since we know X + Y = length of path.
dp = [[-1] * n for _ in range(n)]
dp[0][0] = grid[0][0]

max_path_length = 2 * n - 2
for path_length in range(1, max_path_length + 1):
    # For a given path_length, the range [start_x, end_x] represents all the
    # valid x coordinates that we can consider. We iterate through this
    # range backwards to prevent overwriting old dp values which allows us
    # not to need a second dp array.
    start_x = min(n - 1, path_length)
    end_x = max(0, path_length - n + 1)
    for x1 in range(start_x, end_x - 1, -1):
        # We can let x2 >= x1 since the ordering of the points doesn't matter
        for x2 in range(start_x, x1 - 1, -1):
            y1 = path_length - x1
            y2 = path_length - x2
            
            if grid[x1][y1] == -1 or grid[x2][y2] == -1:
                dp[x1][x2] = -1
                continue

            # Consider 4 cases:
            # 1. Move right for path 1 and 2.
            # 2. Move right for path 1 and down for path 2.
            # 3. Move down for path 1 and right for path 2.
            # 4. Move down for both path 1 and path 2.
            # Return -1 if out of bounds
            dp[x1][x2] = max(
                dp[x1][x2],
                -1 if x1 == 0 else dp[x1 - 1][x2],
                -1 if x2 == 0 else dp[x1][x2 - 1],
                -1 if x1 == 0 and x2 == 0 else dp[x1 - 1][x2 - 1],
            )

            if dp[x1][x2] >= 0:
                num_cherries = grid[x1][y1] + grid[x2][y2] if x1 != x2 else grid[x1][y1]
                dp[x1][x2] += num_cherries

result = dp[n - 1][n - 1]
pp(dp)