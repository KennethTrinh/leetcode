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

message = "this is really a very awesome message"
limit = 9
# ["thi<1/14>","s i<2/14>","s r<3/14>","eal<4/14>","ly <5/14>","a v<6/14>","ery<7/14>"," aw<8/14>","eso<9/14>","me<10/14>"," m<11/14>","es<12/14>","sa<13/14>","ge<14/14>"]

def splitMessage(message, limit):
    sz = lambda s: len(str(s))
    cur = k = i = 0 # curr is the length of all indices, k is the denominator, i is the index of the message
    while 3 + sz(k) * 2 < limit and cur + len(message) + (3 + sz(k)) * k > limit * k:
        k += 1
        cur += sz(k)
    res = []
    if 3 + sz(k) * 2 < limit:
        for j in range(1, k + 1):
            l = limit - (sz(j) + 3 + sz(k))
            res.append( f'{message[i:i+l]}<{j}/{k}>' ) 
            i += l
    return res



    
print(
splitMessage(message, limit)
)
edges = [[0,1],[1,2],[1,3],[3,4]]
bob = 3
amount = [-2,4,2,-4,6]
def mostProfitablePath(edges, bob, amount) -> int:
    n = len(edges) + 1
    G = [[] for i in range(n)]
    for i,j in edges:
        G[i].append(j)
        G[j].append(i)
    seen = [0] * n

    def dfs(i, d0):
        seen[i] = 1
        res = float('-inf')
        db = 0 if i == bob else n
        for j in G[i]:
            if seen[j]: continue
            cur, kk = dfs(j, d0 + 1)
            res = max(res, cur)
            db = min(db, kk)
        if res == float('-inf'): res = 0
        if d0 == db: res += amount[i] // 2
        if d0 < db: res += amount[i]
        return res, db + 1

    return dfs(0, 0)[0]

# print(mostProfitablePath(edges, bob, amount))

low = 2
high = 3
zero = 1
one = 2

low = 3
high = 3
zero = 1
one = 1
def countGoodStrings(low, high, zero, one):
    dp = [0] * (high + 1)
    dp[zero] +=1
    dp[one] +=1
    for i in range( min(zero, one) + 1, high + 1):
        dp[i] += dp[i - zero]
        dp[i] += dp[i - one]
        dp[i] %= 1000000007
        # if i - zero >= 0:
        #     dp[i] += dp[i - zero]
        # if i - one >= 0:
        #     dp[i] += dp[i - one]
    return dp[high] if low == high else sum(dp[i] for i in range(low, high+1)) % 1000000007


print(
countGoodStrings(low, high, zero, one)
)

