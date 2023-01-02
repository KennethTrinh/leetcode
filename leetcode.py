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

s = "abaccdbbd"
k = 3

def maxPalindrome(s,k):
    isPalindrome = lambda i, j: False if i< 0 else s[i:j] == s[i:j][::-1]
    dp = [0] * (len(s) + 1)
    for i in range(k, len(s) + 1):
        dp[i] = dp[i - 1]
        if isPalindrome(i-k, i):
            dp[i] = max(dp[i], dp[i - k] + 1)
        if isPalindrome(i-k-1, i):
            dp[i] = max(dp[i], dp[i - k - 1] + 1)

        
        # print(s[i-k:i], s[i-k-1:i], isPalindrome(s[i-k:i]), isPalindrome(s[i-k-1:i]))

    print(dp[-1])
        # if isPalindrome(s[])



root = [1,4,3,7,6,8,5,'null','null','null','null',9,'null',10]
root = [1,3,2,7,6,5,4]
# root = [1,2,3,4,5,6]
root = TreeNode.list_to_node(root)
# print(TreeNode.levelOrder(root))

def minOperations(root):
    def levelOrder(root):
        if not root:
            return []
        res = []
        queue = [root]
        while queue:
            res.append([node.val for node in queue])
            queue = [child for node in queue for child in (node.left, node.right) if child]
        return res
    nodes = levelOrder(root)
    def perm(arr):                                            # this function simulates cycle sort,
        pos = {m:j for j,m in enumerate(sorted(arr))}         # namely, traverses every cycle in
        vis, tot = [0] * len(arr), 0                          # the permutation of elements and 
        for i in range(len(arr)):                             # counts the number of swaps 
            cnt = 0
            while not vis[i] and i != pos[arr[i]]:            # it is known that cycle sort is the
                vis[i], i = 1, pos[arr[i]]                    # sorting algorithm with the minmal
                cnt += 1                                      # number of memory operations (swaps)
            tot += max(0, cnt-1)                              # needed to sort an array
        return tot
    
    return sum( perm(level) for level in nodes)

def minSwapsForSorted(arr):                                            # this function simulates cycle sort,
    pos = {m:j for j,m in enumerate(sorted(arr))}         # namely, traverses every cycle in
    vis, tot = [0] * len(arr), 0                          # the permutation of elements and 
    for i in range(len(arr)):                             # counts the number of swaps 
        cnt = 0
        while not vis[i] and i != pos[arr[i]]:            # it is known that cycle sort is the
            vis[i], i = 1, pos[arr[i]]                    # sorting algorithm with the minmal
            cnt += 1                                      # number of memory operations (swaps)
        tot += max(0, cnt-1)                              # needed to sort an array
    return tot
    

# print(minSwapsForSorted([7, 6, 5, 4]))


# print( TreeNode.levelOrder( minOperations(root)) )
"""Given an array of positive integers a, 
your task is to calculate the sum of every possible a[i] ∘ a[j], 
where a[i] ∘ a[j] is the concatenation of the string representations of a[i] and a[j] respectively.
"""
nums = [5, 3]
nums = [1,2,3]

def concatSum(nums):
    """
    Quadratic time complexity
    """
    return sum( int(str(a)+str(b)) for a,b in product(nums, repeat=2) )

def concatSum(nums):
    lowSum, offsetSum = 0, 0
    for i in range(len(nums)):
        lowSum += nums[i]
        size = len(str(nums[i]))
        offset = 10**size
        offsetSum += offset
    print(lowSum, offsetSum)
    return lowSum * len(nums) + offsetSum * lowSum


nums = [3,6,2,7,1]
k = 6

print(subarrays(list(range(1,4))))

from math import lcm
def subarrayLCM(nums,k):
    res = 0
    for i in range(len(nums)):
        l = nums[i]
        for j in range(i, len(nums)):
            l = lcm(l, nums[j])
            if l == k:
                res += 1
            if l > k:
                break
    return res

# print(subarrayLCM(nums,k))


# print(subarrayLCM(nums, 6))
print(subarrays([1,11,1]))