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



nums = [1,2,3,4]
k = 4

def countPartitions(nums, k):
    if sum(nums) < k * 2: return 0
    mod = 10**9 + 7
    dp = [1] + [0] * (k - 1)
    for a in nums:
        for i in range(k - 1 - a, -1, -1):
            print(i + a, i, dp)
            dp[i + a] += dp[i]
    print(dp)
    return (pow(2, len(nums), mod) - sum(dp) * 2) % mod

def countPartitions(nums, k):
    # find the number of subset < k
    # dp[0] ~ dp[k - 1]
    if sum(nums) < 2 * k: return 0
    n = len(nums)
    MOD = 10 ** 9 + 7 
    dp = [[0] * k for _ in range(n + 1)]
    
    # dp[i][j] = the total number of subsets of nums[:i] that sum to j
    for i in range(n + 1):
        dp[i][0] = 1

    for i in range(1, n + 1):
        for j in range(1, k):
            if j - nums[i - 1] >= 0:
                dp[i][j] = dp[i - 1][j] + dp[i - 1][j - nums[i - 1]] # choose current element
            else:
                dp[i][j] = dp[i - 1][j] # don't choose current element
    pprint(dp)
    return (2**n - sum(dp[-1]) * 2) % MOD

# print(countPartitions(nums, k))

price = [13,5,1,8,21,2]
k = 3
def maximumTastiness(price, k):
    price.sort()
    def check(x):
        last, count, i = price[0], 1, 1
        for i in range(1, len(price)):
            if price[i] - last >= x:
                last, count = price[i], count + 1
        return count >= k
    lo, hi = 0, 10 ** 9
    while lo < hi:
        mid = (lo + hi) // 2
        if check(mid): 
            lo = mid + 1
            check(7)
        else: hi = mid
    return lo - 1


# print(maximumTastiness(price, k))

s = "aabaaaacaabc"
k = 2



from functools import lru_cache
def takeCharacters(s,k):
    @lru_cache(None)
    def helper(count, l, r, a, b, c):
        if a>=k and b>=k and c>=k:
            return count
        if l > r: return -1
        leftCount = helper(count+1, l+1, r, 
                           a+1 if s[l] == 'a' else a,
                           b+1 if s[l] == 'b' else b,
                           c+1 if s[l] == 'c' else c)
        rightCount = helper(count+1, l, r-1,
                            a+1 if s[r] == 'a' else a,
                            b+1 if s[r] == 'b' else b,
                            c+1 if s[r] == 'c' else c)
        return min(leftCount, rightCount)
    left, right = 0, len(s) - 1
    return helper(0, left, right, 0, 0, 0)




from collections import Counter
def takeCharacters(s, k):
    count = Counter(s)
    for c in 'abc':
        if count[c] < k:
            return -1
    n = len(s)
    i, j, ans = n - 1, n - 1, n
    while i >= 0:
        count[s[i]] -= 1

        # while any(count[c] < k for c in 'abc'):
        while any( c < k for c in count.values() ):
            count[s[j]] += 1
            j -= 1
        
        ans = min(ans, i + ( n -1 - j ))
        i -= 1
    
    return ans

def takeCharacters(s,k):
    count = Counter(s)
    for c in 'abc':
        if count[c] < k:
            return -1
    n = len(s)
    i,j,ans = 0, 0, n# i is the right pointer, j is the left pointer
    while i < n:
        count[s[i]] -= 1
        while any(c < k for c in count.values()):
            count[s[j]] += 1
            j += 1
        print(i, j, j + (n - 1 - i))
        ans = min(ans, j + (n - 1 - i))
        i += 1
    return ans


print(takeCharacters(s, k))