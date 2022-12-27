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


s = "103301"

def bruteForce(s):
    # get all subsequences of length 5 of s and check if they are palindromes
    subseqeunces = [''.join(arr) for arr in subsets(s) if len(arr) == 5]
    # print(subseqeunces)
    isPalindrome = lambda x: x == x[::-1]
    return sum([1 for sub in subseqeunces if isPalindrome(sub)])


def countAppearances(s,t):
    """
    Counts the number of subsequences of t that appear in s
    """
    m,n = len(s), len(t)
    dp = [0] * (n+1)
    dp[-1] = 1
    for i in range(m):
        for j in range(n):
            if s[i] == t[j]:
                dp[j] += dp[j+1]
    return dp[0]

# print(
#     countAppearances('0000000', '00000'),
# )





def countPalindromes(s):
    ans = 0 
    for x in range(10): 
        for y in range(10):
            pattern = f"{x}{y}|{y}{x}" 
            dp = [0]*6
            dp[-1] = 1 
            for i in range(len(s)): 
                for j in range(5): 
                    if s[i] == pattern[j] or j == 2:
                        dp[j] += dp[j+1]
            ans = (ans + dp[0]) % 1_000_000_007
    return ans 



# print(countPalindromes(s))



from collections import Counter
customers = "YYNY"
customers = "NNNNN"
customers = "YYYY"
customers = "YNYY"
def bestClosingTime(customers):
    c = Counter(customers)
    penalty = best = c['Y']
    index = 0
    # print(penalty, best)
    for i in range(1, len(customers)):
        if customers[i-1] == 'Y':
            penalty -= 1
        else:
            penalty += 1
        if penalty < best:
            best = penalty
            index = i
        # print(penalty)
    index = len(customers) if customers[-1] == 'Y' and penalty - 1 < best else index
    return index

def bestClosingTime(customers):
    binarize = lambda arr, y: [1 if x == y else 0 for x in arr]
    prefixSum = lambda arr: [0] + list(accumulate(arr))
    prefixSumN = prefixSum( binarize(customers, 'N') )
    suffixSumY = prefixSum( binarize(customers[::-1], 'Y') ) [::-1]
    ans, ind = float('inf'), 0
    for i in range(len(customers) + 1):
        penalty = prefixSumN[i] + suffixSumY[i]
        if penalty < ans:
            ans = penalty
            ind = i
    return ind
    
    # y,n = [0], [0]
    # cnt = 0
    # for i in range(len(customers)):
    #     if customers[i] == 'N':
    #         cnt += 1
    #     n.append(cnt)
    # cnt= 0
    # for i in range(len(customers)-1, -1, -1):
    #     if customers[i] == 'Y':
    #         cnt += 1
    #     y.append(cnt)
    # y = y[::-1]
    # print(n)
    # print(y)
    # print(prefixSumN)
    # print(suffixSumY)


# print(
# bestClosingTime(customers)
# )        
            
import numpy as np
grid = [[0,1,1],[1,0,1],[0,0,1]]
# grid = [[1,1,1],[1,1,1]]

def onesMinusZeroes(grid):
    grid = np.array(grid)
    m, n = grid.shape

    onesRow = grid.sum(axis=1)
    onesCol = grid.sum(axis=0)
    zerosRow = m - onesRow
    zerosCol = n - onesCol

    print('Ones Row: ', onesRow)
    print('Ones Col: ', onesCol)
    print('Zeros Row: ', zerosRow)
    print('Zeros Col: ', zerosCol)

    # for i in range(m):
    #     for j in range(n):
    #         matrix[i][j] = onesRow[i] + onesCol[j] - zerosRow[i] - zerosCol[j]

    #         matrix[i][j] = onesRow[i] + onesCol[j] - (m - onesRow[i]) - (n - onesCol[j])
    #         matrix[i][j] = 2 * (onesRow[i] + onesCol[j]) - m - n

    # print(matrix)

    matrix = onesRow[:, np.newaxis] + onesCol - zerosRow[:, np.newaxis] - zerosCol
    matrix = np.add.outer(onesRow, onesCol) - np.add.outer(zerosRow, zerosCol)
    print('Ones Row:', onesRow[:, np.newaxis])
    print('Ones Col:', onesCol)
    print('Zeros Row:', zerosRow[:, np.newaxis])
    print('Zeros Col:', zerosCol)

    print('u: ', np.add.outer(onesRow, onesCol))
    print('v: ', np.add.outer(zerosRow, zerosCol))

