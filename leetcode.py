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


creators = ["alice","bob","alice","chris"]
ids = ["one","two","three","four"]
views = [5,10,5,4]

creators = ["alice","alice","alice"]
ids = ["a","b","c"]
views = [1,2,2]
def mostViewedVideosByCreator(creators, ids, views):
    m = {}
    ID = {} # creator -> id, view
    # keep track of the highest views and the id with the lowest value
    highest = 0
    for creator, id, view in zip(creators, ids, views):
        m[creator] = m.get(creator, 0) + view
        if creator not in ID:
            ID[creator] = (id, view)
        # elif ID[creator][1] > view:
        #     ID[creator] = ID[creator]
        # if creator == 'p':
        #     print(id, view)
        # elif ID[creator][1] >= view:
        elif ID[creator][1] < view or (ID[creator][1] == view and (id < ID[creator][0])):
            ID[creator] = (id, view)
        # elif ID[creator][1] >= view and isLexographicallySmaller(ID[creator][0],id):
        #     ID[creator] = (id, view)
        # ID[creator] = (id, view) if creator not in ID else ID[creator] if ID[creator][1] >= view and isLexographicallySmaller(ID[creator][0],id) else (id, view)
        highest = max(highest, m[creator])
    # print(ID['p'])
    # print(max(m.items(), key=lambda x: x[1]))
    res = []
    for creator, totalViews in m.items():
        if totalViews == highest:
            res += [creator, ID[creator][0]],
    return (res)

# creators = ["u","ajf","n","kkmq","mwkim","p","ktjvr","ihmh","oulo","b","q","ofdim","rqbft","mdf","txt","xyjv","rlx","re","fyd","dq","frc","fag","xshlj","z","gii","z","le","fcvgf","yqbnk","vhke","udvp","rb","ppy","jywvl","xj","hb","ppqsq","waf","wpuw","qg","rnux","d","kxbcl","yoaqf","hnphp","w","nivm","hvymz","xze","bq","u","wbye","lmqoo","pc","q","t","jgiy","guv","fyc","ng","pvlg","aj","fhdo","maeu","zwfun","ravm","yypgx","cd","fkzmb","tvq","dm","aphdl","rbcp","dtcr","ehcv","k","c","hc","tg","wgin","mrrr","glr","fvxy","cap","xjjtq","hqp","bn","t","bc","cbbwf","ztxnz","xzmw","wsx","osim","m","cr","sp","s","v","he","mhcp","flz","owcx","zzi","p","wvvm","su","jp","qf","icqz","yy","k","jfv","qscyf","v","wj","nhlsi","r","vmd","nnbca","u","s","r","uoavb","qm","m","wio","ernca","h","bzrv"]
# ids = ["z","w","scghn","mnmvy","xgnhf","khuxq","hei","wsowq","yae","cs","h","hyrrv","vli","pma","bxsh","xmm","qkimd","ut","fj","xyzw","scjsj","y","k","c","qgx","fgk","mg","rmgse","txsgi","fzn","z","t","ew","yi","wzitv","tqg","b","o","sesb","jpw","u","rwc","ermmg","rjsjw","qh","mqf","ax","anh","hanz","ooors","mv","shaca","doon","d","x","f","egmiy","lbfvj","edrsz","epwai","spvwi","xlh","eux","c","flw","udo","bmft","ohnl","o","novqs","l","vosc","nasy","p","vk","cx","krdo","zdusc","pm","pcc","ye","sx","cjjx","je","i","iywdt","sd","kmx","dfq","kcq","zbgjc","awvkp","utdq","wos","y","sch","jmsxr","aewo","ngy","b","tt","dfzb","db","nzm","fl","om","s","gmpa","ie","yj","nbey","v","yz","oqf","glo","daeig","wim","ay","d","qsgp","l","y","er","e","pz","wn","ys","upvkl","lzjn","fjs"]
# views = [29,383,953,680,836,892,572,308,987,154,409,689,693,144,187,104,95,683,987,723,196,220,429,194,840,201,408,283,329,530,657,73,897,888,261,177,32,87,948,752,367,190,546,575,223,936,549,367,148,350,217,393,989,834,730,425,799,835,325,960,749,809,842,270,172,731,567,739,707,581,261,106,409,63,576,27,288,693,567,950,967,400,552,869,456,579,355,392,977,394,945,423,529,804,243,308,508,218,210,155,857,875,838,283,632,641,17,80,688,780,403,300,846,65,315,132,437,898,460,374,526,870,421,696,669,133,17,937,200,377]
# print(
#     mostViewedVideosByCreator(creators, ids, views)
#     )

n = 16
target = 6

n = 467
target = 6



# an integer is considered beautiful if digit sum is less than or equal to target
# return the minimum non-negative integer x such that n + x is beautiful
# digitSum(n + x) <= target
# x <= inverseDigitSum(target) - n
digitSum = lambda x: sum(map(int, str(x)))

def bruteForce(n, target):
    for x in range(1000000):
        if digitSum(n + x) <= target:
            return x
    return -1

# for n in range(467, 480):
#     print( n, bruteForce(n, target) )

# n = 467
# for target in range(1, 150):
#     print( target, bruteForce(n, target) )



# dp(n=467, target=6) = 3 + dp(n=470, target=6) = 3 + 30 + dp(n=500, target=6) = 3 + 30 + 500 + dp(n=1000, target=6)

def makeIntegerBeautiful(n, target):
    def helper(n,exp):
        if digitSum(n) <= target:
            return 0
        carry = pow(10, exp) - n % pow(10, exp)
        return helper(n + carry, exp + 1) + carry
    
    return helper(n, 1)

def makeIntegerBeautiful(n, target):
    n0 = n
    i = 0
    while sum(map(int, str(n))) > target:
        n = n // 10 + 1
        i += 1
    return n * (10 ** i) - n0

# print( makeIntegerBeautiful(n, target) )


root = [1,3,4,2,'null',6,5,'null','null','null','null','null',7]
queries = [4]
root = [5,8,9,2,1,3,7,4,6]
queries = [3,2,4,8]
root = TreeNode.list_to_node(root)
print(
    TreeNode.levelOrder(root)
)




from functools import lru_cache

def treeQueries(root, queries):
    @lru_cache(None)
    def height(r): return 0 if not r else 1 + max(height(r.left), height(r.right))
    ans = {}
    def dfs(r, depth, maximum):
        if not r: return
        ans[r.val] = maximum
        dfs(r.left, depth + 1, max(maximum, depth + height(r.right)))
        dfs(r.right, depth + 1, max(maximum, depth + height(r.left)))
    dfs(root, 0, 0)
    return [ans[q] for q in queries]

print(
    treeQueries(root, queries)
)
