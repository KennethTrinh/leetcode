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



"""
E1 : Invalid Input Format
E2: Duplicate pair
E3: Parent node has more than 2 children
E4: Multiple Roots
E5: Input Contains Cycle
"""

def SExpression(nodes):
    if checkE1(nodes):
        return "E1"
    graph = [[False for _ in range(26)] for _ in range(26)]
    node = set()
    E2 = checkE2(nodes, graph, node)
    E3 = checkE3(graph)
    if E2:
        return "E2"
    if E3:
        return "E3"

    numRoots = 0
    root = ' '

    for n in node:
        for i in range(26):
            if graph[i][ord(n) - ord('A')]:
                break
            if i == 25:
                numRoots += 1
                root = n
                if dfs(n, graph, [False for _ in range(26)]):
                    return "E5"
    if numRoots > 1:
        return "E4"
    if numRoots == 0:
        return "E5"
    return helper(root, graph)

def checkE1(nodes):
    """
    check that nodes is:
    - one line
    - has no leading or trailing whitespace
    - each pair is formatted as an open parenthesis ( 
        followed be a parent followed by a comma followed by a child
        followed by a closing parenthesis )
    - all values are single uppercase letters
    - parent child pairs are separated by a single space
    # example:(A,B) (B,C) (A,E) (B,D)
    """
    ok = True
    for i, c in enumerate(nodes):
        if i%6 == 0:
            ok &= c == '('
        elif i%6 == 1:
            ok &= len(c) == 1 and c.isalpha() and c.isupper()
        elif i%6 == 2:
            ok &= c == ','
        elif i%6 == 3:
            ok &= len(c) == 1 and c.isalpha() and c.isupper()
        elif i%6 == 4:
            ok &= c == ')'
        elif i%6 == 5:
            ok &= c == ' '
    return not ok

print(checkE1("(A,B) (B,C) (A,E) (B,D)"))
            
def addIntChar(x, c):
    return chr(x + ord(c))

def checkE3(graph):
    for i in range(26):
        count = 0
        for j in range(26):
            if graph[i][j]:
                count += 1
        if count > 2:
            return True
    return False

def checkE2(nodes, graph, node):
    for i in range(1, len(nodes), 6):
        parent = ord(nodes[i]) - ord('A')
        child = ord(nodes[i + 2]) - ord('A')
        if graph[parent][child]:
            return True
        graph[parent][child] = True
        node.add(addIntChar(parent, 'A'))
        node.add(addIntChar(parent, 'A'))

    return False

def dfs(node, graph, visited):
    if visited[ord(node) - ord('A')]:
        return True
    visited[ord(node) - ord('A')] = True
    for i in range(26):
        if graph[ord(node) - ord('A')][i]:
            if dfs(addIntChar(i, 'A'), graph, visited):
                return True

    return False

def helper(root, graph):
    l = ""
    r = ""
    for i in range(26):
        if graph[ord(root) - ord('A')][i]:
            l = helper(addIntChar(i, 'A'), graph)
            for j in range(i + 1, 26):
                if graph[ord(root) - ord('A')][j]:
                    r = helper(addIntChar(j, 'A'), graph)
                    break
            break
    return "(" + root + l + r + ")"
# test the above code

nodes = "(A,B);(B,D);(D,E);(A,C);(C,F);(E,G)"
nodes = "(A,B);(A,C);(B,D);(D,C)"
print(SExpression(nodes))

# def NumDaysBetween(year1, month1, day1, year2, month2, day2):
#     if year1 > year2:
#         return -1
#     count1 = yearsToDays(month1,year1) + monthsToDays(month1,year1) + day1
#     count2 = yearsToDays(month2,year2) + monthsToDays(month2,year2) + day2
#     print(count1, count2)
#     return count2 - count1

# yearsToDays = lambda month,year: 365*year + countLeaps(month,year)

# def countLeaps(month,year):
#     if month<=2:
#         year -=1
#     return (year // 4) - (year // 100) + (year // 400)

# monthsToDays = lambda month, year: sum(DaysInMonth(i, year) for i in range(1, month))
    
# def DaysInMonth(month,year):
#     if month == 2:
#         if isLeapYear(year):
#             return 29
#         return 28
#     if month in [4,6,9,11]:
#         return 30
#     return 31

# def isLeapYear(year):
#     if year % 4 != 0:
#         return False
#     if year % 100 != 0:
#         return True
#     if year % 400 != 0:
#         return False
#     return True

# # def monthsToDays(month,year):
# #     days = 0
# #     for i in range(1,month):
# #         days += DaysInMonth(i,year)
# #     return days

# # Constant time
# # def monthsToDays(month,year):
# #     days = 0
# #     for i in range(1, month):
# #         days += daysInMonthWithoutLeap(i)
# #     return days

# # def daysInMonthWithoutLeap(month): #doesnt account for feb 29, that is accounted for in countLeaps
# #     if month == 2:
# #         return 28
# #     if month in [4, 6, 9, 11]:
# #         return 30
# #     return 31

# print(NumDaysBetween(2010, 1, 1, 2011, 7, 1))