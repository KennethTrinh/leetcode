
transpose = lambda arr: [list(row) for row in list(zip(*arr))]

pprint = lambda arr: [print(row) for row in arr]
isPowerofTwo = lambda x: (x & (x-1)) == 0
prefixSum = lambda arr: [0] + list(accumulate(arr))
countTotal = lambda P, x, y: P[y+1] - P[x]



# f = lambda x, r: r - abs( x%(r*2) -r)
# def f_inv(y, r, n):
#     val1 = (2*r) - y + (2*r*n)
#     val2 = y + (2*r*n)
#     return [min(val1, val2) , max(val2, val1)]

#ZIGZAG DP one
# result = ''
# for i in range(numRows):
#     r_indices = [ f_inv(i, numRows-1, n) for n in range( max(len(s)//numRows, 1)) ]
#     prev = None
#     print([y for x in r_indices for y in x] )
#     for j in [y for x in r_indices for y in x]:
#         if j >= len(s):
#             break
#         result += s[j] if prev != j else ''
#         prev = j
# print(result)

# if numRows ==1 or numRows >= len(s):
#     return s

# L = [''] * numRows
# for i, c in enumerate(s):
#     L[ f(i, numRows-1) ] += c

# ''.join(L)





# dp = [[1 for i in range(n)]] + [[1] + [0]*(n-1) for r in range(m-1)]

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





# first = Node(1)
# second = Node(2)
# third = Node(3)
# fourth = Node(4)

# first.neighbors = [second, fourth]
# second.neighbors = [first, third]
# third.neighbors = [second, fourth]
# fourth.neighbors = [first, third]






def buildAdjacencyList(n, edgesList):
        adjList = [[] for _ in range(n)]
        # c2 (course 2) is a prerequisite of c1 (course 1)
        # i.e c2c1 is a directed edge in the graph
        for c1, c2 in edgesList:
            adjList[c2].append(c1)
        return adjList


numCourses = 2
prerequisites = [[1,0]]
def isPossible(numCourses, prerequisites):
    adjList = buildAdjacencyList(numCourses, prerequisites)

    inDegrees = [0] * numCourses
    for v1, v2 in prerequisites:
        inDegrees[v1] += 1

    queue = []
    for v in range(numCourses):
        if inDegrees[v] == 0:
            queue += v,

    count = 0
    topoOrder = []
    while queue:
        node = queue.pop(0)
        topoOrder += node,
        count += 1
        for des in adjList[node]:
            inDegrees[des] -= 1
            if inDegrees[des] == 0:
                queue += des,

    print(True if count ==  numCourses else False)


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



s = 'babad'

# implement manacher's algorithm
def manacher(s):
    T = '#'.join('^{}$'.format(s))
    n = len(T)
    P = [0] * n
    C, R = 0, 0
    for i in range(1, n-1):
        P[i] = (R > i) and min(R - i, P[2*C - i])
        print(T[i + P[i] + 1], T[i - P[i] - 1])
        while T[i + P[i] + 1] == T[i - P[i] - 1]:
            P[i] += 1
        if i + P[i] > R:
            C, R = i, i + P[i]

    maxLen, centerIndex = max((n, i) for i, n in enumerate(P))        
    return s[(centerIndex - maxLen) // 2 : (centerIndex + maxLen) // 2]




# calculate minimum penalty for finding alternating prime sequence given array of integers
# arr = [3,7,1,4,6,6]
# output = 1
# explanation: can construct sequence [4,3,6,7,6] with penalty 1
def sieve(n):
    primes = [True] * (n+1)
    p = 2
    while p * p <= n:
        if primes[p] == True:
            for i in range(p * p, n+1, p):
                primes[i] = False
        p += 1
    return primes




def longestAlternatingPrimeSequence(nums):
    primes = sieve(1000000)
    if len(nums) < 2:
        return nums
    up = down = 1
    for i in range(1, len(nums)):
        if primes[nums[i]] and not primes[nums[i-1]]:
            up = down + 1
        elif not primes[nums[i]] and primes[nums[i-1]]:
            down = up + 1
    return max(up, down)

def citadel():
    def shrinkingNumberLine(points, k):
        points = sorted(points)
        n = len(points)
        result = points[-1] - points[0]
        for i in range(n - 1):
            first = points[0] + k
            current = points[i] + k
            next = points[i+1] - k
            last = points[-1] - k
            
            diff = max(last, current) - min(next, first)
            result = min(result, diff)
        return result

    print(shrinkingNumberLine([-3,0,1], 3))
    print(shrinkingNumberLine([4,7,-7], 5))
    print(shrinkingNumberLine([-100000, 100000], 100000))

    def isEven(s):
        return ord(s) % 2 == 0

    def isProductEven(arr):
        n = len(arr)
        for i in range(n):
            if ((ord(arr[i]) & 1) == 0):
                return True
        return False

    isEven = lambda s: 1 if ord(s) % 2 == 0 else 0

    s = ['abc', 'abcd']
    def solve(m, s):
        count = 0
        for substring in s:
            ordinal = isProductEven(list(substring)) #list(map(isEven, substring))
            if ordinal:
                count += 1
        return 'EVEN' if (len(s) - count) % 2 ==0 else 'ODD'
            

    print(solve(2, s))
    print(solve(50, ['aceace', 'ceceaa', 'abdbdbdbakjkljhkjh']))
    print(solve(47, ['azbde','abcher', 'acegk']))



def optiver():
    """
    E1 : Invalid Input Format
    E2: Duplicate pair
    E3: Parent node has more than 2 children
    E4: Multiple Roots
    E5: Input Contains Cycle
    """

    def SExpression(nodes):
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

    # nodes = "(A,B);(B,D);(D,E);(A,C);(C,F);(E,G)"
    # nodes = "(A,B);(A,C);(B,D);(D,C)"
    # print(SExpression(nodes))

# teamSize = [1, 2, 2, 3, 4]
# k = 2
# # teamSize = [1,2,3,4]
# # k = 1
# # output = 4
# # explanation: the team size of the last 2 teams can be reduced to 2 and 2
# # making the total number of teams with team size 2 equal to 4


# def calculateLargerElementArray(arr, curr, k):
#     result = []
#     for i in range(len(arr)):
#         if arr[i] > curr and k > 0:
#             result.append(True)
#             k -= 1
#         else:
#             result.append(False)
#     return result



# def maxTeamSizeAfterReducingKTeams(teamSize, k):
#     n = len(teamSize)
#     maximum = 0
#     for i in range(n):
#         counts = calculateLargerElementArray(teamSize, teamSize[i], k) #[True if teamSize[j] >= teamSize[i] else False for j in range(n)]
#         print( counts)
#         maximum = max(maximum, sum(counts) )
#     return maximum



# print(maxTeamSizeAfterReducingKTeams(teamSize, k))

def stripe():
    name_request = ['acct_1ab3c|The, Inc.']
    def check_avaliability(name_request):
        SUFFIXES = {'Inc.', 'Corp.', 'LLC', 'L.L.C.', 'LLC.'}
        STOPWORDS = {'The', 'An', 'A'}
        record = set()
        for request in name_request:
            user, name = request.split('|')
            name = name.replace('&', ' ')
            name = name.replace(',', ' ')
            # remove suffixes if at the end of the name
            if name.split() and name.split()[-1] in SUFFIXES:
                name = ' '.join(name.split()[:-1])
            # remove stopwords if at the beginning of the name
            if name.split()[0] in STOPWORDS:
                name = ' '.join(name.split()[1:])
            name = name.lower()
            # replace multiple spaces in name with single space
            name = ' '.join(name.split())
            # ignore the word 'and' unless it is the first word
            name = ' '.join([word for word in name.split() if word != 'and' or word == name.split()[0]])
            print(name)
            if name not in record and not(all(i == ' ' for i in name)):
                record.add(name)
                print( user + '|' + 'Name Available')
            else:
                print( user + '|' + 'Name Not Available')
    check_avaliability(name_request)

# given a square matrix nxn, where n is odd and contains only 0s, 1s, and 2s
# find the minimum numbber of operations needed to draw the letter Y with either 0s, 1s, or 2s
# Example: matrix = [ 
#                    [2, 0, 0, 0, 2],
#                    [1, 2, 1, 2, 0],
#                    [0, 1, 2, 1, 0],
#                    [0, 0, 2, 1, 1],
#                    [1, 1, 2, 1, 1]
#                  ]
# the output should be solution(matrix) = 8
# the best solution is to change the 0s to 1s

def roblox():
    def solution(matrix):
        n = len(matrix)
        mid = (n // 2)+ 1
        best = float('inf')
        possibilites = [(1, 2), (0, 1), (0, 2), (1, 0),  (2, 0), (2, 1)] #list(permutations([0,1,2], 2))
        for p in possibilites:
            count = 0
            background, foreground = p
            for i in range(mid):
                for j in range(mid):
                    if i == mid -1 and j == mid -1:
                        count += matrix[i][j] != foreground
                    elif i==j:
                        count += matrix[i][j] != foreground
                        count += matrix[i][n-j-1] != foreground
                        count += matrix[n-i-1][j] != background
                        count += matrix[n-i-1][n-j-1] != background
                    elif i == mid - 1:
                        count += matrix[i][j] != background
                        count += matrix[i][n-j-1] != background
                    elif j == mid-1:
                        count += matrix[i][j] != background 
                        count += matrix[n-i-1][j] != foreground
                    else:
                        count += matrix[i][j] != background
                        count += matrix[n-i-1][j] != background
                        count += matrix[i][n-j-1] != background
                        count += matrix[n-i-1][n-j-1] != background
            best = min(best, count)
        return best
    
# candidates = [2,3,6,7]
# target = 7
# def solution(candidates, target):
#     result = []
#     def backtrack(candidates, target, path):
#         if sum(path) == target:
#             result.append(path)
#         for i in range(len(candidates)):
#             if sum(path) + candidates[i] <= target:
#                 backtrack(candidates[i:], target, path + [candidates[i]])
#     backtrack(candidates, target, [])
#     return result


