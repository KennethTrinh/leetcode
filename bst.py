class BSTNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


class BST:
    def __init__(self, root=None):
        self.root = None if root is None else BSTNode(root)

    def add(self, value, node=None):
        if node is None:
            node = self.root

        if self.root is None:
            self.root = BSTNode(value)
        elif value < node.value:
            if node.left is not None:
                self.add(value, node.left)
            else:
                node.left = BSTNode(value)
        else:
            if node.right is not None:
                self.add(value, node.right)
            else:
                node.right = BSTNode(value)

    def print_postorder(self, node=None):
        """
        Left, Right, Root
        """
        if node is None:
            node = self.root
        if node.left is not None:
            self.print_postorder(node.left)
        if node.right is not None:
            self.print_postorder(node.right)
        print(node.value)
    
    def print_levelorder(self):
        if self.root is None:
            return

        queue = [self.root]
        while queue:
            level = []
            for _ in range(len(queue)):
                node = queue.pop(0)
                level.append(node.value)
                if node.left is not None:
                    queue.append(node.left)
                if node.right is not None:
                    queue.append(node.right)
            print(" ".join(level))
    def insert(self, value, child):
        if self.root is None:
            return
        else:
            self._insert(value, child, self.root)
    def _insert(self, value, child, node):
        if node is None:
            return
        if node.value == value:
            if node.left is None:
                node.left = BSTNode(child)
            elif node.right is None:
                node.right = BSTNode(child)
        else:
            self._insert(value, child, node.left)
            self._insert(value, child, node.right)
        

        
bst = BST()
bst.add('K')
bst.add('G')
bst.add('P')
bst.add('A')
bst.add('H')
bst.add('N')
bst.insert('H', 'I')
bst.insert('H', 'J')

print('Level Order: ')
bst.print_levelorder()
print('Post Order: ')
bst.print_postorder()