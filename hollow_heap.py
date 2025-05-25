from sortedcontainers import SortedDict
class HollowHeapNode:
    def __init__(self, key, item, node_id):
        self.key = key
        self.item = item
        self.id = node_id
        self.children = None
        self.next = None
        self.second_parent = None
        self.rank = 0
        self.hollow = False


class HollowHeap:
    def __init__(self):
        self.nodes = {}
        self.root = None
        self.nodes_used = 0

        self.rankmap = {}
        self.to_delete = []

        self.eqlinks = self.links = self.ranked = 0
        self.inserts = self.decs = 0
        self.in_heap = SortedDict()

    def make_new_node(self, key, item):
        self.nodes_used += 1
        node = HollowHeapNode(key, item, self.nodes_used)
        self.nodes[self.nodes_used] = node
        return node

    def link(self, u_id, v_id):
        self.links += 1
        u = self.nodes[u_id]
        v = self.nodes[v_id]

        if u.key < v.key:
            parent, child = u, v
        elif v.key < u.key:
            parent, child = v, u
        else:
            self.eqlinks += 1
            if u.rank < v.rank:
                parent, child = u, v
            else:
                parent, child = v, u

        child.next = parent.children
        parent.children = child
        return parent.id

    def push(self, key, item):
        self.inserts += 1
        node = self.make_new_node(key, item)
        if not self.root:
            self.root = node.id
        else:
            self.root = self.link(self.root, node.id)
        return node.id

    def find_min(self):
        if not self.root:
            return None
        return self.nodes[self.root]

    def decrease_key(self, u_id, new_key):
        self.decs += 1
        u = self.nodes[u_id]

        if u.id == self.root:
            u.key = new_key
            return u.id

        new_node = self.make_new_node(new_key, u.item)
        if u.rank > 2:
            new_node.rank = u.rank - 2
        else:
            new_node.rank = 0

        u.hollow = True
        old_root = self.root
        self.root = self.link(self.root, new_node.id)

        if self.root == old_root:
            new_node.children = u
            u.second_parent = new_node.id

        return new_node.id

    def delete_min(self):
        if not self.root:
            return

        self.to_delete = [self.root]
        max_rank = -1
        rankmap = {}

        while self.to_delete:
            parent_id = self.to_delete.pop(0)
            parent = self.nodes[parent_id]
            child = parent.children

            while child:
                next_child = child.next

                if not child.hollow:
                    while child.rank in rankmap:
                        other_id = rankmap.pop(child.rank)
                        other = self.nodes[other_id]
                        child = self.nodes[self.link(child.id, other.id)]
                        self.ranked += 1
                        child.rank += 1

                    rankmap[child.rank] = child.id
                    max_rank = max(max_rank, child.rank)
                else:
                    if not child.second_parent:
                        self.to_delete.append(child.id)
                    elif child.second_parent == parent.id:
                        child.second_parent = None
                        break
                    else:
                        child.second_parent = None
                        child.next = None

                child = next_child

        if not rankmap:
            self.root = None
            return

        rank_items = sorted(rankmap.items(), reverse=True)
        self.root = rank_items[0][1]
        for _, node_id in rank_items[1:]:
            self.root = self.link(self.root, node_id)

    def empty(self):
        return self.root is None

    def print_heap(self, index=None, level=0):
        if index is None:
            index = self.root
        if index is None:
            return

        node = self.nodes[index]
        print("   " * level + f"{node.id}.key={node.key}, rank={node.rank}, hollow={node.hollow}")

        child = node.children
        while child:
            self.print_heap(child.id, level + 1)
            child = child.next

if (__name__=='__main__'):
    hh = HollowHeap()
    a = hh.push(10, "a")
    b = hh.push(5, "b")
    c = hh.push(7, "c")
    hh.decrease_key(a, 3)
    print("Min:", hh.find_min())  # Should print "a"
    hh.delete_min()
    print("Min after delete_min:", hh.find_min())  # Should be next smallest