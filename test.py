class Solution:
    def pathSum(self, root, sum):
        """
        前缀和
        """
        def pre_order(root, sum_list):
            if not root:
                return 0
            sum_list = [val + root.val for val in sum_list]
            sum_list.append(root.val)

            count = sum_list.count(sum)
            return count + pre_order(root.left, sum_list) + pre_order(root.right, sum_list)

        return pre_order(root, [])
