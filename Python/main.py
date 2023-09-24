def is_reflected(points) -> bool:
    """Найти линию, паралельную y, которая отражает точки (O(n))"""
    min_x, max_x = float('inf'), float('-inf')
    point_set = set()
    for x, y in points:
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        point_set.add((x, y))
    s = min_x + max_x
    # для каждой пары (x, y) создаётся кортеж (s-x, y)
    # далее проверяется есть ли уже такая пара множества
    return all((s - x, y) in point_set for x, y in points)


def longest_subarray(nums) -> int:
    """Найти длиннейший подмассив из 1, после удаления одного 0/1"""
    n = len(nums)
    left = 0
    zeros = 0
    ans = 0
    for right in range(n):
        if nums[right] == 0:
            zeros += 1
        while zeros > 1:
            if nums[left] == 0:
                zeros -= 1
            left += 1
        # ans = max(ans, right - left + 1 - zeros)
        ans = max(ans, right - left)
    return ans - 1 if ans == n else ans


def summary_ranges(nums) -> str:
    """Превратить запись в отсортированном массиве в более короткую"""
    if not nums:
        return []
    ranges = []
    start = nums[0]
    for i in range(1, len(nums)):
        if nums[i] != nums[i - 1] + 1:
            if start == nums[i - 1]:
                ranges.append(str(start))
            else:
                ranges.append(str(start) + "->" + str(nums[i - 1]))
            start = nums[i]
    if start == nums[-1]:
        ranges.append(str(nums[-1]))
    else:
        ranges.append(str(start) + "->" + str(nums[-1]))
    return ranges


def compress(chars) -> int:
    """Скомпрессовать строку аааа -> а4 и т.п."""
    ans = 0
    i = 0
    while i < len(chars):
        letter = chars[i]
        cnt = 0
        while i < len(chars) and chars[i] == letter:
            cnt += 1
            i += 1
        chars[ans] = letter
        ans += 1
        if cnt > 1:
            for c in str(cnt):
                chars[ans] = c
                ans += 1
    return ans


class ZigzagIterator:
    """Класс для реализации зигзагового итератора"""

    def __init__(self, v1, v2):
        self.data = [(len(v), iter(v)) for v in (v1, v2) if v]

    def next(self):
        if not self.data:
            raise StopIteration

        length, it = self.data.pop(0)
        result = next(it)

        if length > 1:
            self.data.append((length - 1, it))

        return result

    def hasNext(self):
        return bool(self.data)

    def print_data(self):
        print(self.data)


def is_palindrome(s) -> bool:
    """Приведение к строчным буквам и проверка на палиндром"""
    s = [c.lower() for c in s if c.isalnum()]
    return all(s[i] == s[~i] for i in range(len(s) // 2))


def is_one_edit_distance(s, t):
    """Можно ли за одно изменение сделать строки равными"""
    m, n = len(s), len(t)
    for i in range(min(m, n)):
        if s[i] != t[i]:
            if m > n:
                return s[i + 1:] == t[i:]
            elif m == n:
                return s[i + 1:] == t[i + 1:]
            else:
                return t[i + 1:] == s[i:]
    return abs(m - n) == 1


def subarray_sum(nums, k):
    """Находит количество подмассивов в текущем, которые в сумме - k"""
    res = 0
    prefix_sum = 0
    d = {0: 1}

    for num in nums:
        prefix_sum += num
        if prefix_sum - k in d:
            res += d[prefix_sum - k]
        if prefix_sum not in d:
            d[prefix_sum] = 1
        else:
            d[prefix_sum] += 1

    return res


def move_zeros(nums) -> None:
    """Перемещение нулей в конец массива"""
    i = 0
    for j in range(len(nums)):
        if nums[j] != 0:
            if j != i:
                nums[j], nums[i] = nums[i], nums[j]
            i += 1


def group_anagrams(strs) -> list:
    """Нахождение перестановок из набора букв"""
    from collections import defaultdict
    anagram_dict = defaultdict(list)

    for word in strs:
        sorted_word = ''.join(sorted(word))
        anagram_dict[sorted_word].append(word)
    return list(anagram_dict.values())


class RandomizedSet:
    """Вставка, удаление, получение рандомного элемента за O(1)"""

    def __init__(self):
        self.data_map = {}  # Словарь для хранения элементов и их индексов
        self.data = []  # Список для хранения элементов

    def insert(self, val: int) -> bool:
        # Вставка элемента в RandomizedSet. Если элемент уже существует, возвращаем False.
        if val in self.data_map:
            return False

        # Добавляем элемент в словарь, где ключ - элемент, значение - его индекс в списке
        self.data_map[val] = len(self.data)

        # Добавляем элемент в список
        self.data.append(val)

        return True

    def remove(self, val: int) -> bool:
        # Удаление элемента из RandomizedSet. Если элемент не существует, возвращаем False.
        if val not in self.data_map:
            return False

        # Получаем последний элемент из списка
        last_elem_in_list = self.data[-1]
        index_of_elem_to_remove = self.data_map[val]

        # Перемещаем последний элемент в место элемента, который мы хотим удалить
        self.data_map[last_elem_in_list] = index_of_elem_to_remove
        self.data[index_of_elem_to_remove] = last_elem_in_list

        # Удаляем последний элемент из списка
        self.data.pop()

        # Удаляем элемент, который мы хотим удалить, из словаря
        self.data_map.pop(val)
        return True

    def getRandom(self) -> int:
        # Получение случайного элемента из списка
        import random
        return random.choice(self.data)


class Node:
    def __init__(self, key, value):
        self.key = key
        self.val = value
        self.next = None
        self.prev = None


class LRUCache:
    """Реализация LRU"""

    def __init__(self, capacity: int):
        self.net = capacity
        self.cache = dict()
        self.mru = Node(-1, -1)
        self.lru = Node(-1, -1)
        self.mru.next = self.lru
        self.lru.prev = self.mru

    def get(self, key: int) -> int:
        if key in self.cache:
            node = self.cache[key]
            self.remove(node)
            self.insert(node)
            return node.val
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            node = self.cache[key]
            self.remove(node)
            self.insert(Node(key, value))
        else:
            if len(self.cache) == self.net:
                self.remove(self.lru.prev)
            self.insert(Node(key, value))

    def remove(self, node):
        del self.cache[node.key]
        node.prev.next = node.next
        node.next.prev = node.prev

    def insert(self, node):
        self.mru.next.prev = node
        node.prev = self.mru
        node.next = self.mru.next
        self.mru.next = node

        self.cache[node.key] = node


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __repr__(self):
        return "ListNode(val=" + str(self.val) + ", next={" + str(self.next) + "})"


def list_to_LL(arr):
    if len(arr) < 1:
        return None
    if len(arr) == 1:
        return ListNode(arr[0])
    return ListNode(arr[0], next=list_to_LL(arr[1:]))


def reverse_linked_list(head: ListNode):
    """Разворот односвязного списка O(n)"""
    new_list = None

    while head:
        next_node = head.next
        head.next = new_list
        new_list = head
        head = next_node

    return new_list


def checkInclusion(s1, s2):
    """Проверка есть ли в каком-то виде перестановка строки s2 в s1"""
    from collections import Counter
    cntr, w, match = Counter(s1), len(s1), 0

    for i in range(len(s2)):
        if s2[i] in cntr:
            if not cntr[s2[i]]:
                match -= 1
            cntr[s2[i]] -= 1
            if not cntr[s2[i]]:
                match += 1

        if i >= w and s2[i - w] in cntr:
            if not cntr[s2[i - w]]:
                match -= 1
            cntr[s2[i - w]] += 1
            if not cntr[s2[i - w]]:
                match += 1

        if match == len(cntr):
            return True

    return False


def mergeKLists(lists) -> ListNode:
    """Слияние k сортированных массивов, которые находятся в связном списке"""
    v = []
    for i in lists:
        x = i
        while x:
            v += [x.val]
            x = x.next
    v = sorted(v, reverse=True)
    ans = None
    for i in v:
        ans = ListNode(i, ans)
    return ans


def min_usb_cost(n, m, c2, c5):
    """Задача с Яндекс контеста
       n - количество слотов
       m - количество нужных слотов
       c2 - цела за двойник
       c5 - цена за пятерник"""
    import math

    if m <= n:
        return 0
    total_cost = 0
    total_cost1 = 0
    total_cost2 = 0
    hubs_needed = 0
    splitters_needed = 0
    if c2 * 4 >= c5:
        hubs_needed = (m - n) // 4
    else:
        splitters_needed = m - n
    tmp = hubs_needed*5 + splitters_needed*2 + (n - hubs_needed - splitters_needed)
    if m > tmp:
        total_cost1 += math.ceil((m-tmp)) * c2
        total_cost2 += math.ceil((m-tmp)/4) * c5
        if total_cost1 >= total_cost2:
            total_cost += total_cost2
        else:
            total_cost += total_cost1
    total_cost += splitters_needed * c2 + hubs_needed * c5

    return total_cost


def isSubsequence(self, s: str, t: str) -> bool:
    """Есть ли набор символов из t в s"""
    i, j = 0, 0
    while i < len(s) and j < len(t):
        if s[i] == t[j]:
            i += 1
        j += 1
    return i == len(s)


def longestStrChain(words):
    """Поиск самой длинной цепочки (LC1048)"""
    dp = {}
    for w in sorted(words, key=len):
        dp[w] = max(dp.get(w[:i] + w[i + 1:], 0) + 1 for i in range(len(w)))
    return max(dp.values())

def champagneTower(poured: int, query_row: int, query_glass: int) -> float:
    """Задача про пирамиду из бокалов (LC799)"""
    tower = [[0] * (i + 1) for i in range(query_row + 1)]
    tower[0][0] = poured

    for row in range(query_row):
        for glass in range(len(tower[row])):
            excess = (tower[row][glass] - 1) / 2.0
            if excess > 0:
                tower[row + 1][glass] += excess
                tower[row + 1][glass + 1] += excess

    return min(1.0, tower[query_row][query_glass])


if __name__ == '__main__':
    # Для group_anagram()
    # input_strs = input().split(',')
    # print(group_anagrams(input_strs))

    # Для reverse_linked_list()
    tmp = list_to_LL([1, 2, 3, 4, 5])
    tmp2 = list_to_LL([1, 3, 4, 7])
    tmp3 = list_to_LL([2, 4, 6])

    tmp_all = [tmp, tmp2, tmp3]
    print(mergeKLists(tmp_all))

    # From Yandex.contest
    # n = int(input())  # Number of USB ports
    # m = int(input())  # Number of gadgets
    # c2 = int(input())  # Cost of a splitter
    # c5 = int(input())  # Cost of a hub
    #
    # result = min_usb_cost(n, m, c2, c5)
    # print(result)