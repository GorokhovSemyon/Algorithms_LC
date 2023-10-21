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
    """
        Найти длиннейший подмассив из 1, после удаления одного 0/1
    """
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
    """
        Класс для реализации зигзагового итератора
    """

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
    """
        Приведение к строчным буквам и проверка на палиндром
    """
    s = [c.lower() for c in s if c.isalnum()]
    return all(s[i] == s[~i] for i in range(len(s) // 2))


def is_one_edit_distance(s, t):
    """
        Можно ли за одно изменение сделать строки равными
    """
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
    """
        Находит количество подмассивов в текущем, которые в сумме - k
    """

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

    def get_random(self) -> int:
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
    """
        Разворот односвязного списка O(n)
    """
    new_list = None

    while head:
        next_node = head.next
        head.next = new_list
        new_list = head
        head = next_node

    return new_list


def check_inclusion(s1, s2):
    """
        Проверка есть ли в каком-то виде
        перестановка строки s2 в s1
    """
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


def merge_k_lists(lists) -> ListNode:
    """
        Слияние k сортированных массивов,
        которые находятся в связном списке
    """
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
    """
       Задача с Яндекс контеста
       n - количество слотов
       m - количество нужных слотов
       c2 - цела за двойник
       c5 - цена за пятерник
    """
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
    tmp = hubs_needed * 5 + splitters_needed * 2 + (n - hubs_needed - splitters_needed)
    if m > tmp:
        total_cost1 += math.ceil((m - tmp)) * c2
        total_cost2 += math.ceil((m - tmp) / 4) * c5
        if total_cost1 >= total_cost2:
            total_cost += total_cost2
        else:
            total_cost += total_cost1
    total_cost += splitters_needed * c2 + hubs_needed * c5

    return total_cost


def is_subsequence(s: str, t: str) -> bool:
    """
        Есть ли набор символов из t в s
    """
    i, j = 0, 0
    while i < len(s) and j < len(t):
        if s[i] == t[j]:
            i += 1
        j += 1
    return i == len(s)


def longest_str_chain(words):
    """
        Поиск самой длинной цепочки (LC1048)
    """
    dp = {}
    for w in sorted(words, key=len):
        dp[w] = max(dp.get(w[:i] + w[i + 1:], 0) + 1 for i in range(len(w)))
    return max(dp.values())


def champagne_tower(poured: int, query_row: int, query_glass: int) -> float:
    """
        Задача про пирамиду из бокалов (LC799)
    """
    tower = [[0] * (i + 1) for i in range(query_row + 1)]
    tower[0][0] = poured

    for row in range(query_row):
        for glass in range(len(tower[row])):
            excess = (tower[row][glass] - 1) / 2.0
            if excess > 0:
                tower[row + 1][glass] += excess
                tower[row + 1][glass + 1] += excess

    return min(1.0, tower[query_row][query_glass])


def find_the_difference(s: str, t: str) -> str:
    """
        Найти отличающийся элемента (LC389) XOR
    """
    result = 0
    for char in s + t:
        result ^= ord(char)
    return chr(result)


def remove_duplicates(nums) -> int:
    """
        Удаление дубликатов, работа внутри исходного отсортированного массива
    """
    j = 1
    for i in range(1, len(nums)):
        if nums[i] != nums[i - 1]:
            nums[j] = nums[i]
            j += 1
    return j


def remove_duplicate_letters(s) -> str:
    """
        Удаление повторяющихся символов и сортировка по последнему появлению
    """
    stack = []
    seen = set()
    last_occ = {c: i for i, c in enumerate(s)}

    for i, c in enumerate(s):
        if c not in seen:

            while stack and c < stack[-1] and i < last_occ[stack[-1]]:
                seen.discard(stack.pop())
            seen.add(c)
            stack.append(c)

    return ''.join(stack)


def decode_at_index(s: str, k: int) -> str:
    """
        Задача поиска символа в предварительно расшифрованной строке
    """
    length = 0
    i = 0

    # в этом цикле вычисляется длина расшифрованной строки в которой будет ответ
    while length < k:
        if s[i].isdigit():
            length *= int(s[i])
        else:
            length += 1
        i += 1

    # в этом цикле осуществляется обратный проход, чтобы не создавать отдельную
    # переменную под расшифрованную строку (т.к. это большие затраты памяти)
    for j in range(i - 1, -1, -1):
        char = s[j]
        if char.isdigit():
            length //= int(char)
            k %= length
        elif k == 0 or k == length:
            return char
        else:
            length -= 1


def sort_array_by_parity(nums) -> list:
    """
        Сортировка по признаку чётности
    """
    return [x for x in nums if x % 2 == 0] + [x for x in nums if x % 2 == 1]


def is_monotonic(nums) -> bool:
    """
        Проверка монотонности последовательности
    """
    if len(nums) < 2:
        return True

    direction = 0  # 0 means unknown, 1 means increasing, -1 means decreasing

    for i in range(1, len(nums)):
        if nums[i] > nums[i - 1]:  # increasing
            if direction == 0:
                direction = 1
            elif direction == -1:
                return False
        elif nums[i] < nums[i - 1]:  # decreasing
            if direction == 0:
                direction = -1
            elif direction == 1:
                return False

    return True


def find132pattern(nums) -> bool:
    """
        Решение задачи с паттерном 132
    """
    stack, last = [], float('-inf')

    for num in reversed(nums):
        if num < last:
            return True
        while stack and stack[-1] < num:
            last = stack.pop()
        stack.append(num)
    return False


def reverse_words(s: str) -> str:
    """
        Разворот всех слов в строке
    """
    return ' '.join(map(lambda word: word[::-1], s.split()))


def winner_of_game(colors: str) -> bool:
    """
        Выявление победителя в игре
    """
    from collections import Counter
    from itertools import groupby

    c = Counter()  # пустой счётчик
    for x, t in groupby(colors):  # обновление счётчика
        c[x] += max(len(list(t)) - 2, 0)  # A/B : max извлечений

    if c['A'] > c['B']:
        return True
    return False


def winner_of_game_improved(colors: str) -> bool:
    """
        Улучшенная версия без использования доп памяти
    """
    alice_plays, bob_plays = 0, 0
    count = 1

    for i in range(1, len(colors)):
        if colors[i] == colors[i - 1]:
            count += 1
        else:
            if colors[i - 1] == 'A':
                alice_plays += max(0, count - 2)
            else:
                bob_plays += max(0, count - 2)
            count = 1

    if colors[-1] == 'A':
        alice_plays += max(0, count - 2)
    else:
        bob_plays += max(0, count - 2)

    return alice_plays > bob_plays


def numIdentical_pairs(nums) -> int:
    """
        Найти все хорошие пары (nums[i] == nums[j] and i < j)
    """
    res_dict = {}
    cnt = 0

    for elem in nums:
        if elem not in res_dict:
            res_dict[elem] = 1
        else:
            res_dict[elem] += 1
            cnt += res_dict[elem] - 1

    return cnt


def majority_element(nums):
    """
        Вывести числа, которые встречаются более n/3 раз в массиве
    """
    from collections import Counter
    num_counts = Counter(nums)

    # Создаем пустой список для хранения результатов
    result = []

    # Проходимся по парам (число, количество) в словаре num_counts
    for num, count in num_counts.items():
        # Проверяем условие
        if count > len(nums) // 3:
            result.append(num)

    return result


def integer_break(n: int) -> int:
    """
        Находит наибольшее произведение из разложения на слагаемые
    """
    if n == 2:
        return 1
    if n == 3:
        return 2
    q_3 = n // 3
    mod = n % 3
    if mod == 0:
        return 3 ** q_3
    elif mod == 1:
        return (3 ** (q_3 - 1)) * 4
    else:
        return (3 ** q_3) * 2


def max_dot_product(nums1, nums2) -> int:
    """
        Максимальное скалярное произведение
    """
    n = len(nums1)
    m = len(nums2)
    # Вспомогательная конструкция для хранения промежуточных результатов
    dot_prod = [[float('-inf')] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dot_prod[i][j] = max(nums1[i - 1] * nums2[j - 1],
                                 dot_prod[i - 1][j - 1] + nums1[i - 1] * nums2[j - 1])
            dot_prod[i][j] = max(dot_prod[i][j], dot_prod[i][j - 1])
            dot_prod[i][j] = max(dot_prod[i][j], dot_prod[i - 1][j])

    return dot_prod[n][m]


def search_range(nums, target):
    """
        Найти индексы элементов справа и слева за O(log(n))
    """
    def binary_search(nums, target, left) -> list:
        low, high = 0, len(nums) - 1
        index = -1
        while low <= high:
            mid = (low + high) // 2
            if nums[mid] == target:
                index = mid
                if left:
                    high = mid - 1
                else:
                    low = mid + 1
            elif nums[mid] < target:
                low = mid + 1
            else:
                high = mid - 1
        return index

    left_index = binary_search(nums, target, left=True)
    right_index = binary_search(nums, target, left=False)

    return [left_index, right_index]


def min_operations(nums) -> int:
    """
        Вычисляет минимальное количество операций
        для того, чтобы выполнить условия:
        - все элементы массива уникальны
        - max(nums) - min(nums) == len(nums) - 1
    """
    from bisect import bisect_right
    from sys import maxsize
    n = len(nums)
    nums = sorted(set(nums))
    ans = maxsize

    for i, s in enumerate(nums):
        elem = s + n - 1
        idx = bisect_right(nums, elem)

        ans = min(ans, n - (idx - i))

    return ans


def full_bloom_flowers(flowers, people) -> list:
    """
        Проверяет какое количество раскрывшихся цветов застанет
        (время цветения в формате [начало, конец] в flowers)
        человек пришедший во время (time) из списка people
    """
    from bisect import bisect_right, bisect_left
    start = sorted([s for s, e in flowers])
    end = sorted([e for s, e in flowers])

    return [bisect_right(start, time) - bisect_left(end, time) for time in people]


def find_in_mountain_array(target, mountain_arr) -> int:
    """
        Задача поиска target элемента в массиве, отражающем высоты
    """
    def find_peak(mountain_arr):
        left, right = 0, mountain_arr.length() - 1
        while left < right:
            mid = left + (right - left) // 2
            if mountain_arr.get(mid) < mountain_arr.get(mid + 1):
                left = mid + 1
            else:
                right = mid
        return left

    def binary_search(left, right, is_increasing):
        while left <= right:
            mid = left + (right - left) // 2
            mid_val = mountain_arr.get(mid)
            if mid_val == target:
                return mid
            if mid_val < target:
                if is_increasing:
                    left = mid + 1
                else:
                    right = mid - 1
            else:
                if is_increasing:
                    right = mid - 1
                else:
                    left = mid + 1
        return -1

    peak_index = find_peak(mountain_arr)
    result = binary_search(0, peak_index, True)
    if result == -1:
        result = binary_search(peak_index + 1, mountain_arr.length() - 1, False)
    return result


def min_cost_climbing_stairs(cost) -> int:
    """
        Минимальная стоимость достижения вершины
        cost - список стоимостей перемещения на 1 или 2 ступеньки с i-й
        prev1, prev2 - минимальная стоимость достижения предыдущих ступенек
    """
    n = len(cost)
    prev1, prev2 = 0, 0

    for i in range(2, n + 1):
        current_cost = min(prev1 + cost[i - 1], prev2 + cost[i - 2])
        prev2, prev1 = prev1, current_cost

    return prev1


def getRow(rowIndex: int) -> list:
    if rowIndex == 0:
        return [1]
    if rowIndex == 1:
        return [1, 1]

    res = [1]
    prev = 1

    for i in range(1, rowIndex + 1):
        next_val = prev * (rowIndex - i + 1) // i
        res.append(next_val)
        prev = next_val

    return res


def backspace_сompare(s: str, t: str) -> bool:
    """
        Решение задачи сравнения строк, если внутри
        них # эквивалентен backspace
        :param s:
        :param t:
        :return: bool
    """
    i = len(s) - 1  # Идём от конца строки
    j = len(t) - 1

    skipS = 0  # Счётчик "#", количество указывает удалённость от корректного символа
    skipT = 0

    while i >= 0 or j >= 0:
        while i >= 0:
            if s[i] == "#":
                skipS += 1  # Учёт "#"
                i -= 1

            elif skipS > 0:
                skipS -= 1  # Пропускаем символы, которые как бы будут удалены
                i -= 1

            else:
                break

        while j >= 0:
            if t[j] == "#":
                skipT += 1  # Такая же логика
                j -= 1

            elif skipT > 0:
                skipT -= 1
                j -= 1

            else:
                break
        # print("Comparing", s[i], t[j])  # Debug
        if i >= 0 and j >= 0 and s[i] != t[j]:  # Сравнивание текущих символов в обеих строках
            return False

        if (i >= 0) != (j >= 0):  # Also ensure that both the character indices are valid. If it is not valid,
            return False  # it means that we are comparing a "#" with a valid character.

        i -= 1
        j -= 1

    return True  # Если не вышли ни на одном return, то строки эквивалентны

def constrained_subset_sum(nums, k) -> int:
    """
        Вернуть максимальную сумму подсписка длиной не менее k
        LC1425
        :param nums:
        :param k:
        :return: максимальная сумма
    """
    from collections import deque
    dq = deque()
    for i in range(len(nums)):
        nums[i] += nums[dq[0]] if dq else 0

        while dq and (i - dq[0] >= k or nums[i] >= nums[dq[-1]]):
            dq.pop() if nums[i] >= nums[dq[-1]] else dq.popleft()

        if nums[i] > 0:
            dq.append(i)

    return max(nums)

def remove_element(nums, val: int):
    """
        Найти количество элементов списка, отличных от val
        LC27
        :param nums:
        :param val:
        :return: Количество элементов отличных от val
    """
    j = 0

    for i in range(len(nums)):
        if nums[i] != val:
            nums[j] = nums[i]
            j += 1

    return j


if __name__ == '__main__':
    # Для group_anagram()
    # input_strs = input().split(',')
    # print(group_anagrams(input_strs))

    # Для reverse_linked_list()
    # tmp = list_to_LL([1, 2, 3, 4, 5])
    # tmp2 = list_to_LL([1, 3, 4, 7])
    # tmp3 = list_to_LL([2, 4, 6])
    #
    # tmp_all = [tmp, tmp2, tmp3]
    # print(mergeKLists(tmp_all))

    # From Yandex.contest
    # n = int(input())  # Number of USB ports
    # m = int(input())  # Number of gadgets
    # c2 = int(input())  # Cost of a splitter
    # c5 = int(input())  # Cost of a hub
    #
    # result = min_usb_cost(n, m, c2, c5)
    # print(result)

    # For remove_duplicates()
    # nums = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]  # Input
    # print(remove_duplicates(nums))  # test

    # For min_operations
    nums = [1, 10, 100]
    print(min_operations(nums))

    # For backspace_compare
    # if (backspace_сompare("ash#", "asd#")):
    #     print("Yes")
    # else:
    #     print("No")
