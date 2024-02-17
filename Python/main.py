def is_reflected(points) -> bool:
    """
        Найти линию, паралельную y, которая отражает точки (O(n))
        :param points: входной массив, отражающий точки линии [List[int]]
        :return: является отражённой [bool]
    """
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
        :param nums: входной список [List[int]]
        :return: длина найденного подмассива [int]
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
    """
        Превратить запись в отсортированном массиве в более короткую
        :param nums: входной список [List[int]]
        :return: сокращённая запись списка [List[int]]
    """
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
    """
        Скомпрессовать строку аааа -> а4 и т.п.
        :param chars: входная строка [str]
        :return: строка результата [str]
    """
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
        Класс для реализации зигзагообразного итератора
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
        :param s: входная строка [str]
        :result: является палиндромом [bool]
    """
    s = [c.lower() for c in s if c.isalnum()]
    return all(s[i] == s[~i] for i in range(len(s) // 2))


def is_one_edit_distance(s, t):
    """
        Можно ли за одно изменение сделать строки равными
        :return: Можно/нельзя [bool]
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
        :param nums: входной массив чисел [List[int]]
        :param k: необходимая сумма подмассивов [int]
        :return res: количество подходящих массивов [int]
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
    """
        Перемещение нулей к концу списка
        :param nums: входной список [List[int]]
        :return: None
    """
    i = 0
    for j in range(len(nums)):
        if nums[j] != 0:
            if j != i:
                nums[j], nums[i] = nums[i], nums[j]
            i += 1


def group_anagrams(strs) -> list:
    """
        Нахождение перестановок из набора букв
        :param strs: входные строки [str]
        :return: список анограм [List[str]]
    """
    from collections import defaultdict
    anagram_dict = defaultdict(list)

    for word in strs:
        sorted_word = ''.join(sorted(word))
        anagram_dict[sorted_word].append(word)
    return list(anagram_dict.values())


class RandomizedSet:
    """
        Вставка, удаление, получение рандомного элемента за O(1)
    """

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
        '''
            Удаление элемента из RandomizedSet. Если элемент не существует, возвращаем False.
        '''
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
        :param s1: первая входная строка [str]
        :param s2: вторая входная строка [str]
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
        :param lists: связные списки
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
       :param n: количество слотов [int]
       :param m: количество нужных слотов [int]
       c2 - цела за двойник [int]
       c5 - цена за пятерник [int]
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
        :param s: входная строка, внутри которой осуществляется поиск [str]
        :param t: строка, символы которой ищутся в s [str]
        :return: Есть/нет [bool]
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
        :param words: список слов [List[str]]
    """
    dp = {}
    for w in sorted(words, key=len):
        dp[w] = max(dp.get(w[:i] + w[i + 1:], 0) + 1 for i in range(len(w)))
    return max(dp.values())


def champagne_tower(poured: int, query_row: int, query_glass: int) -> float:
    """
        Задача про пирамиду из бокалов
        LC799
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
        :param s: входная строка [str]
        :param k: индекс [int]
        :return: найденный символ [char]
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
        :param nums: входной массив [List[int]]
        :return: отсортированный список [List[int]]
    """
    return [x for x in nums if x % 2 == 0] + [x for x in nums if x % 2 == 1]


def is_monotonic(nums) -> bool:
    """
        Проверка монотонности последовательности
        :param nums: входной список
        :return: да/нет
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
        :param nums: входной список [List[int]]
    """
    stack, last = [], float('-inf')

    for num in reversed(nums):
        if num < last:
            return True
        while stack and stack[-1] < num:
            last = stack.pop()
        stack.append(num)
    return False


def reverse_words(s) -> str:
    """
        Разворот всех слов в строке
        :param s: входная строка [str]
        :return: перевёрнутая вхожная строка [str]
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
    cnt = 1

    for i in range(1, len(colors)):
        if colors[i] == colors[i - 1]:
            cnt += 1
        else:
            if colors[i - 1] == 'A':
                alice_plays += max(0, cnt - 2)
            else:
                bob_plays += max(0, cnt - 2)
            cnt = 1

    if colors[-1] == 'A':
        alice_plays += max(0, cnt - 2)
    else:
        bob_plays += max(0, cnt - 2)

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
    num_cnts = Counter(nums)

    # Создаем пустой список для хранения результатов
    result = []

    # Проходимся по парам (число, количество) в словаре num_cnts
    for num, cnt in num_cnts.items():
        # Проверяем условие
        if cnt > len(nums) // 3:
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
        :param nums: входной массив
        :param target: элемент - цель
        :return: пара индексов
    """

    def binary_search(nums, target, left) -> list:
        """
            :param nums: входной список [List[int]]
        """
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
        - элементы массива уникальны
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
        :param flowers:
        :param people:
        :return:
    """
    from bisect import bisect_right, bisect_left
    start = sorted([s for s, e in flowers])
    end = sorted([e for s, e in flowers])

    return [bisect_right(start, time) - bisect_left(end, time) for time in people]


def find_in_mountain_array(target, mountain_arr) -> int:
    """
        Задача поиска target элемента в массиве, отражающем высоты
        :param target: цель
        :param mountain_arr: список вершин
        :return: индекс целевой высоты
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
        Мин стоимость достижения вершины
        :param cost: список стоимостей перемещения на 1 или 2 ступеньки с i-й
        prev1, prev2 - мин стоимость достижения предыдущих ступенек
        :return: мин стоимость
    """
    n = len(cost)
    prev1, prev2 = 0, 0

    for i in range(2, n + 1):
        current_cost = min(prev1 + cost[i - 1], prev2 + cost[i - 2])
        prev2, prev1 = prev1, current_cost

    return prev1


def get_row(rowIndex: int) -> list:
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
        них # == backspace
        :param s: входная строка
        :param t: строчка для сравнения
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
        :return: Количество отличных от val
    """
    j = 0

    for i in range(len(nums)):
        if nums[i] != val:
            nums[j] = nums[i]
            j += 1

    return j


def str_str(haystack: str, needle: str) -> int:
    """
    Проверяет есть ли подстрока в строке, если есть,
    то возвращает индекс начала подстроки в строке
    LC29
    :param haystack:
    :param needle:
    :return: -1 / индекс первого вхождения
    """
    if len(needle) > len(haystack):
        return -1

    for i in range(len(haystack)):
        if haystack[i] == needle[0]:
            j = 1
            while j < len(needle) and i + j < len(haystack):
                if haystack[i + j] != needle[j]:
                    break
                j += 1
            if j == len(needle):
                return i

    return -1


def maximumScore(nums, k: int) -> int:
    """
        Дан массив целых чисел nums и целое число k
        Оценка подмассива определяется(i, j) как min(nums[i], nums[i+1],
         ..., nums[j]) * (j - i + 1). Хороший подмассив — это подмассив,
        в котором i <= k <= j
        :param nums:
        :param k:
        :return: Возвращает максимально возможную оценку хорошего подмассива
    """
    left, right = k, k
    min_val = nums[k]
    max_score = min_val

    while left > 0 or right < len(nums) - 1:
        if left == 0 or (right < len(nums) - 1 and nums[right + 1] > nums[left - 1]):
            right += 1
        else:
            left -= 1

        min_val = min(min_val, nums[left], nums[right])
        max_score = max(max_score, min_val * (right - left + 1))

    return max_score


def is_power_of_four(n) -> bool:
    """
        Является ли число степенью четвёрки
        LC342
        :param n: Проверяемое число
        :return: Да / Нет
    """
    mask = 0x55555555
    # return n > 0 and (n & (n - 1)) == 0 and (n & mask) == n
    return (n & mask) == n > 0 == (n & (n - 1))


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
def largest_values_rec(root) -> list:
    """
        Рекурсивное решение заачи поиска max
        элемента на уровне бинарного дерева
        LC515
        :param root: корень дерева
        :return: список максимумов
    """

    def dfs(node, depth):
        if not node:
            return

        if depth == len(res):
            res.append(node.val)
        else:
            res[depth] = max(res[depth], node.val)

        dfs(node.right, depth + 1)
        dfs(node.left, depth + 1)

    res = []
    dfs(root, 0)
    return res


def largest_values(root) -> list:
    """
        Решение заачи поиска max элемента
        на уровне бинарного дерева с deq
        LC515 v2.0
        :param root: корень дерева
        :return: список максимумов
    """
    from collections import deque
    if not root:
        return []

    result = []
    dq = deque([root])

    while dq:
        curr_level_size = len(dq)
        max_val = float('-inf')

        for _ in range(curr_level_size):
            node = dq.popleft()
            max_val = max(max_val, node.val)

            for child in [node.left, node.right]:
                if child:
                    dq.append(child)

        result.append(max_val)

    return result


def num_factored_binary_trees(arr):
    """
        Подсчёт количества возможных поддеревьев
        LC823
        :param arr: массив элементов из которых строятся поддеревья
        :return: количество возможных поддеревьев
    """
    arr.sort()
    # сортировка для гарантии того, что при переборе для предыдущих
    # элементов уже подсчитано число возможных поддереьев

    dct = {elem: 1 for elem in arr}
    # сохраняем все элементы входного массива в словарь (считаем что каждый - корень)
    # так можно сделать, так как каждое число встречаптся ровно один раз

    for elem in arr:
        for factor in arr:
            if factor == elem:
                break
            if elem % factor == 0 and elem // factor in dct:
                dct[elem] += dct[factor] * dct[elem // factor]

    return sum(dct.values()) % (pow(10, 9) + 7)


def kth_grammar(n, k):
    """
        Задача определения, какой элемент стоит
        на k-й позиции в n-м уровне
        LC779
        :param n:
        :param k:
        :return:
    """
    # Инициализируем флаг для отслеживания совпадения значений k и первого элемента.
    are_val_same = True

    # Рассчитываем общее количество элементов в n-й строке, которое равно 2^(n-1).
    n = 2 ** (n - 1)

    # Продолжаем выполнение до тех пор, пока не достигнем первой строки.
    while n != 1:
        # Делим количество элементов в строке пополам.
        n //= 2

        # Если k находится во второй половине строки, корректируем k и меняем флаг.
        if k > n:
            k -= n
            are_val_same = not are_val_same

    # Возвращаем 0, если флаг указывает на совпадение значений; в противном случае возвращаем 1.
    return 0 if are_val_same else 1


def cnt_vowel_permutation(n: int) -> int:
    """
        Подсчёт комбинаций с предусловиями
        LC1220
        :param n: длина строки
        :return: количество комбинаций
    """
    MOD = 10 ** 9 + 7

    a, e, i, o, u = 1, 1, 1, 1, 1

    for _ in range(1, n):
        a_next = e
        e_next = (a + i) % MOD
        i_next = (a + e + o + u) % MOD
        o_next = (i + u) % MOD
        u_next = a

        a, e, i, o, u = a_next, e_next, i_next, o_next, u_next

    return (a + e + i + o + u) % MOD


def poor_pigs(buckets: int, minutesToDie: int, minutesToTest: int) -> int:
    """
        Определить отравленное ведро
        LC458
        :param buckets: количество вёдер
        :param minutesToDie: время на тест...
        :param minutesToTest: Время для определения
        :return: количество требуемых свинок
    """
    ratio = minutesToTest / minutesToDie + 1

    poor_pigs = 0

    while ratio ** poor_pigs < buckets:
        poor_pigs += 1

    return poor_pigs


def sort_by_bits(arr):
    """
        Сортировка по количеству единиц в бинарном представлении
        LC1356
        :param arr: список чисел
        :return: специально отсортированный список [List[int]]
    """

    def binary_sort_key(num):
        cnt_of_positive_bits = bin(num).cnt('1')
        return (cnt_of_positive_bits, num)

    arr.sort(key=binary_sort_key)

    return arr


def find_array(pref) -> list:
    """
        Восстановить исходный список
        LC2433
        :param pref: список после применения к нему xor
        :return: исходный список
    """
    return [pref[0]] + [pref[i] ^ pref[i - 1] for i in range(1, len(pref))]


def is_reachable_at_time(sx, sy, fx, fy, t) -> bool:
    """
        Проверка, можно ли достичь конечной точки из начальной ровно за t единиц времени
        LC2849
        :param sx: x координата начальной точки
        :param sy: y координата начальной точки
        :param fx: x координата конечной точки
        :param fy: y координата конечной точки
        :param t: количество единиц времени
        :return: да или нет
    """
    xDiff = abs(sx - fx)
    yDiff = abs(sy - fy)

    # особый случай, который гарантированно False это связано с тем что за 1 единицу
    # времени нельзя выйти и вернуться в точку (если начальная и конечная совпадают)
    if xDiff == 0 and yDiff == 0 and t == 1:
        return False

    return (min(xDiff, yDiff) + abs(xDiff - yDiff)) <= t


def cnt_palindromic_subsequence(s: str) -> int:
    """
        Находит все возможные палиндромы длины 3
        LC1930
        :param s: входная строка
        :return: количество возможных палиндромов
    """
    cnt = 0
    for i in range(26):
        l, r = s.find(chr(i + 97)), s.rfind(chr(i + 97))
        if l != -1 and r != -1:
            cnt += len(set(s[l + 1:r]))
    return cnt


def find_diagonal_order(input_list):
    """
        Вывод последовательно всех диагоналей сонаправленных главной
        LC1424
        :param A: List[List[int]]
        :return: List[int]
    """
    from collections import defaultdict
    d = defaultdict(list)
    for i in range(len(input_list)):
        for j in range(len(input_list[i])):
            d[i + j].append(input_list[i][j])

    return [v for k in d.keys() for v in reversed(d[k])]


def get_sum_absolute_differences(nums):
    """
        Найти список, в котором разница между соседями фиксирована
        LC1685
        :param nums: Входной список List[int]
        :return: List[int]
    """
    # Sum(|Aj - Ai|) == tmp + (2*i-n)*x - total
    n = len(nums)
    tmp = 0  # член суммы Sum(Aj), где i < j
    total = sum(nums)  # член разложения суммы Sum(Aj), где j < i
    ans = [0] * n

    for i, x in enumerate(nums):
        ans[i] = (2 * i - n) * x + total - tmp
        tmp += x
        total -= x
    return ans

def hamming_weight(n):
    """
        Нахождения количества положительных битов
        LC191
        :param n: входной параметр битов [int]
        :return: количество единиц [int]
    """
    pos_bit_cnt = 0
    while n != 0:
        n &= n - 1
        pos_bit_cnt += 1
    return pos_bit_cnt

def array_strings_are_equal(word1, word2) -> bool:
    """
        Сравнение двух списков по внутренним символам
        LC1662
        :param word1: Первой слово [List[str]]
        :param word2: Второе слово [List[str]]
        :return: Да/Нет
    """
    if ''.join(word1) == ''.join(word2):
        return True
    return False

def largest_odd_number(num) -> str:
    """
        Найти наибольшее нечётное число в строке
        LC1903
        :param num: входная строка [str]
        :return: строка с наибольшим числом, либо пустая строка
    """
    for i in range(len(num) - 1, -1, -1):
        if int(num[i]) % 2 != 0:
            return num[:i + 1]

    return ""

def max_product(nums) -> int:
    """
        Нахождение максимума и пред максимума
        LC1464
        :param nums: входной список
        :return: (max-1)(prev_max-1)
    """
    max = nums[0]
    prev_max = 0
    for i in range(1, len(nums)):
        if nums[i] > max:
            prev_max = max
            max = nums[i]
        else:
            if nums[i] > prev_max:
                prev_max = nums[i]

    return (max - 1) * (prev_max - 1)

def num_special(mat):
    """
        Находит одинокие по горизонтали и вертикали "1"
        LC1582
        :param mat: входная матрица [List[List[int]]]
        :return: количество "специальных" единиц [int]
    """
    cnt = 0
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if mat[i][j] == 1 and sum(mat[i]) == 1 and sum(row[j] for row in mat) == 1:
                cnt += 1
    return cnt

def dest_city(paths) -> str:
    """
        Найти конечный путь во входном списке пар [начальный пункт, конечный пункт]
        LC1436
        :param paths: список пар [начальный пункт[str], конечный пункт[str]]
        :return: конечный пункт [str]
    """
    res_dict = {}
    for elem in paths:
        if elem[0] not in res_dict:
            res_dict[elem[0]] = 1
        else:
            res_dict[elem[0]] += 1
        if elem[1] not in res_dict:
            res_dict[elem[1]] = 3
        else:
            res_dict[elem[1]] += 1

    for key, value in reversed(res_dict.items()):
        if value == 3:
            return key

    return ""

def max_product_difference(nums):
    """
        Найти наибольшую разницу между произведениями
        LC1913
        :param nums: входной список List[int]
        :return: наибольшая разница [int]
    """
    largest, secondLargest = 0, 0
    smallest, secondSmallest = float('inf'), float('inf')

    for num in nums:
        if num < smallest:
            secondSmallest = smallest
            smallest = num
        elif num < secondSmallest:
            secondSmallest = num

        if num > largest:
            secondLargest = largest
            largest = num
        elif num > secondLargest:
            secondLargest = num

    return (largest * secondLargest) - (smallest * secondSmallest)

def image_smoother(img):
    """
        Найти среднее между значением в текущей ячейке и соседними
        LC661
        :param nums: входной двумерный список [List[List[int]]]
        :return: двумерный список средних [List[List[int]]]
    """
    n = len(img)
    m = len(img[0])
    res = []
    for i in range(n):
        temp = []
        for j in range(m):
            count = 1
            sum = img[i][j]
            if i - 1 >= 0 and j - 1 >= 0:
                sum += img[i - 1][j - 1]
                count += 1
            if j - 1 >= 0:
                sum = sum + img[i][j - 1]
                count += 1
            if i + 1 <= n - 1 and j - 1 >= 0:
                sum += img[i + 1][j - 1]
                count += 1
            if i + 1 <= n - 1:
                sum += img[i + 1][j]
                count += 1
            if i + 1 <= n - 1 and j + 1 <= m - 1:
                sum += img[i + 1][j + 1]
                count += 1
            if j + 1 <= m - 1:
                sum += img[i][j + 1]
                count += 1
            if i - 1 >= 0 and j + 1 <= m - 1:
                sum += img[i - 1][j + 1]
                count += 1
            if i - 1 >= 0:
                sum += img[i - 1][j]
                count += 1
            temp.append(sum // count)
        res.append(temp)
    return res

def max_width_of_vertical_area(points) -> int:
    """
        Нахождение наибольшей ширины между точками
        LC661
        :param points: входной список точек [List[List[int]]]
        :return: наибольшая ширина
    """
    points.sort(key=lambda x: x[0])

    max_width = 0

    for i in range(1, len(points)):
        width = points[i][0] - points[i - 1][0]
        max_width = max(max_width, width)

    return max_width


def max_score(s: str) -> int:
    """
        Найти наибольшую сумму, считаемую по "0" и "1" при разделении строки на две
        LC1422
        :param s: Входная строка
        :return: наибольшая сумма
    """
    max_score, count_zeros_left = 0, 0
    count_ones_right = s.count('1')

    for i in range(len(s) - 1):
        count_zeros_left += s[i] == '0'
        count_ones_right -= s[i] == '1'
        max_score = max(max_score, count_zeros_left + count_ones_right)

    return max_score


class Solution(object):
    def min_operations(self, s):
        """
            Строка только с "0" и "1", нужно сделать так, чтобы рядом не было
            одинаковых, посчитать минимальное количество замен для этого
            LC1758
            :param s: входная строка [str]
            :return: min количество операций [int]
        """
        c_0 = s[0]
        cnt1 = self.cnt(s, c_0)
        cnt2 = self.cnt(s, '0' if c_0 == '1' else '1') + 1
        return min(cnt1, cnt2)

    def cnt(self, s, c_pre):
        cnt = 0
        for i in range(1, len(s)):
            current = s[i]
            if current == c_pre:
                cnt += 1
                c_pre = '0' if c_pre == '1' else '1'
            else:
                c_pre = current
        return cnt

def plus_one(digits) -> list:
    """
        Сложение с "1" decimal числа
        LC66
        :param digits: decimal число в виде списка
        :return: decimal число digits + 1
    """
    for i in range(len(digits) - 1, -1, -1):
        if digits[i] == 9:
            digits[i] = 0
        else:
            digits[i] = digits[i] + 1
            return digits
    return [1] + digits


def make_equal(words) -> bool:
    """
        LC1897
        :param words: список слов
        :return: возвращает булевый результат, можно ли сделать все слова одинаковыми
    """
    n = len(words)
    f = [0] * 26
    for word in words:
        for c in word:
            f[ord(c) - ord('a')] += 1

    for x in f:
        if x % n != 0:
            return False

    return True


def min_operations(nums) -> int:
    """
        Функция нахождения минимального количества операций, за которое можно очистить массив
        LC2870
        :param nums: список входных значений [List[int]]
        :return: min количество операций [int]
    """
    from collections import Counter
    cntr = Counter(nums)

    cnt = 0
    for t in cntr.values():
        if t == 1:
            return -1
        cnt += t // 3
        if t % 3:
            cnt += 1

    return cnt


def minSteps(s: str, t: str) -> int:
    """
        Поиск минимального числа перестановокв строках, чтобы сделать из них анаграмму 
        (чтобы были одинаковые символы в строках)
        LC1347
        :param s: первая входная строка
    """
    count_s = [0] * 26
    count_t = [0] * 26

    for char in s:
        count_s[ord(char) - ord('a')] += 1

    for char in t:
        count_t[ord(char) - ord('a')] += 1

    steps = 0
    for i in range(26):
        steps += abs(count_s[i] - count_t[i])

    return steps // 2


def majorityElement(nums):
    """
        Найти число, которое появляется чаще всего во входном списке
        LC169
        :param nums: входной список [List[int]]
        :result: чаще всего появляющееся число [int]
    """
    dict = {}

    for n in nums:
        if n not in dict:
            dict[n] = 1
        else:
            dict[n] += 1

    return max(dict, key=dict.get) 


if __name__ == '__main__':
    # Для group_anagram()
    # input_strs = input().split(',')
    # print(group_anagrams(input_strs))

    # Для reverse_linked_list()
    # tmp = list_to_LL([1, 2, 3, 4])
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
