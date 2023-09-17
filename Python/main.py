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
        ans = max(ans, right - left + 1 - zeros)
    return ans - 1 if ans == n else ans


def summary_ranges(nums) -> str:
    """Превратить запись в отсортерованном массиве в более короткую"""
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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    if is_reflected([[1, 1], [-1, -1]]):
        print("Yes")
    else:
        print("No")
