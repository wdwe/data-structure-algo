def count_inver(arr):
    return _count_inver(arr, [0] * len(arr), 0, len(arr) - 1)

def _count_inver(arr, aux, beg, end):
    if beg == end:
        return 0
    mid = beg + (end - beg) // 2
    left_inv = _count_inver(arr, aux, beg, mid)
    right_inv = _count_inver(arr, aux, mid + 1, end)

    aux[beg: end + 1] = arr[beg: end + 1]
    
    split_inv = 0
    i, k, j = beg, beg, mid + 1

    while k <= end:
        if i == mid + 1:
            arr[k: end + 1] = aux[j: end + 1]
            break
        if j == end + 1:
            arr[k: end + 1] = aux[i: mid + 1]
            break
        if aux[i] <= aux[j]:
            arr[k] = aux[i]
            i += 1
        else:
            arr[k] = aux[j]
            j += 1
            split_inv += (mid - i + 1)
        k += 1

    return left_inv  + right_inv + split_inv


def brute_force_inver(arr):
    num_inv = 0
    for i in range(len(arr) - 1):
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[i]:
                num_inv += 1
    return num_inv

if __name__ == "__main__":
    import random
    length = 1000
    arr = [random.randint(0, 9999) for _ in range(length)]
    arr1 = list(arr)
    print(count_inver(arr))
    assert arr == sorted(arr)
    print(brute_force_inver(list(arr1)))