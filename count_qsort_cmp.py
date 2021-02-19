import random



def count_qsort_cmp(arr, pivot = "first", inplace = False):
    assert pivot in {"first", "last", "median"}
    if inplace == False:
        arr = list(arr)
    random.shuffle(arr)
    return _count_qsort_cmp(arr, 0, len(arr) - 1, pivot)


def _count_qsort_cmp(arr, beg, end, pivot_type):
    if beg >= end:
        return 0
    if pivot_type == "first":
        pass
    elif pivot_type == "last":
        arr[beg], arr[end] = arr[end], arr[beg]
    elif pivot_type == "median":
        mid = beg + (end - beg) // 2
        temp = [beg, mid, end]
        for i in range(2):
            for j in range(i + 1, 3):
                if arr[temp[i]] > arr[temp[j]]:
                    temp[i], temp[j] = temp[j], temp[i]
        arr[beg], arr[temp[1]] = arr[temp[1]], arr[beg]

    pivot = arr[beg]

    i, j = beg, beg
    while j <= end:
        if arr[j] < pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
        j += 1
    
    arr[beg], arr[i] = arr[i], arr[beg]
    right_count = _count_qsort_cmp(arr, beg, i - 1, pivot_type)
    left_count = _count_qsort_cmp(arr, i + 1, end, pivot_type)
    return right_count + left_count + (end - beg)



if __name__ == "__main__":
    length = 1000
    arr = [random.randint(0, 9999) for _ in range(length)]

    print("Use first element")
    print(count_qsort_cmp(arr, "first"))
    print("Use last element")
    print(count_qsort_cmp(arr, "last"))
    print("Use median element")
    print(count_qsort_cmp(arr, "median"))
