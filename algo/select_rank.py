import random

# deteministic select
def deterministic_select(arr, idx):
    arr = list(arr)
    # no shuffle should be needed
    val =  _dselect(arr, 0, len(arr) - 1, idx)
    return val


def _dselect(arr, beg, end, idx):
    if beg == end:
        assert beg == idx
        return arr[beg]

    i = 0
    medians = []
    while i < len(arr):
        temp = sorted(arr[i: i+5])
        mid = round(len(temp) / 2)
        medians.append(temp[mid])
        i += 5

    med = _dselect(medians, 0, len(medians) - 1, round(len(medians)/2))

    for med_idx in range(beg, end + 1):
        if arr[med_idx] == med:
            break

    arr[med_idx], arr[beg] = arr[beg], arr[med_idx]

    pivot = arr[beg]
    i, j = beg, end + 1
    while True:
        while True:
            i += 1
            if i == end or arr[i] >= pivot:
                break
        while True:
            j -= 1
            if j == beg or arr[j] <= pivot:
                break
        
        if i >= j:
            break
        arr[i], arr[j] = arr[j], arr[i]
    
    arr[beg], arr[j] = arr[j], arr[beg]

    if j == idx:
        return pivot
    if j < idx:
        return _dselect(arr, j + 1, end, idx)
    if j > idx:
        return _dselect(arr, beg, j - 1, idx)




# random select
def random_select(arr, idx):
    arr = list(arr)
    random.shuffle(arr)
    assert idx < len(arr)
    return _random_select(arr, 0, len(arr) - 1, idx)


def _random_select(arr, beg, end, idx):
    if beg == end:
        assert beg == idx
        return arr[beg]
    pivot = arr[beg]
    i, j = beg, end + 1
    while True:
        while True:
            i += 1
            if i == end or arr[i] >= pivot:
                break
        while True:
            j -= 1
            if j == beg or arr[j] <= pivot:
                break
        if i >= j:
            break

        arr[i], arr[j] = arr[j], arr[i]
    arr[j], arr[beg] = arr[beg], arr[j]

    if j == idx:
        return pivot
    if j < idx:
        return _random_select(arr, j + 1, end, idx)
    else:
        return _random_select(arr, beg, j - 1, idx)


if __name__ == "__main__":
    import time
    random.seed(777)

    # test deteministic select
    arr_length = 4000
    arr = [random.randint(0, arr_length) for _ in range(arr_length)]
    test_length = 1000
    tic = time.time()
    for i in range(test_length):
        idx = random.randint(0, arr_length - 1)
        # sorted_arr = sorted(arr)
        # ans = sorted_arr[idx]
        result = deterministic_select(arr, idx)
        # if i < 10:
        #     print(ans, " vs ", result)
        # assert ans == result
    print(arr_length, time.time() - tic)


    # test random select
    arr_length = 4000
    arr = [random.randint(0, 999) for _ in range(arr_length)]
    test_length = 1000
    tic = time.time()
    for i in range(test_length):
        idx = random.randint(0, arr_length - 1)
        # sorted_arr = sorted(arr)
        # ans = sorted_arr[idx]
        result = random_select(arr, idx)
        # assert ans == result
        # if i < 10:
        #     print(ans, " vs ", result)
    print(arr_length, time.time() - tic)
