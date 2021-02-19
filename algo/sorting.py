import random
import functools
import time


def _mergesort(arr, aux, beg, end):
    if beg == end:
        return

    mid = beg + (end - beg) // 2
    _mergesort(arr, aux, beg, mid)
    _mergesort(arr, aux, mid + 1, end)

    aux[beg:end + 1] = arr[beg: end + 1]
    i, j, k = beg, mid + 1, beg
    while k <= end:
        if i == mid + 1:
            arr[k: end+1] = aux[j: end + 1]
            break
        if j == end + 1:
            arr[k: end+1] = aux[i: mid + 1]
            break

        if aux[i] > aux[j]:
            arr[k] = aux[j]
            j += 1
        else:
            arr[k] = aux[i]
            i += 1
        k += 1


def mergesort(arr):
    _mergesort(arr, [0]*len(arr), 0, len(arr) - 1)
    return arr




"""
Stanford and Princeton refers to the implementations taught in stanford/princeton algo respetively

I believe the stanford implementation will incur n^2 running time when sorting arr of the same vals
"""

def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tic = time.time()
        for _ in range(100):
            result = func(*args, **kwargs)
        toc = time.time()
        print(f"[{func.__name__}]time: {tic - toc}, avg time: {(toc - tic) / 100} ")
        return result
    return wrapper


@timeit
def quicksort(arr, vers = "stanford", inplace = False):
    assert vers in {"stanford", "princeton"}
    if not inplace:
        arr = list(arr)

    random.shuffle(arr)
    if vers == "stanford":
        _qsort_stan(arr, 0, len(arr) - 1)
    else:
        _qsort_prin(arr, 0, len(arr) - 1)

    return arr

def _qsort_stan(arr, beg, end):
    if beg >= end:
        return
    pivot = arr[beg]
    i, j = beg, beg + 1
    while j <= end:
        if arr[j] < pivot:
            i += 1
            arr[j], arr[i] = arr[i], arr[j]
        j += 1
    arr[beg], arr[i] = arr[i], arr[beg]
    _qsort_stan(arr, beg, i - 1)
    _qsort_stan(arr, i + 1, end)



def _qsort_prin(arr, beg, end):
    if beg >= end:
        return

    pivot = arr[beg]
    i, j = beg, end + 1 # as we need to use do while loop
    while True:
        # need to
        # _________________________
        # |v|____<=v|__?__|>=v____|
        # equal sign for both partitions
        while True: # need to use do while loop, else the algo may get stuck as break condition can be met without increment/decrecment
            i += 1
            if i == end or arr[i] >= pivot: # do use equal sign here, as we need to stop on equal keys, else may be n^2
                break
        
        # note: both need to stop on equal key, if one does and the other doesn't, elements equal to the pivot
        # will be on the same side, if both do not, n^2 time for arrays of the same element
        while True: 
            j -= 1
            if j == beg or arr[j] <= pivot:  # same as above
                break
        
        if i >= j: # need to have equal sign here, else the function may be stuck in the infinite loop
            break

        arr[i], arr[j] = arr[j], arr[i]
    
    # because the way it is formulated, i, j will be one past the boundary
    # j ends up to be the left most 
    arr[beg], arr[j] = arr[j], arr[beg]

    _qsort_prin(arr, beg, j - 1)
    _qsort_prin(arr, j + 1, end)



def counting_sort(arr, inplace = False):
    """Use counting sort to sort a list of extended ASCII charaters."""
    R = 256
    counts = [0] * (R + 1)
    for c in arr:
        counts[ord(c) + 1] += 1
    for i in range(R):
        counts[i+1] += counts[i]
    temp_arr = list(arr)
    if not inplace:
        arr = [None] * len(arr)
    for c in temp_arr:
        arr[counts[ord(c)]] = c
        counts[ord(c)] += 1

    if not inplace:
        return arr



def LSD_radix_sort(arr, inplace = False):
    """Least significant digit first radix sort for string of extended ASCII charaters
    
    The strings in the list must be of the same length.
    """
    R = 256
    for string in arr:
        assert len(string) == len(arr[0]), \
            "Please ensure all the strings in the array have the same length"
    temp_arr = list(arr)
    if not inplace:
        arr = list(arr)
    for i in range(len(arr[0])-1, -1, -1):
        counts = [0] * (R + 1)
        for string in temp_arr:
            counts[ord(string[i]) + 1] += 1
        for j in range(256):
            counts[j+1] += counts[j]
        for string in temp_arr:
            arr[counts[ord(string[i])]] = string
            counts[ord(string[i])] += 1
        temp_arr = list(arr)
    if not inplace:
        return arr


def _msd_radix_sort(arr, temp_arr, beg, end, idx):
    R = 256
    if beg >= end:
        return
    counts = [0] * (R + 2) # account for string with length less than idx + 1
    for i in range(beg, end + 1):
        word = arr[i]
        try:
            counts[ord(word[idx]) + 2] += 1
        except IndexError:
            counts[1] += 1
    for i in range(R + 1):
        counts[i+1] += counts[i]
    temp_arr[beg:end+1] = arr[beg:end+1]
    for i in range(beg, end + 1):
        word = temp_arr[i]
        try:
            arr[beg + counts[ord(word[idx])+1]] = word
            counts[ord(word[idx])+1] += 1
        except IndexError:
            arr[beg + counts[0]] = word
            counts[0] += 1

    for i in range(0, R):
        # ignoring the string without sufficient length
        _msd_radix_sort(arr, temp_arr, beg + counts[i], beg + counts[i+1] - 1, idx + 1)
    


def MSD_radix_sort(arr, inplace=False):
    """Most significant digit first radix sort for string of extended ASCII charaters."""
    temp_arr = [None] * len(arr)
    if not inplace:
        arr = list(arr)
    _msd_radix_sort(arr, temp_arr, 0, len(arr) - 1, 0)
    if not inplace:
        return arr



def _str_quicksort(arr, beg, end, idx):
    if beg >= end:
        return
    lt, gt = beg, end
    try:
        pivot = ord(arr[beg][idx])
    except IndexError:
        pivot = -1
    i = beg + 1
    while i <= gt:
        try:
            t = ord(arr[i][idx])
        except IndexError:
            t = -1
        
        if t < pivot:
            arr[i], arr[lt] = arr[lt], arr[i]
            lt += 1
            i += 1
        elif t > pivot:
            arr[i], arr[gt] = arr[gt], arr[i]
            gt -= 1
            # note, we do not increment i
            # as we have not check the value of what i is currently pointing
        else:
            i += 1
    
    _str_quicksort(arr, beg, lt - 1, idx)
    if pivot >= 0:
        # only if the current charachters are the same
        # we add 1 to the idx to continue comparing the next characters
        _str_quicksort(arr, lt, gt, idx + 1)
    _str_quicksort(arr, gt + 1, end, idx)


def str_quicksort(arr, inplace=False):
    """3-way string quick sort."""
    if not inplace:
        arr = list(arr)
    _str_quicksort(arr, 0, len(arr) - 1, 0)
    if not inplace:
        return arr



if __name__ == "__main__":
    # num = 9999
    # arr = [random.randint(0, 9999) for _ in range(num)]
    # assert mergesort(arr) == sorted(arr)



    # arr_length = 200
    # arr = [random.randint(0, 999) for _ in range(arr_length)]
    # # arr = [0] * length
    # print("Stanford implementation")
    # stan_result = quicksort(arr, "stanford")
    # print("Princeton implementation")
    # prin_result = quicksort(arr, "princeton")
    # ans = sorted(arr)

    # if stan_result != ans:
    #     print("stanford qsort is wrong")
    
    # if prin_result != ans:
    #     print("princeton qsort is wrong")

    # # counting sort
    # print("counting sort")
    # n = 200
    # arr = [chr(random.randint(0, 255)) for _ in range(n)]
    # arr_1 = list(arr)
    # assert counting_sort(arr) == sorted(arr)
    # counting_sort(arr_1, inplace=True)
    # assert arr_1 == sorted(arr)

    # # lsd radix sort
    # print("LSD radix sort")
    # n = 200
    # str_len = 8
    # arr = ["".join([chr(random.randint(0, 255)) for _ in range(str_len)]) for _ in range(n)]
    # arr_1 = list(arr)
    # assert LSD_radix_sort(arr) == sorted(arr)
    # LSD_radix_sort(arr_1, inplace=True)
    # assert arr_1 == sorted(arr)
    
    # # msd radix sort
    # print("MSD radix sort")
    # n = 200
    # arr = []
    # for _ in range(n):
    #     str_len = random.randint(0, 8)
    #     string = "".join([chr(random.randint(0, 255)) for _ in range(str_len)])
    #     arr.append(string)
    # assert MSD_radix_sort(arr) == sorted(arr)

    # quick string sort
    print("3-way string quick sort")
    n = 100
    arr = []
    for _ in range(n):
        str_len = random.randint(0, 8)
        string = "".join([chr(random.randint(97, 122)) for _ in range(str_len)])
        arr.append(string)
    assert str_quicksort(arr) == sorted(arr)
