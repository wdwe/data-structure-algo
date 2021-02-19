"""The functions below only work if 
    all inputs and results are positive

"""

def add(num_1, num_2):
    num_1, num_2 = list(reversed(num_1)), list(reversed(num_2))
    ans = []
    carry = 0
    for i in range(max(len(num_1), len(num_2))):
        a = num_1[i] if i < len(num_1) else 0
        b = num_2[i] if i < len(num_2) else 0
        result = a + b + carry
        d = result % 10
        carry = result // 10
        ans.append(d)
    
    if carry == 1:
        ans.append(carry)
    return list(reversed(ans))

def minus(num_1, num_2):
    num_1, num_2 = list(reversed(num_1)), list(reversed(num_2))
    ans = []
    borrow = 0
    # here num_1 must be at least as long as num_2
    for i in range(len(num_1)):
        a = num_1[i] if i < len(num_1) else 0
        b = num_2[i] if i < len(num_2) else 0
        result = a - b - borrow
        if result < 0:
            d = result + 10
            borrow = 1
        else:
            borrow = 0
            d = result
        ans.append(d)
    assert borrow == 0

    return list(reversed(ans))



def mult_by_single_digit(num, digit):
    assert len(digit) == 1
    digit = digit[0]
    num = list(reversed(num))
    ans = []
    carry = 0
    for a in num:
        result = a * digit + carry
        d = result % 10
        carry = result // 10
        ans.append(d)
    if carry > 0:
        ans.append(carry)

    return list(reversed(ans))


def num2list(num):
    return [int(i) for i in list(str(num))]

def list2num(num_list):
    return int("".join([str(num) for num in num_list]))


def karatsuba_mult(num_1, num_2):
    assert isinstance(num_1, list) and isinstance(num_2, list)
    if len(num_1) == 1:
        return mult_by_single_digit(num_2, num_1)
    if len(num_2) == 1:
        return mult_by_single_digit(num_1, num_2)
    
    length = max(len(num_1), len(num_2))

    num_1 = [0] * (length - len(num_1)) + num_1
    num_2 = [0] * (length - len(num_2)) + num_2 

    split = length // 2
    a = num_1[:split]
    b = num_1[split:]
    c = num_2[:split]
    d = num_2[split:]

    ac = karatsuba_mult(a, c)
    bd = karatsuba_mult(b, d)
    a_b = add(a, b)
    c_d = add(c, d)
    a_bc_d = karatsuba_mult(a_b, c_d)
    ad_bc = minus(minus(a_bc_d, ac), bd)

    base = length - split

    return add(add(ac + 2 * base * [0], ad_bc + base * [0]), bd)


if __name__ == "__main__":
    import random
    for _ in range(5000):
        num1 = random.randint(0, 99999)
        num2 = random.randint(0, 99999)

        kara_result = list2num(karatsuba_mult(num2list(num1), num2list(num2)))
        result = num1 * num2
        assert kara_result == result
    
