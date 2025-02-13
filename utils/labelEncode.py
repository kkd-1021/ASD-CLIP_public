


def labelEncode(lst):
    # 1+5+2+1+1(占位）
    id=lst[-1]
    lst=lst[:-1]
    lst.append(5)
    num = 0
    base = 11  # 选择一个合适的基数，只要保证结果不溢出即可，可按需调整
    for i, val in enumerate(lst):
        num += val * (base ** i)
    num*=1000
    num+=id
    return int(num)


def labelDecode(inp):

    num=inp
    id = num % 1000
    num //= 1000
    result = []
    base = 11
    while num > 0:
        remainder = num % base
        result.append(remainder)
        num //= base

    result = result[:-1]
    result.append(id)
    return result



def Feat2Idx(inp,orgCss=False):
    vector=inp
    if orgCss:
        for idx in range(6, 8):
            if vector[idx] <= 3:
                vector[idx] = 0
            elif vector[idx] <= 6:
                vector[idx] = 1
            else:
                vector[idx] = 2
    #assert vector.shape[0] == 8
    value_1, value_2, value_3, value_4, value_5, value_6, value_7, value_8 = vector

    base_1 = 2
    base_2 = 2
    base_3 = 2
    base_4 = 2
    base_5 = 2
    base_6 = 2
    base_7 = 3
    base_8 = 3

    decimal_num = value_1 * (base_2 * base_3 * base_4 * base_5 * base_6 * base_7 * base_8) + \
                  value_2 * (base_3 * base_4 * base_5 * base_6 * base_7 * base_8) + \
                  value_3 * (base_4 * base_5 * base_6 * base_7 * base_8) + \
                  value_4 * (base_5 * base_6 * base_7 * base_8) + \
                  value_5 * (base_6 * base_7 * base_8) + \
                  value_6 * (base_7 * base_8) + \
                  value_7 * base_8 + \
                  value_8

    assert decimal_num<base_1*base_2*base_3*base_4*base_5*base_6*base_7*base_8
    return decimal_num


def Idx2Feat(num):
    base_1 = 2
    base_2 = 2
    base_3 = 2
    base_4 = 2
    base_5 = 2
    base_6 = 2
    base_7 = 3
    base_8 = 3

    max_product_1 = base_2 * base_3 * base_4 * base_5 * base_6 * base_7 * base_8
    value_1 = num // max_product_1
    num %= max_product_1

    max_product_2 = base_3 * base_4 * base_5 * base_6 * base_7 * base_8
    value_2 = num // max_product_2
    num %= max_product_2

    max_product_3 = base_4 * base_5 * base_6 * base_7 * base_8
    value_3 = num // max_product_3
    num %= max_product_3

    max_product_4 = base_5 * base_6 * base_7 * base_8
    value_4 = num // max_product_4
    num %= max_product_4

    max_product_5 = base_6 * base_7 * base_8
    value_5 = num // max_product_5
    num %= max_product_5

    max_product_6 = base_7 * base_8
    value_6 = num // max_product_6
    num %= max_product_6

    value_7 = num // base_8
    num %= base_8

    value_8 = num

    return [value_1, value_2, value_3, value_4, value_5, value_6, value_7, value_8]