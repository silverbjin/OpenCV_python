#0301_immutable.py
print('-'*50)
print('*'*20, 'immutable' ,'*'*20)
x = 10
print(id(x), ": id(x)")
x += 1
print( id(x), ": id(x += 1)",)

print('-'*50)
print('*'*20, 'mutable' ,'*'*20)
def modify_list(x):
    print(id(x), x, ": inside before")
    x.append(100)        # 가변 객체인 리스트에 요소를 추가
    print(id(x), x, ": inside after")

def reassign_list(x):
    print(id(x), x, ": inside before")
    x = ["new"]          # 재할당: 완전히 새로운 객체를 가리키도록 함
    print(id(x), x, ": inside after")

lst = [10, 20, 30]
print(id(lst), lst, ": before call")
modify_list(lst)
print(id(lst), lst, ": after modify_list")

print("--- reassign test ---")
print(id(lst), lst, ": before call")
reassign_list(lst)
print(id(lst), lst, ": after reassign_list")
