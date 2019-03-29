import math
t=input();
space=len(t)-2
j=0
s=''
for i in range(math.ceil(len(t)/2)):
    s=" "
    for k in range(i):
        s=s+" "
    s=s+t[i]
    for x in range(space):
        s=s+" "
    if i!=space+i+1:
        s=s+t[space+i+1]
    print(s)
    space=space-2

for i in range(math.ceil(len(t)/2)-1):
