import numpy as np
import json
import matplotlib.pyplot as plt
'''my_list = [1, 2, 3]

# Write the list to a file
# Read the list back into Python
read_list = []
with open('my_list.txt', 'r') as file:
    for line in file:
        read_list.append(int(line.strip()))
read_list1 = []
with open('my_list1.txt', 'r') as file:
    for line in file:
        read_list1.append(int(line.strip()))
read_list2 = []
with open('my_list2.txt', 'r') as file:
    for line in file:
        read_list2.append(int(line.strip()))
read_list3 = []
with open('my_list3.txt', 'r') as file:
    for line in file:
        read_list3.append(float(line.strip()))
print(len(read_list),len(read_list1),len(read_list2),len(read_list3))
add_list=[]
result=np.argsort(np.array(read_list3))[:1000]
for i in result:
    if (i in read_list1) or (i in read_list2):
        add_list.append(i)
print("finish",len(add_list))
print(read_list3.index(max(read_list3)))
if read_list3.index(max(read_list3)) in read_list1 or read_list3.index(max(read_list3)) in read_list2:
    print("true")'''
'''array = np.arange(10, 21)
result=np.argsort(array)[:5]
print(result)'''
'''array = np.arange(-10, 10)  # 创建一个包含随机正负数和零的数组
array[0]=-5
print(array)
new_score=array
non_zero_elements = new_score[new_score != 0]
non_zero_indices = np.where(new_score != 0)[0]
sorted_indices = np.argsort(-non_zero_elements)
result = non_zero_indices[sorted_indices][:10]
#print(array)
#result=np.argsort(array)[:10]

print(result)'''

with open('loss_values.json', 'r') as f:
    data = json.load(f)
    loss_max = data['loss_max_list']
    loss_true = data['loss_list']
x_indices = range(len(loss_max))
plt.plot(x_indices, loss_max, marker='o', label='Loss Group 1')
plt.plot(x_indices, loss_true, marker='x', label='Loss Group 2')
plt.title('Loss over Steps')
plt.xlabel('step')
plt.ylabel('Loss')
plt.xticks(range(0, len(loss_max), 20))
plt.legend()
plt.savefig('loss_plot.png')
