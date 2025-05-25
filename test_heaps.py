from fibonacci_heap import FibonacciHeap
from hollow_heap import HollowHeap
import heapq
import time
import random
import heapq

times = [0,0,0]
number_of_elements = 1000000
number_of_additions = 500000
number_of_deletions = 800000
number_of_rounds = 3
counter = 0
elements = [(random.randint(1,10000000),i) for i in range(10000000)]

start_time_heapq = time.time()
heap = []
#initialization of the heap
for i in range(number_of_elements):
	heapq.heappush(heap, elements[i])
counter = number_of_elements
for round in range(number_of_rounds):
	for i in range(number_of_additions):
		heapq.heappush(heap, elements[counter+i]) 
	counter += number_of_additions
	for i in range(number_of_deletions):
		key, element = heapq.heappop(heap)
end_time_heapq = time.time()
times[0] = end_time_heapq-start_time_heapq
print(times[0])
#============================================
counter = 0
start_time_fib = time.time()
fib_heap = FibonacciHeap()
#initialization of the heap
for i in range(number_of_elements):
	fib_heap.insert(elements[i][0], elements[i][1])
counter = number_of_elements
for round in range(number_of_rounds):
	for i in range(number_of_additions):
		fib_heap.insert(elements[counter+i][0], elements[counter+i][1])
	counter += number_of_additions
	for i in range(number_of_deletions):
		element_node = fib_heap.extract_min()
end_time_fib = time.time()
times[1] = end_time_fib-start_time_fib
print(times[1])
#===================================================
counter = 0
start_time_hollow = time.time()
hollow_heap = HollowHeap()
#initialization of the heap
for i in range(number_of_elements):
	hollow_heap.push(elements[i][0], elements[i][1])
counter = number_of_elements
for round in range(number_of_rounds):
	for i in range(number_of_additions):
		hollow_heap.push(elements[counter+i][0], elements[counter+i][1])
	counter += number_of_additions
	for i in range(number_of_deletions):
		element_node = hollow_heap.find_min()
		hollow_heap.delete_min()
end_time_hollow = time.time()
times[2] = end_time_hollow-start_time_hollow

print(times) # experiments show that heapq is still better than the 2 other implementations
# times = [17.82, 170.36, 103.55]
#            ^
#            |
#   the time for heapq is
#   smaller than the rest
