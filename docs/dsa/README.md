# Data Structures, Algorithms, and Complexity

**Mức đã trao đổi:** **4/5**

### Cần nhớ

Từ tốt đến xấu:

`O(1) < O(log n) < O(n) < O(n log n) < O(n²) < O(n³) < O(2^n) < O(n!)`

### Interviewer có thể hỏi thêm

- What is Big-O notation?
  - Big-O notation is a way to describe the complexity of an algorithm as the input size n increases.

- What is the difference between time complexity and space complexity?
  - Time complexity = how the algorithm's running time increases as the input grows.

  - Space complexity = how the amount of memory the algorithm requires increases as the input grows.

- Why is binary search O(log n)?
  - Because with each loop, it reduces a half array, so compute step is log2(n)
  - Code :

    ```cpp
    #include <iostream>
    #include <vector>
    using namespace std;

    int binarySearch(const vector<int>& arr, int target) {
        int left = 0;
        int right = arr.size() - 1;
        while (left <= right) {
            int mid = left + (right-left)/2
            if (arr[mid] == target){
                return mid;
            }
            if (arr[mid] < target){
                left = mid + 1;
            } else {
                right = mid -1;
            }
        }
        return -1;
    }
    ```

- Why is Merge Sort O(n log n)?
  - Merge sort use divide arrray to binary tree. In this step, it divides the half array for each step, so the number of step will scale with the number of elements with O(logn). And then, it compares each elements with each other in sub array in same level, so in this step, big0 is O(n). So finally, bigO is O(n)*O(logn) = O(nlogn)

- What is the average complexity of QuickSort?
  - Quicksort use pivot to divide binary tree, so with each step, it use O(logn) compute. For each floor, it must devide for left and right. So it computes O(n). Finally, it uses O(nlogn) for all.

---

### 1.6 Heap / QuickSort / basic algorithms

**Mức đã trao đổi:** **3–4/5** — **chưa xác nhận chính xác**

### Interviewer có thể hỏi thêm

- What is a heap?
  - Heap is binary tree that features parent node is greater than to its chidlren.

- What is the difference between a min-heap and max-heap?
  - Min heap is binary tree that features parent node is smaller than to its children. And max heap is binary tree that features parent node is greater than to its children.

- What is heapify?
  - Heapify is generate features for binary tree meaning parent node is greater than chill nodel

- How does HeapSort work?
  - HeapSort use heapify to build binary tree that I mentioned. And it is call heapify for nodes that is not leaf node (foot node in tree). It means from n/2-1 to 0. Finally, when I have binary tree completely, I just swap a[i] with a[0] because with binary tree, a[0] is max node. And I keep to run i from head vector to tail vector

- How does QuickSort work?
  - It chooses late element to make pivot. And it use partition to move element that smaller than pivot to left pivot and greater than pivot to right pivot. And finally, it recursion from left and right pivot.

- What is the pivot?
  - It use pivot to divide array to binary tree.

- Why can QuickSort become O(n²)?
  - If quicksort use pivot high (late in array), the pivot may have ability max array or min array, not mid array, so it take a lot of time to choose pivot to mid array.

- What is the average complexity of QuickSort?
  - O(nlogn)

- QuickSort vs MergeSort: when would you use each one?
- Is QuickSort stable?
  - When I choose pivot that is near median array.

- Is MergeSort stable?
  - Almost stable O(nlogn)

---
