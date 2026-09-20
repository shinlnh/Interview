# Data Structures, Algorithms, and Complexity

**Mức đã trao đổi:** **4/5**

### Cần nhớ

Từ tốt đến xấu:

`O(1) < O(log n) < O(n) < O(n log n) < O(n²) < O(n³) < O(2^n) < O(n!)`

### Interviewer có thể hỏi thêm

- What is Big-O notation?
  + Big-O notation is a way to describe the complexity of an algorithm as the input size n increases.
- What is the difference between time complexity and space complexity?
  + Time complexity = how the algorithm's running time increases as the input grows.

  + Space complexity = how the amount of memory the algorithm requires increases as the input grows.
- Why is binary search O(log n)?
  + Because with each loop, it reduces a half array, so compute step is log2(n)
  + Code :
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

- Why is Merge Sort O(n log n)?
- What is the average complexity of QuickSort?
- What is the worst-case complexity of QuickSort?
- Why can two O(n) algorithms have different runtime in practice?
- What is amortized complexity?
- What is the complexity of accessing an array element?
- What is the complexity of searching in a hash table?
- What is the complexity of inserting into a heap?
- What is the complexity of matrix multiplication?
- If an algorithm has two nested loops, is it always O(n²)?
- How would you reduce an O(n²) matching operation?

---

### 1.6 Heap / QuickSort / basic algorithms

**Mức đã trao đổi:** **3–4/5** — **chưa xác nhận chính xác**

### Interviewer có thể hỏi thêm

- What is a heap?
- What is the difference between a min-heap and max-heap?
- What is the complexity of heap insertion?
- What is heapify?
- How does HeapSort work?
- How does QuickSort work?
- What is the pivot?
- Why can QuickSort become O(n²)?
- What is the average complexity of QuickSort?
- QuickSort vs MergeSort: when would you use each one?
- Is QuickSort stable?
- Is MergeSort stable?
- What is an in-place sorting algorithm?

---
