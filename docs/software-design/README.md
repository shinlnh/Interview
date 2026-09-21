# Software Design: Patterns and OOP

**Mức đã trao đổi:** **3/5**

### Interviewer có thể hỏi thêm

- What is a software design pattern?
  - A software design pattern is a reusable solution to a common software design problem.

- Why do we use design patterns?
  - We use design patterns because they provide proven solutions to common software design problems. They help make code more maintainable, reusable, flexible, and easier to extend.

- Three pattern i learn :
  - Strategy pattern
  - Factory pattern
  - Observer pattern

- What is the Strategy pattern?
  - If you have a task which has many the way to solve the problem. You should divide those solutions within a box. You don't need to know what's inside; whenever you need a specific solution, you simply call it by name from the outside to use it. That is the Strategy pattern in OOP at work. It is using Polymorphism in OOP
  - For example, I have three way to solve sort issue.

    ```cpp
    class SortStrategy {
    public:
        virtual void sort() = 0;
    };

    class BubbleSort : public SortStrategy {
    public:
        void sort() override {
            cout << "Bubble Sort";
        }
    };

    class QuickSort : public SortStrategy {
    public:
        void sort() override {
            cout << "Quick Sort";
        }
    };

    SortStrategy* s;

    s = new BubbleSort();
    s->sort();              // Bubble Sort

    s = new QuickSort();
    s->sort();
    ```

- What is the Factory pattern?
  - With strategy pattern, you must call object way if you want use that object to solve. However, Factory pattern improve compare with Strategy. If you want what object, Factory will make this object for you. You don't create object handmade.
  - For example :

    ```cpp
    #include <iostream>
    #include <string>

    using namespace std;

    class SortStrategy {
    public:
        virtual void sort() = 0;
    };

    class BubbleSort : public SortStrategy {
    public:
        void sort() override {
            cout << "Bubble Sort" << endl;
        }
    };

    class QuickSort : public SortStrategy {
    public:
        void sort() override {
            cout << "Quick Sort" << endl;
        }
    };

    class SortFactory {
    public:
        SortStrategy* createSort(string type) {

            if (type == "bubble") {
                return new BubbleSort();
            }

            if (type == "quick") {
                return new QuickSort();
            }

            return nullptr;
        }
    };

    int main() {

        SortFactory factory;
        SortStrategy* s;

        s = factory.createSort("bubble");
        s->sort();

        s = factory.createSort("quick");
        s->sort();

        return 0;
    }
    ```

- What is the Observer pattern?
  - Ìf you want when a object has been changed, all object will be received notify from it.
  - For example :

    ```cpp
    #include <iostream>
    #include <vector>

    using namespace std;


    // Observer
    class SortObserver {
    public:
        virtual void update(int number) = 0;
    };


    // Observer 1
    class QuickSort : public SortObserver {
    public:
        void update(int number) override {
            cout << "QuickSort nhan so moi: "
                 << number << endl;
        }
    };


    // Observer 2
    class HeapSort : public SortObserver {
    public:
        void update(int number) override {
            cout << "HeapSort nhan so moi: "
                 << number << endl;
        }
    };


    // Subject
    class NumberSource {
    private:
        vector<SortObserver*> observers;

    public:
        void subscribe(SortObserver* observer) {
            observers.push_back(observer);
        }

        void addNumber(int number) {

            cout << "So moi: " << number << endl;

            // Thông báo cho tất cả Observer
            for (SortObserver* observer : observers) {
                observer->update(number);
            }
        }
    };


    int main() {

        NumberSource source;

        QuickSort quick;
        HeapSort heap;

        // Hai thuật toán đăng ký theo dõi source
        source.subscribe(&quick);
        source.subscribe(&heap);

        // Source nhận số mới
        source.addNumber(10);

        return 0;
    }
    ```

- Give an example of a design pattern in a robotics or ML pipeline.
- When can Singleton become a bad design?
- What is dependency injection?
- How would you design interchangeable detector or tracker modules?

### Robotics example

Một pipeline kiểu:

`Detector interface → YOLO / SSD / StreamPETR implementation`

rất dễ liên hệ với **Strategy / Factory**.

---

### 1.8 Object-Oriented Programming

**Mức đã trao đổi:** **4/5**

### Interviewer có thể hỏi thêm

- What are the four main principles of OOP?
- What is encapsulation?
- What is inheritance?
- What is polymorphism?
- What is abstraction?
- What is the difference between inheritance and composition?
- What is a virtual function in C++?
- What is a pure virtual function?
- What is an abstract class?
- What is function overloading vs overriding?
- What is a constructor/destructor?
- What is RAII in C++?
- What is a smart pointer?
- unique_ptr vs shared_ptr?
- Why is composition often preferred over inheritance?
- How would you design a tracker interface supporting CSRT, KCF and a Siamese tracker?

---
