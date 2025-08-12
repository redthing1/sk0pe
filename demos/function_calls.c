/*
 * function_calls.c - test program for emulated function calling
 * demonstrates various calling conventions and argument passing
 */

#include <stdint.h>

// simple function with no arguments
int no_args() { return 42; }

// function with one argument
int one_arg(int x) { return x * 2; }

// function with two arguments
int two_args(int a, int b) { return a + b; }

// function with multiple arguments (tests register vs stack passing)
int many_args(int a, int b, int c, int d, int e, int f, int g, int h) { return a + b + c + d + e + f + g + h; }

// function that modifies memory
int memory_test(int* ptr, int value) {
  int old = *ptr;
  *ptr = value;
  return old;
}

// recursive function
int factorial(int n) {
  if (n <= 1) {
    return 1;
  }
  return n * factorial(n - 1);
}

// function with different return types
uint64_t return_64bit(uint32_t low, uint32_t high) { return ((uint64_t) high << 32) | low; }

// function that uses local arrays
int array_sum(int* arr, int len) {
  int sum = 0;
  for (int i = 0; i < len; i++) {
    sum += arr[i];
  }
  return sum;
}

// nested function calls
int nested_calc(int x) { return two_args(one_arg(x), no_args()); }

// function with side effects (global state)
static int counter = 0;

int increment_counter() { return ++counter; }

int get_counter() { return counter; }

// global variable test - proper read/write
static int global_data = 100;

int read_global() { return global_data; }

void write_global(int value) { global_data = value; }

int modify_global(int delta) {
  global_data += delta;
  return global_data;
}

// structure passing tests
typedef struct {
  int x;
  int y;
} Point;

typedef struct {
  int a;
  int b;
  int c;
  int d;
} LargeStruct;

// pass struct by value
int sum_point(Point p) { return p.x + p.y; }

// return struct by value
Point make_point(int x, int y) {
  Point p;
  p.x = x;
  p.y = y;
  return p;
}

// pass large struct (tests stack passing)
int sum_large_struct(LargeStruct s) { return s.a + s.b + s.c + s.d; }

// pass struct by pointer
void modify_point(Point* p, int dx, int dy) {
  p->x += dx;
  p->y += dy;
}

// variadic function
#include <stdarg.h>
int sum_variadic(int count, ...) {
  va_list args;
  va_start(args, count);
  int sum = 0;
  for (int i = 0; i < count; i++) {
    sum += va_arg(args, int);
  }
  va_end(args);
  return sum;
}

// function pointer test
typedef int (*binary_op)(int, int);

int apply_operation(binary_op op, int a, int b) { return op(a, b); }

// string operations
int string_length(const char* str) {
  int len = 0;
  while (str[len] != '\0') {
    len++;
  }
  return len;
}

// buffer operations
void fill_buffer(char* buf, int size, char value) {
  for (int i = 0; i < size; i++) {
    buf[i] = value;
  }
}

// edge cases
int edge_case_zero_div(int a, int b) {
  if (b == 0) {
    return -1;
  }
  return a / b;
}

uint64_t edge_case_overflow(uint32_t a, uint32_t b) { return (uint64_t) a * (uint64_t) b; }

// main function for testing
int main() {
  // prevent optimization
  volatile int result = 0;

  result += no_args();
  result += one_arg(21);
  result += two_args(10, 20);
  result += many_args(1, 2, 3, 4, 5, 6, 7, 8);
  result += factorial(5);
  result += nested_calc(10);

  return result;
}