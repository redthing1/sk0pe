/*
 * Array bounds demo
 * Demonstrates array access patterns for symbolic execution
 */
#include <stdio.h>
#include <string.h>

int process_array(const char* input) {
  int lookup_table[10] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
  int sum = 0;

  // Process input string
  for (int i = 0; i < strlen(input) && i < 5; i++) {
    // Convert char to index (0-9)
    int index = input[i] - '0';

    // Bounds check
    if (index >= 0 && index < 10) {
      sum += lookup_table[index];
    } else {
      return -1; // Invalid input
    }
  }

  // Check for magic sum
  if (sum == 150) { // e.g., "12345" -> 10+20+30+40+50 = 150
    return 1337;    // Success!
  }

  return sum;
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <digits>\n", argv[0]);
    printf("Input should be a string of digits (0-9)\n");
    return 1;
  }

  int result = process_array(argv[1]);

  if (result == -1) {
    printf("Invalid input! Use only digits 0-9\n");
    return 1;
  } else if (result == 1337) {
    printf("Magic sum found! You win!\n");
    return 0;
  } else {
    printf("Sum: %d (try to get 150)\n", result);
    return 1;
  }
}