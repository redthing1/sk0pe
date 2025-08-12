/*
 * Branching paths demo
 * Shows how different inputs lead to different execution paths
 */
#include <stdio.h>
#include <stdlib.h>

int complex_function(int x, int y) {
  int result = 0;

  // Multiple branching paths based on input
  if (x > 100) {
    if (y < 50) {
      result = x + y; // Path 1
    } else {
      result = x - y; // Path 2
    }
  } else if (x == 42) {
    if (y == 1337) {
      result = 0xDEADBEEF; // Secret path!
    } else {
      result = x * y; // Path 3
    }
  } else {
    result = x ^ y; // Path 4
  }

  return result;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <x> <y>\n", argv[0]);
    return 1;
  }

  int x = atoi(argv[1]);
  int y = atoi(argv[2]);

  int result = complex_function(x, y);

  printf("Result: 0x%x (%d)\n", result, result);

  // Check if secret path was found
  if (result == 0xDEADBEEF) {
    printf("You found the secret path!\n");
    return 0;
  }

  return 1;
}