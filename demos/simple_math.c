/*
 * Simple math demo
 * Shows symbolic execution on mathematical operations
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int solve_equation(int x) {
  // Solve: (x * 3 + 7) / 2 == 50
  int step1 = x * 3;
  int step2 = step1 + 7;
  int result = step2 / 2;

  if (result == 50) {
    return 1; // Correct solution
  }
  return 0;
}

int modular_check(int a, int b) {
  // Check if (a * 7 + b * 3) % 256 == 42
  int result = (a * 7 + b * 3) % 256;

  if (result == 42) {
    return 1;
  }
  return 0;
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    printf("Usage:\n");
    printf("  %s solve <x>      - Solve equation\n", argv[0]);
    printf("  %s modular <a> <b> - Check modular arithmetic\n", argv[0]);
    return 1;
  }

  if (strcmp(argv[1], "solve") == 0 && argc == 3) {
    int x = atoi(argv[2]);
    if (solve_equation(x)) {
      printf("Correct! x=%d solves the equation.\n", x);
      return 0;
    } else {
      printf("Wrong. x=%d doesn't solve the equation.\n", x);
      return 1;
    }
  } else if (strcmp(argv[1], "modular") == 0 && argc == 4) {
    int a = atoi(argv[2]);
    int b = atoi(argv[3]);
    if (modular_check(a, b)) {
      printf("Success! (a=%d, b=%d) satisfies the modular equation.\n", a, b);
      return 0;
    } else {
      printf("Failed. (a=%d, b=%d) doesn't work.\n", a, b);
      return 1;
    }
  } else {
    printf("Invalid arguments\n");
    return 1;
  }
}