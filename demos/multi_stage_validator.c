#include <stdio.h>
#include <string.h>
#include <stdint.h>

#define KEY_LENGTH 16
#define SUCCESS 1337
#define FAILURE -1

int validate_stage1_checksum(const uint8_t* key) {
  uint32_t sum = 0;
  for (int i = 0; i < KEY_LENGTH; i++) {
    sum += key[i];
  }

  if ((sum & 0xFF) == 0x42) {
    return 1;
  }
  return 0;
}

int validate_stage2_pattern(const uint8_t* key) {
  for (int i = 0; i < 8; i++) {
    uint8_t expected = (i % 2 == 0) ? 0xAA : 0x55;
    if ((key[i] & 0xF0) != (expected & 0xF0)) {
      return 0;
    }
  }
  return 1;
}

int validate_stage3_state_machine(const uint8_t* key) {
  int state = 0;
  const int target_state = 7;

  for (int i = 8; i < 12; i++) {
    uint8_t nibble = key[i] & 0x0F;

    switch (state) {
    case 0:
      if (nibble == 0x3) {
        state = 1;
      } else if (nibble == 0x7) {
        state = 2;
      } else {
        state = 0;
      }
      break;
    case 1:
      if (nibble == 0x5) {
        state = 3;
      } else if (nibble == 0x3) {
        state = 1;
      } else {
        state = 0;
      }
      break;
    case 2:
      if (nibble == 0x9) {
        state = 4;
      } else {
        state = 0;
      }
      break;
    case 3:
      if (nibble == 0x7) {
        state = 5;
      } else if (nibble == 0x5) {
        state = 3;
      } else {
        state = 1;
      }
      break;
    case 4:
      if (nibble == 0xB) {
        state = 6;
      } else {
        state = 2;
      }
      break;
    case 5:
      if (nibble == 0x9) {
        state = target_state;
      } else {
        state = 3;
      }
      break;
    case 6:
      if (nibble == 0xD) {
        state = target_state;
      } else {
        state = 4;
      }
      break;
    default:
      break;
    }
  }

  return (state == target_state) ? 1 : 0;
}

int validate_stage4_nested_conditions(const uint8_t* key) {
  uint8_t a = key[12];
  uint8_t b = key[13];
  uint8_t c = key[14];

  if (a > 0x50) {
    if (b < 0x30) {
      if ((c ^ a) == 0x42) {
        return 1;
      }
    } else if (b > 0x70) {
      if ((a + b) == 0xE0) {
        return 1;
      }
    }
  } else if (a < 0x20) {
    if ((b & 0x0F) == 0x0F) {
      if (c == (a << 2)) {
        return 1;
      }
    }
  } else {
    if (((a ^ b) & c) == 0x28) {
      return 1;
    }
  }

  return 0;
}

int validate_stage5_loop_counter(const uint8_t* key) {
  uint8_t target = key[15];
  int counter = 0;

  for (int i = 0; i < target; i++) {
    counter++;
    if (counter > 100) {
      return 0;
    }
  }

  if (counter == 42) {
    return 1;
  }

  return 0;
}

int validate_key(const char* input) {
  if (strlen(input) != KEY_LENGTH) {
    printf("Invalid key length!\n");
    return FAILURE;
  }

  const uint8_t* key = (const uint8_t*) input;

  printf("Stage 1 - Checksum validation...\n");
  if (!validate_stage1_checksum(key)) {
    printf("  [FAILED] Checksum incorrect\n");
    return FAILURE;
  }
  printf("  [PASSED]\n");

  printf("Stage 2 - Pattern matching...\n");
  if (!validate_stage2_pattern(key)) {
    printf("  [FAILED] Pattern mismatch\n");
    return FAILURE;
  }
  printf("  [PASSED]\n");

  printf("Stage 3 - State machine...\n");
  if (!validate_stage3_state_machine(key)) {
    printf("  [FAILED] Target state not reached\n");
    return FAILURE;
  }
  printf("  [PASSED]\n");

  printf("Stage 4 - Nested conditions...\n");
  if (!validate_stage4_nested_conditions(key)) {
    printf("  [FAILED] Conditions not satisfied\n");
    return FAILURE;
  }
  printf("  [PASSED]\n");

  printf("Stage 5 - Loop counter...\n");
  if (!validate_stage5_loop_counter(key)) {
    printf("  [FAILED] Counter mismatch\n");
    return FAILURE;
  }
  printf("  [PASSED]\n");

  printf("\n*** ALL STAGES PASSED! Valid key found! ***\n");
  return SUCCESS;
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <16-character-key>\n", argv[0]);
    return 1;
  }

  int result = validate_key(argv[1]);

  if (result == SUCCESS) {
    printf("Success! Return code: %d\n", result);
    return 0;
  } else {
    printf("Failed! Return code: %d\n", result);
    return 1;
  }
}