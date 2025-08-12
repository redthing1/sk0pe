/*
 * Simple password checker demo
 * Tests if input matches a hardcoded password using XOR obfuscation
 */
#include <stdio.h>
#include <string.h>
#include <stdint.h>

int check_password(const char* input) {
    // Expected: "FLAG" XORed with 0x42
    uint8_t expected[] = {0x04, 0x0E, 0x03, 0x05}; // 'F'^0x42, 'L'^0x42, 'A'^0x42, 'G'^0x42
    
    if (strlen(input) != 4) {
        return 0;
    }
    
    for (int i = 0; i < 4; i++) {
        uint8_t xored = input[i] ^ 0x42;
        if (xored != expected[i]) {
            return 0;
        }
    }
    
    return 1;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <password>\n", argv[0]);
        return 1;
    }
    
    if (check_password(argv[1])) {
        printf("Correct password!\n");
        return 0;
    } else {
        printf("Wrong password!\n");
        return 1;
    }
}