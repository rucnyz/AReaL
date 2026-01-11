#include <stdio.h>

int main() {
    int input;
    int result = 0;
    int counter = 0;
    
    printf("Enter an integer: ");
    scanf("%d", &input);
    
    // Simple loop with fixed bound
    for (int i = 0; i < 4; i++) {
        if (input > 10) {
            result += 2;
            counter++;
        } else {
            result += 1;
        }
        
        if (result > 5) {
            counter += 2;
        }
    }
    
    // Final conditional checks
    if (counter > 8) {
        printf("High counter: %d\n", counter);
        result *= 2;
    } else if (counter > 4) {
        printf("Medium counter: %d\n", counter);
        result += 5;
    } else {
        printf("Low counter: %d\n", counter);
    }
    
    printf("Final result: %d\n", result);
    
    return 0;
}