#include <stdio.h>

int main() {
    int x;
    
    printf("Enter a number: ");
    scanf("%d", &x);
    
    if (x > 10) {
        printf("Large: %d\n", x);
        return 1;
    } else if (x < 0) {
        printf("Negative: %d\n", x);
        return 2;
    } else {
        printf("Small: %d\n", x);
        return 0;
    }
}