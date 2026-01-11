#include <stdio.h>

int main() {
    int x, y, z;
    
    printf("Enter three integers: ");
    scanf("%d %d %d", &x, &y, &z);
    
    if (x > 10) {
        if (y < 5) {
            if (z == 0) {
                printf("Path A: x > 10, y < 5, z == 0\n");
            } else {
                printf("Path B: x > 10, y < 5, z != 0\n");
            }
        } else {
            if (z > 100) {
                printf("Path C: x > 10, y >= 5, z > 100\n");
            } else {
                printf("Path D: x > 10, y >= 5, z <= 100\n");
            }
        }
    } else {
        if (y == 0) {
            printf("Path E: x <= 10, y == 0\n");
        } else {
            if (z < 0) {
                printf("Path F: x <= 10, y != 0, z < 0\n");
            } else {
                printf("Path G: x <= 10, y != 0, z >= 0\n");
            }
        }
    }
    
    return 0;
}