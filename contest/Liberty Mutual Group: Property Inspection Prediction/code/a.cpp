#include <cstdio>

int main() {
	int sum = 0;
	for (int i = 0; i < 1000000000; i++)
		sum += i;
	printf("%d\n", sum);
	return 0;
}
