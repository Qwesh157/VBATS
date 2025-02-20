#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <fstream>

#define random(x) (rand() % (x))

int main(int argc, char *argv[])
{

	if (argc < 3)
	{
		printf("Usage: please input two integers\n");
		printf("The first one represents the largest matrix size (M, N)\n");
		printf("The second one represents the K\n");
		exit(EXIT_FAILURE);
	}

	std::fstream fs;
	fs.open("../data/input");
	if (!fs.is_open())
	{
		printf("Error opening input\n");
		exit(EXIT_FAILURE);
	}

	int e = atoi(argv[1]);
	int log_mn = 0;

	while (e >= 16)
	{
		e = e >> 1;
		log_mn++;
	}

	e = atoi(argv[2]);
	int log_k = 0;

	while (e >= 16)
	{
		e = e >> 1;
		log_k++;
	}
	// int K = atoi(argv[2]);
	// read matrix config
	for (int i = 0; i < 256; ++i)
	{
		int M = 16 << random(log_mn);
		int N = 16 << random(log_mn);
		int K = log_k > 2 ? 16 << (random(log_k - 2) + 2) : 16 << random(log_k);
		fs << M << ' ' << N << ' ' << K << std::endl;
	}

	return EXIT_SUCCESS;
}
