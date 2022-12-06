#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>

void findHammingOne(std::multimap<std::vector<bool>, int> hashMap)
{
	for (auto it = hashMap.begin(); it != hashMap.end();) {
		for (auto i = 0; i < (*it).first.size(); ++i) {
			std::vector<bool> data = (*it).first;
			data[i].flip();
			auto range = hashMap.equal_range(data);
			for (auto& jt = range.first; jt != range.second; ++jt) {
				std::cout << "(" << (*it).second << ", " << (*jt).second << ")\n";
			}
		}
		it = hashMap.erase(it);
	}
}

std::multimap<std::vector<bool>, int> readFile(int& N, int& L, const std::string fileName)
{
	std::ifstream file(fileName);
	if (!file.is_open()) {
		fprintf(stderr, "ofstream failed!");
		exit(1);
	}
	file >> N >> L;
	std::multimap<std::vector<bool>, int> hashMap;
	for (auto i = 0; i < N; ++i) {
		std::vector<bool> data;
		for (auto j = 0; j < L; ++j) {
			char b;
			file >> b;
			if (b == '0') {
				data.push_back(false);
			}
			else if (b == '1') {
				data.push_back(true);
			}
		}
		hashMap.insert({ data, i });
	}
	return hashMap;
}

int main()
{
	const auto fileName = "hamming_one.txt";
	auto N = 0;
	auto L = 0;
	auto hashMap = readFile(N, L, fileName);
	auto start = std::chrono::steady_clock::now();
	findHammingOne(hashMap);
	auto end = std::chrono::steady_clock::now();
	std::cout << "Searching for pairs with the Hamming distance equal to one took "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
}