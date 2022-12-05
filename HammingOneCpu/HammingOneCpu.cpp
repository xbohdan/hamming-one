#include <iostream>
#include <fstream>
#include <vector>
#include <map>

void findHammingOne(std::multimap<std::vector<bool>, int> hashMap)
{
	for (auto it = hashMap.begin(); it != hashMap.end();) {
		for (int i = 0; i < (*it).first.size(); ++i) {
			std::vector<bool> data = (*it).first;
			data[i].flip();
			auto range = hashMap.equal_range(data);
			for (auto jt = range.first; jt != range.second; ++jt) {
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
	for (int i = 0; i < N; ++i) {
		std::vector<bool> data;
		for (int j = 0; j < L; ++j) {
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
	const std::string fileName = "hamming_one.txt";
	int N = 0;
	int L = 0;
	std::multimap<std::vector<bool>, int> hashMap = readFile(N, L, fileName);
	findHammingOne(hashMap);
}