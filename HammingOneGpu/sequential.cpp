#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>

constexpr uint_fast64_t firstMultiplier = 16807;
constexpr uint_fast64_t secondMultiplier = 8121;
constexpr uint_fast64_t firstModulus = 2147483647;
constexpr uint_fast64_t secondModulus = 2305843009213693951;

std::multimap<std::pair<int_fast64_t, int_fast64_t>, int> calculateHashes(const std::vector<std::vector<char>>& data)
{
	std::multimap < std::pair<int_fast64_t, int_fast64_t>, int> hashes{};
	for (auto i = 0; i < data.size(); ++i) {
		uint_fast64_t firstHash = 0;
		uint_fast64_t secondHash = 0;
		uint_fast64_t firstCoefficient = firstMultiplier;
		uint_fast64_t secondCoefficient = 1;
		for (auto j = 0; j < data[i].size(); ++j) {
			firstHash = (firstHash + firstCoefficient * data[i][j]) % firstModulus;
			secondHash = (secondHash + secondCoefficient * data[i][j]) % secondModulus;
			firstCoefficient = firstCoefficient * firstMultiplier % firstModulus;
			secondCoefficient = secondCoefficient * secondMultiplier % secondModulus;
		}
		hashes.insert({ { firstHash, secondHash }, i });
	}
	return hashes;
}

void findHammingOne(const std::multimap<std::pair<int_fast64_t, int_fast64_t>, int>& hashes, const std::vector<std::vector<char>>& data)
{
	for (const auto& el : hashes) {
		uint_fast64_t firstCoefficient = firstMultiplier;
		uint_fast64_t secondCoefficient = 1;
		for (int i = 0; i < data[el.second].size(); ++i) {
			int s = data[el.second][i] == 1 ? 1 : -1;
			std::pair<int_fast64_t, int_fast64_t> newHashPair{
				(el.first.first + s * firstCoefficient + firstModulus) % firstModulus,
				(el.first.second + s * secondCoefficient + secondModulus) % secondModulus
			};
			auto range = hashes.equal_range(newHashPair);
			for (auto& it = range.first; it != range.second; ++it) {
				if (el.second < (*it).second) {
					printf("(%d, %d)\n", el.second, (*it).second);
				}
			}
			firstCoefficient = firstCoefficient * firstMultiplier % firstModulus;
			secondCoefficient = secondCoefficient * secondMultiplier % secondModulus;
		}
	}
}

std::vector<std::vector<char>> readFile(int& N, int& L, const std::string fileName)
{
	std::ifstream file(fileName);
	if (!file.is_open()) {
		fprintf(stderr, "ofstream failed!");
		exit(1);
	}
	file >> N >> L;
	std::vector<std::vector<char>> data{};
	for (auto i = 0; i < N; ++i) {
		std::vector<char> sequence;
		for (auto j = 0; j < L; ++j) {
			char b = 0;
			file >> b;
			if (b == '0') {
				sequence.push_back(2);
			}
			else if (b == '1') {
				sequence.push_back(1);
			}
		}
		data.push_back(sequence);
	}
	return data;
}

int main()
{
	const auto fileName = "hamming_one.txt";
	auto N = 0;
	auto L = 0;
	auto data = readFile(N, L, fileName);
	auto start = std::chrono::steady_clock::now();
	auto hashes = calculateHashes(data);
	findHammingOne(hashes, data);
	auto end = std::chrono::steady_clock::now();
	std::cout << "Searching for pairs with the Hamming distance equal to one took "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
}