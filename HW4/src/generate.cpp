#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <climits>

using namespace std;

// source : https://www.codeproject.com/Tips/537904/Console-simple-progress
void DoProgress( string label, int step, int total )
{
    //progress width
    const int pwidth = 72;
 
    //minus label len
    long long width = (long long)pwidth - label.length();
    long long pos = (long long)( step * width ) / total ;
 
     
    float percent = (float)((long double)(step*100.0) / total);
 
    // printf( "%s[", label );
    cout<<label<<"[";
 
    //fill progress bar with =
    for ( int i = 0; i < pos; i++ )  printf( "%c", '=' );
 
    //fill progress bar with spaces
    // printf( "% *c", width - pos + 1, "]");
    for ( int i = 0; i < width - pos; i++ )  printf( "%c", ' ' );
    printf( "%c", ']');
    printf( " %3.2f%%\r", percent );
}

int main(int argc, char** argv) {
  // Read arguments
  int R = 1024;
  int S = 1024;
  string inputPath = "data/input.bin";
  string outputPath = "data/output.bin";
  if (argc == 1) {
    // nothing
  } else if (argc == 5) {
    R = atoi(argv[1]);
    S = atoi(argv[2]);
    inputPath = argv[3];
    outputPath = argv[4];
  } else {
    std::cout << "./generate R S inputPath outputPath" << endl;
    exit(-1);
  }
  // Print arguments
  std::cout << endl << "============= Join Tables =============" << endl;
  std::cout << "R : " << R << endl;
  std::cout << "S : " << S << endl;
  std::cout << "inputPath : " << inputPath << endl;
  std::cout << "outputPath : " << outputPath << endl;
  std::cout << "=======================================" << endl;
  // Initilize variables
  int MAX_ID= R + S + 1000000;                      // !!!CHANGE THIS VALUE RESPECT TO THE TABLE SIZES!!!
  bool* flagA = new bool[MAX_ID];
  bool* flagB = new bool[MAX_ID];
  long long* tableA = new long long[R];
  long long* tableB = new long long[S];
  vector<long long> solution;

  long long* tableID = new long long[MAX_ID];
  // Generate id table
  {
    // random generator
    random_device device;
    mt19937 generator(device());
    uniform_int_distribution<int> distribution(0, MAX_ID - 1);
    {
      // std::cout << "Generate flags for tableA..." << endl;
      int cnt = R;
      while (cnt > 0) {
        int index = distribution(generator);
        if (flagA[index] == false) {
          flagA[index] = true;
          cnt--;
        }
        if ((R-cnt)%1000 == 0) DoProgress("Generate flags for tableA:",R-cnt,R);
      }
      printf("\n");
    }
    {
      // std::cout << "Generate flags for tableB..." << endl;
      int cnt = S;
      int index = 0;
      while (cnt > 0) {
        int index = distribution(generator);
        if (flagB[index] == false) {
          flagB[index] = true;
          cnt--;
        }
        if ((S-cnt)%(1000) == 0) DoProgress("Generate flags for tableB:",S-cnt,S);
      }
      printf("\n");
    }
  }

  //Initialize ID array
  {
    random_device device;
    mt19937_64 now_rand(device());
    uniform_int_distribution<long long> ll_dist(0, LLONG_MAX - 1);
    for (int i = 0; i < MAX_ID; i++) {
      if (flagA[i] || flagB[i]){
        tableID[i] = ll_dist(now_rand);
      }
      if ((i%(1000)) == 0) DoProgress("Initialize ID table",i,MAX_ID);
    }
      printf("\n");
  }

  // Generate tableA and tableB
  {
    // std::cout << "Fill tableA..." << endl;
    int index = 0;
    for (int i = 0; i < MAX_ID; i++) {
      if (flagA[i]) {
        tableA[index] = tableID[i];
        index++;
        if ((i%(1000)) == 0) DoProgress("Fill tableA:",i,MAX_ID);
      }
    }
    printf("\n");
    random_shuffle(tableA, tableA + R);
  }
  {
    // std::cout << "Fill tableB..." << endl;
    int index = 0;
    for (int i = 0; i < MAX_ID; i++) {
      if (flagB[i]) {
        tableB[index] = tableID[i];
        index++;
        if ((i%(1000)) == 0) DoProgress("Fill tableB:",i,MAX_ID);
      }
    }
    printf("\n");
    random_shuffle(tableB, tableB + S);
  }
  // Generate solution
  {
    // std::cout << "Fill tableC(intersection of tableA and tableB)" << endl;
    for (int i = 0; i < MAX_ID; i++) {
      if (flagA[i] && flagB[i]) {
        solution.push_back(tableID[i]);
        if ((i%(1000)) == 0) DoProgress("Fill tableC(intersection of tableA and tableB):",i,MAX_ID);
      }
    }
    printf("\n");
    random_shuffle(solution.begin(), solution.end());
  }


  // Write input file
  std::cout << "Write input file..." << endl;
  ofstream inputFile(inputPath.data(),ios::out | ios::binary);
  if (inputFile.is_open()) {
    inputFile.write((char*)&R, sizeof(int));
    inputFile.write((char*)&S, sizeof(int));
    inputFile.write((char*)tableA, sizeof(long long)*R);
    inputFile.write((char*)tableB, sizeof(long long)*S);

    inputFile.close();
  }
  // Write output file
  std::cout << "Write output file..." << endl;
  ofstream outputFile(outputPath.data(), ios::out | ios::binary);
  if (outputFile.is_open()) {
    int solution_size = solution.size();
    outputFile.write((char*)&solution_size,sizeof(int));
    for (auto const& value : solution) outputFile.write((char*)&value,sizeof(long long));
    outputFile.close();
  }
  // Free memory
  std::cout << "Free memories..." << endl;
  delete[] flagA;
  delete[] flagB;
  delete[] tableA;
  delete[] tableB;
  delete[] tableID;

  return 0;
}
