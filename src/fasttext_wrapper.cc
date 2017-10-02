#include <iostream>
#include <istream>
#include "fasttext.h"
#include "real.h"
#include <streambuf>
#include <string.h>

extern "C" {

struct membuf : std::streambuf
{
    membuf(char* begin, char* end) {
        this->setg(begin, begin, end);
    }
};

fasttext::FastText g_fasttext_model;
bool g_fasttext_initialized = false;

void load_model(char *path) {
  if (!g_fasttext_initialized) {
    g_fasttext_model.loadModel(std::string(path));
    g_fasttext_initialized = true;
  }
}

int predict(char *query, float *prob, char **buf, int *size) {
  membuf sbuf(query, query + strlen(query));
  std::istream in(&sbuf);

  std::vector<std::pair<fasttext::real, std::string>> predictions;

  g_fasttext_model.predict(in, 1, predictions);

  for (auto it = predictions.cbegin(); it != predictions.cend(); it++) {
    *prob = (float)it->first;
    *size = it->second.size();
    *buf = (char*)malloc(*size);
    if (*buf == NULL) {
	    return 1;
    }
    memcpy(*buf, it->second.c_str(), *size);
    return 0;
  }
  return 1;
}

}
