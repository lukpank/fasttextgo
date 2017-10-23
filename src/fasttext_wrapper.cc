#include <iostream>
#include <istream>
#include "fasttext.h"
#include "real.h"
#include <streambuf>
#include <string.h>
#include "args.h"
#include "fasttext_wrapper.h"

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

int predict_k(char *query, int k, float *prob, char **buf, int *sizes) {
  membuf sbuf(query, query + strlen(query));
  std::istream in(&sbuf);

  std::vector<std::pair<fasttext::real, std::string>> predictions;

  g_fasttext_model.predict(in, k, predictions);

  std::vector<std::string> labels;
  size_t size = 0;
  int n = 0;
  for (auto it = predictions.cbegin(); it != predictions.cend(); it++) {
    prob[n] = (float)it->first;
    sizes[n++] = it->second.size();
    size += it->second.size();
    labels.push_back(it->second);
  }
  *buf = (char *)malloc(size);
  if (*buf == NULL) {
    return -1;
  }
  char *p = *buf;
  for (int i = 0; i < n; i++) {
    memcpy(p, labels[i].c_str(), sizes[i]);
    p += sizes[i];
  }
  return n;
}

void *fasttext_new() {
  return new fasttext::FastText();
}

void fasttext_delete(void *p) {
  delete ((fasttext::FastText*) p);
}

char *fasttext_train(void *p, fasttext_args *args) {
  fasttext::FastText *ft = (fasttext::FastText*) p;
  std::shared_ptr<fasttext::Args> a = std::make_shared<fasttext::Args>();
  a->input = args->input;
  a->output = args->output;
  if (args->lr != -1) {
    a->lr = args->lr;
  }
  if (args->lrUpdateRate != -1) {
    a->lrUpdateRate = args->lrUpdateRate;
  }
  if (args->dim != -1 ) {
    a->dim = args->dim;
  }
  if (args->ws != -1) {
    a->ws = args->ws;
  }
  if (args->epoch != -1) {
    a->epoch = args->epoch;
  }
  if (args->minCount != -1) {
    a->minCount = args->minCount;
  }
  if (args->minCountLabel != -1) {
    a->minCountLabel = args->minCountLabel;
  }
  if (args->neg != -1) {
    a->neg = args->neg;
  }
  if (args->wordNgrams != -1) {
    a->wordNgrams = args->wordNgrams;
  }
  if (args->loss != NULL) {
    if (strcmp(args->loss, "hs") == 0) {
      a->loss = fasttext::loss_name::hs;
    } else if (strcmp(args->loss, "ns") == 0) {
      a->loss = fasttext::loss_name::ns;
    } else if (strcmp(args->loss, "softmax") == 0) {
      a->loss = fasttext::loss_name::softmax;
    } else {
      return strdup("unrecognized loss value");
    }
  }
  if (args->model != NULL) {
    if (strcmp(args->model, "cbow") == 0) {
      a->model = fasttext::model_name::cbow;
    } else if (strcmp(args->model, "sg") == 0) {
      a->model = fasttext::model_name::sg;
    } else if (strcmp(args->model, "sup") == 0) {
      a->model = fasttext::model_name::sup;
    } else {
      return strdup("unrecognized model value");
    }
  }
  if (args->bucket != -1) {
    a->bucket = args->bucket;
  }
  if (args->minn != -1) {
    a->minn = args->minn;
  }
  if (args->maxn != -1) {
    a->maxn = args->maxn;
  }
  if (args->thread != -1) {
    a->thread = args->thread;
  }
  if (args->t != -1) {
    a->t = args->t;
  }
  if (args->label != "") {
    a->label = args->label;
  }
  if (args->verbose != -1) {
    a->verbose = args->verbose;
  }
  ft->train(a);
  return NULL;
}

}
