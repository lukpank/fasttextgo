#ifdef __cplusplus
extern "C" {
#endif

typedef struct fasttext_args {
  char *input;
  char *output;
  double lr;
  int lrUpdateRate;
  int dim;
  int ws;
  int epoch;
  int minCount;
  int minCountLabel;
  int neg;
  int wordNgrams;
  char *loss;
  char *model;
  int bucket;
  int minn;
  int maxn;
  int thread;
  double t;
  char *label;
  int verbose;
} fasttext_args;

void *fasttext_new();
void fasttext_delete(void *ft);
char *fasttext_train(void *p, fasttext_args *args);


#ifdef __cplusplus
}
#endif
