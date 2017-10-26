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

void load_model(char *path);
void *fasttext_new();
void fasttext_delete(void *ft);
char *fasttext_train(void *ft, fasttext_args *args);
void fasttext_load_model(void *ft, char *path);
int fasttext_predict(void *ft, char *query, float *prob, char **buf, int *size);
int fasttext_predict_k(void *ft, char *query, int k, float *prob, char **buf, int *sizes);

#ifdef __cplusplus
}
#endif
