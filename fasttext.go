package fasttextgo

// #cgo LDFLAGS: -L${SRCDIR} -lfasttext -lstdc++ -lm
// #include <stdlib.h>
// #include "src/fasttext_wrapper.h"
// void load_model(char *path);
// int predict(char *query, float *prob, char **buf, int *size);
// int predict_k(char *query, int k, float *prob, char **buf, int *sizes);
import "C"
import (
	"errors"
	"unsafe"
)

// LoadModel - load FastText model
func LoadModel(path string) {
	C.load_model(C.CString(path))
}

// Predict - predict
func Predict(sentence string) (prob float32, label string, err error) {

	var cprob C.float
	var buf *C.char
	var size C.int

	if sentence != "" && sentence[len(sentence)-1] != '\n' {
		sentence += "\n"
	}
	cs := C.CString(sentence)
	ret := C.predict(cs, &cprob, &buf, &size)
	C.free(unsafe.Pointer(cs))

	if ret != 0 {
		err = errors.New("error in prediction")
	} else {
		label = C.GoStringN(buf, size)
		prob = float32(cprob)
		C.free(unsafe.Pointer(buf))
	}

	return prob, label, err
}

// Prediction is used in a result of PredictK
type Prediction struct {
	Prob  float32
	Label string
}

// PredictK returns K top predictions
func PredictK(sentence string, k int) ([]Prediction, error) {
	var cprob *C.float
	cprob = (*C.float)(C.calloc(C.size_t(k), C.sizeof_float))
	var csizes *C.int
	csizes = (*C.int)(C.calloc(C.size_t(k), C.sizeof_int))
	var buf *C.char

	if sentence != "" && sentence[len(sentence)-1] != '\n' {
		sentence += "\n"
	}
	cs := C.CString(sentence)
	ret := C.predict_k(cs, C.int(k), cprob, &buf, csizes)
	C.free(unsafe.Pointer(cs))

	if ret == -1 {
		C.free(unsafe.Pointer(cprob))
		C.free(unsafe.Pointer(csizes))
		return nil, errors.New("error in prediction")
	}

	ps := make([]Prediction, 0, ret)
	pos := 0
	for i := 0; i < int(ret); i++ {
		f := float32(*(*C.float)(unsafe.Pointer(uintptr(unsafe.Pointer(cprob)) + uintptr(i*C.sizeof_float))))
		size := *(*C.int)(unsafe.Pointer(uintptr(unsafe.Pointer(csizes)) + uintptr(i*C.sizeof_int)))
		s := C.GoStringN((*C.char)(unsafe.Pointer(uintptr(unsafe.Pointer(buf))+uintptr(pos))), C.int(size))
		ps = append(ps, Prediction{f, s})
		pos += int(size)
	}
	C.free(unsafe.Pointer(cprob))
	C.free(unsafe.Pointer(csizes))
	C.free(unsafe.Pointer(buf))

	return ps, nil
}

// FastText represent instance of fasttext classifier
type FastText struct {
	ft unsafe.Pointer
}

// New returns new instance of fasttext classifier
func New() *FastText {
	return &FastText{C.fasttext_new()}
}

// Close deletes the underlying fastext classifier. It is safe to call
// Close multiple times, but calling other method on the closed
// FastText instance will panic.
func (f *FastText) Close() {
	if f.ft != nil {
		C.fasttext_delete(f.ft)
		f.ft = nil
	}
}

// Args represent arguments for training (value of -1 and "" mean to
// use the default value)
type Args struct {
	LR            float64
	LRUpdateRate  int
	Dim           int
	Ws            int
	Epoch         int
	MinCount      int
	MinCountLabel int
	Neg           int
	WordNgrams    int
	Loss          string
	Bucket        int
	MinN          int
	MaxN          int
	Thread        int
	T             float64
	Label         string
	Verbose       int
}

// Supervised performs supervised training
func (f *FastText) Supervised(input string, output string, args *Args) error {
	if f.ft == nil {
		return errors.New("Supervised called on closed FastText")
	}
	return f.train(input, output, "sup", args)
}

func (f *FastText) train(input string, output string, model string, args *Args) error {
	var a C.fasttext_args
	a.input = C.CString(input)
	defer C.free(unsafe.Pointer(a.input))
	a.output = C.CString(output)
	defer C.free(unsafe.Pointer(a.output))
	a.lr = C.double(args.LR)
	a.lrUpdateRate = C.int(args.LRUpdateRate)
	a.dim = C.int(args.Dim)
	a.ws = C.int(args.Ws)
	a.epoch = C.int(args.Epoch)
	a.minCount = C.int(args.MinCount)
	a.minCountLabel = C.int(args.MinCountLabel)
	a.neg = C.int(args.Neg)
	a.wordNgrams = C.int(args.WordNgrams)
	a.loss = nil
	if args.Loss != "" {
		a.loss = C.CString(args.Loss)
		defer C.free(unsafe.Pointer(a.loss))
	}
	a.model = C.CString(model)
	a.bucket = C.int(args.Bucket)
	a.minn = C.int(args.MinN)
	a.maxn = C.int(args.MaxN)
	a.thread = C.int(args.Thread)
	a.t = C.double(args.T)
	a.label = nil
	if args.Label != "" {
		a.label = C.CString(args.Label)
		defer C.free(unsafe.Pointer(a.label))
	}
	a.verbose = C.int(args.Verbose)
	var e = C.fasttext_train(f.ft, &a)
	if e != nil {
		defer C.free(unsafe.Pointer(e))
		return errors.New(C.GoString(e))
	}
	return nil
}
