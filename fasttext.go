package fasttextgo

// #cgo LDFLAGS: -L${SRCDIR} -lfasttext -lstdc++ -lm
// #include <stdlib.h>
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
