package fasttextgo

// #cgo LDFLAGS: -L${SRCDIR} -lfasttext -lstdc++ -lm
// #include <stdlib.h>
// void load_model(char *path);
// int predict(char *query, float *prob, char **buf);
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

	if sentence != "" && sentence[len(sentence)-1] != '\n' {
		sentence += "\n"
	}
	cs := C.CString(sentence)
	ret := C.predict(cs, &cprob, &buf)
	C.free(unsafe.Pointer(cs))

	if ret != 0 {
		err = errors.New("error in prediction")
	} else {
		label = C.GoString(buf)
		prob = float32(cprob)
		C.free(unsafe.Pointer(buf))
	}

	return prob, label, err
}
