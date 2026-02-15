package utils

import (
	"errors"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
	"runtime"
	"strings"
)

// ContainsErrorSubstring checks if the error or any of its wrapped errors contain the target substring.
func ContainsErrorSubstring(err error, target string) bool {
	for err != nil {
		if strings.Contains(err.Error(), target) {
			return true
		}
		err = errors.Unwrap(err)
	}
	return false
}
func PrintStack(title string, log Logger) {
	log.Errorf(" %s Stack trace:", title)
	// skip = 2 to ignore printStack and its caller (defer wrapper)
	for i := 2; ; i++ {
		pc, file, line, ok := runtime.Caller(i)
		if !ok {
			break
		}
		fn := runtime.FuncForPC(pc)
		log.Errorf("     *** %s (%s:%d)", fn.Name(), file, line)
	}
}
