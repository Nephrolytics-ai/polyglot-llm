package utils

import (
	"errors"
	"fmt"
	"runtime"
	"strings"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/logging"
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

func WrapIfNotNil(err error, context ...string) error {
	if err == nil {
		return nil
	}

	callerName := "unknown"
	if pc, _, _, ok := runtime.Caller(1); ok {
		if fn := runtime.FuncForPC(pc); fn != nil {
			callerName = fn.Name()
		}
	}

	parts := make([]string, 0, 1+len(context))
	parts = append(parts, callerName)
	parts = append(parts, context...)

	return fmt.Errorf("%s: %w", strings.Join(parts, " - "), err)
}

func PrintStack(title string, log logging.Logger) {
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
