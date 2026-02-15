package logging

import (
	"context"
	"sync"
)

type LoggerFactory interface {
	CreateLogger(ctx context.Context) Logger
}

var (
	loggerFactoryMu sync.RWMutex
	loggerFactory   LoggerFactory
)

func SetLoggerFactory(factory LoggerFactory) {
	loggerFactoryMu.Lock()
	defer loggerFactoryMu.Unlock()

	loggerFactory = factory
}

func GetLoggerFactory() LoggerFactory {
	loggerFactoryMu.RLock()
	defer loggerFactoryMu.RUnlock()

	return loggerFactory
}
