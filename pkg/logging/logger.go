package logging

import (
	"context"

	"github.com/sirupsen/logrus"
)

type logrusLogger struct {
	entry *logrus.Entry
}

func (l *logrusLogger) Debug(args ...any) {
	l.entry.Debug(args...)
}

func (l *logrusLogger) Debugf(format string, args ...any) {
	l.entry.Debugf(format, args...)
}

func (l *logrusLogger) Info(args ...any) {
	l.entry.Info(args...)
}

func (l *logrusLogger) Infof(format string, args ...any) {
	l.entry.Infof(format, args...)
}

func (l *logrusLogger) Error(args ...any) {
	l.entry.Error(args...)
}

func (l *logrusLogger) Errorf(format string, args ...any) {
	l.entry.Errorf(format, args...)
}

func (l *logrusLogger) Warn(args ...any) {
	l.entry.Warn(args...)
}

func (l *logrusLogger) Warnf(format string, args ...any) {
	l.entry.Warnf(format, args...)
}

func (l *logrusLogger) Fatal(args ...any) {
	l.entry.Fatal(args...)
}

func (l *logrusLogger) Fatalf(format string, args ...any) {
	l.entry.Fatalf(format, args...)
}

func NewLogger(ctx context.Context) Logger {
	factory := GetLoggerFactory()
	if factory != nil {
		return factory.CreateLogger(ctx)
	}

	return newLogrusLogger(ctx)
}

func newLogrusLogger(ctx context.Context) Logger {
	logger := logrus.New()
	return &logrusLogger{entry: logger.WithContext(ctx)}
}
