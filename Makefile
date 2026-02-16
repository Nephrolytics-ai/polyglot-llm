.PHONY: test test-unit test-external

GO ?= go
GOCACHE_DIR ?= /tmp/go-build-cache

test-unit:
	GOCACHE=$(GOCACHE_DIR) $(GO) test ./pkg/...

test-external:
	GOCACHE=$(GOCACHE_DIR) $(GO) test ./tests/...

test: test-unit test-external
