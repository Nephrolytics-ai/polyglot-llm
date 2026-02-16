.PHONY: test test-unit test-external

GO ?= go
GOCACHE_DIR ?= /tmp/go-build-cache

test-unit:
	mkdir -p $(GOCACHE_DIR)
	GOCACHE=$(GOCACHE_DIR) $(GO) test ./pkg/...

test-external:
	mkdir -p $(GOCACHE_DIR)
	GOCACHE=$(GOCACHE_DIR) $(GO) test ./tests/...

test: test-unit test-external
