.PHONY: build test test-unit test-external build-unit build-external

GO ?= go
GOFLAGS ?= -mod=vendor

build-unit:
	GOFLAGS="$(GOFLAGS)" $(GO) test -run='^$$' ./pkg/...

build-external:
	GOFLAGS="$(GOFLAGS)" $(GO) test -run='^$$' ./tests/...

build: build-unit build-external

test-unit:
	GOFLAGS="$(GOFLAGS)" $(GO) test ./pkg/...

test-external:
	GOFLAGS="$(GOFLAGS)" $(GO) test ./tests/...

test: test-unit test-external
