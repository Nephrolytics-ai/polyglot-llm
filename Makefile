.PHONY: test test-unit test-external

GO ?= go
GOFLAGS ?= -mod=vendor

test-unit:
	GOFLAGS="$(GOFLAGS)" $(GO) test ./pkg/...

test-external:
	GOFLAGS="$(GOFLAGS)" $(GO) test ./tests/...

test: test-unit test-external
