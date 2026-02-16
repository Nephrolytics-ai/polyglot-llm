# Vendor License Audit

- Date: 2026-02-16
- Scope: `vendor/` dependencies from `vendor/modules.txt`
- Modules scanned: 54
- Risky/copyleft findings (GPL/AGPL/LGPL/SSPL/MPL/EPL/CDDL): 0

## Summary

- No GPL-family or similarly restrictive licenses were detected in vendored dependencies.
- Detected license mix:
  - Apache-2.0: 28
  - MIT: 13
  - MIT + Apache-2.0 (dual): 1
  - BSD-like: 11
  - ISC: 1

## Scan Method

- Parsed modules from `vendor/modules.txt`.
- For each module root, inspected the first matching file among `LICENSE*`, `LICENCE*`, `COPYING*`, and `NOTICE*` (up to depth 3).
- Searched vendored files for risky markers including: `GPL`, `AGPL`, `LGPL`, `SSPL`, `copyleft`, `MPL`, `EPL`, `CDDL`, `Business Source License`, and `Elastic License`.

## Detailed Results

| Module | Version | License | Evidence File |
|---|---:|---|---|
| `cloud.google.com/go` | `v0.116.0` | Apache-2.0 | `cloud.google.com/go/LICENSE` |
| `cloud.google.com/go/auth` | `v0.9.3` | Apache-2.0 | `cloud.google.com/go/auth/LICENSE` |
| `cloud.google.com/go/compute/metadata` | `v0.5.0` | Apache-2.0 | `cloud.google.com/go/compute/metadata/LICENSE` |
| `github.com/aws/aws-sdk-go-v2` | `v1.41.1` | Apache-2.0 | `github.com/aws/aws-sdk-go-v2/LICENSE.txt` |
| `github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream` | `v1.7.4` | Apache-2.0 | `github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream/LICENSE.txt` |
| `github.com/aws/aws-sdk-go-v2/config` | `v1.32.7` | Apache-2.0 | `github.com/aws/aws-sdk-go-v2/config/LICENSE.txt` |
| `github.com/aws/aws-sdk-go-v2/credentials` | `v1.19.7` | Apache-2.0 | `github.com/aws/aws-sdk-go-v2/credentials/LICENSE.txt` |
| `github.com/aws/aws-sdk-go-v2/feature/ec2/imds` | `v1.18.17` | Apache-2.0 | `github.com/aws/aws-sdk-go-v2/feature/ec2/imds/LICENSE.txt` |
| `github.com/aws/aws-sdk-go-v2/internal/configsources` | `v1.4.17` | Apache-2.0 | `github.com/aws/aws-sdk-go-v2/internal/configsources/LICENSE.txt` |
| `github.com/aws/aws-sdk-go-v2/internal/endpoints/v2` | `v2.7.17` | Apache-2.0 | `github.com/aws/aws-sdk-go-v2/internal/endpoints/v2/LICENSE.txt` |
| `github.com/aws/aws-sdk-go-v2/internal/ini` | `v1.8.4` | Apache-2.0 | `github.com/aws/aws-sdk-go-v2/internal/ini/LICENSE.txt` |
| `github.com/aws/aws-sdk-go-v2/service/bedrockruntime` | `v1.49.0` | Apache-2.0 | `github.com/aws/aws-sdk-go-v2/service/bedrockruntime/LICENSE.txt` |
| `github.com/aws/aws-sdk-go-v2/service/internal/accept-encoding` | `v1.13.4` | Apache-2.0 | `github.com/aws/aws-sdk-go-v2/service/internal/accept-encoding/LICENSE.txt` |
| `github.com/aws/aws-sdk-go-v2/service/internal/presigned-url` | `v1.13.17` | Apache-2.0 | `github.com/aws/aws-sdk-go-v2/service/internal/presigned-url/LICENSE.txt` |
| `github.com/aws/aws-sdk-go-v2/service/signin` | `v1.0.5` | Apache-2.0 | `github.com/aws/aws-sdk-go-v2/service/signin/LICENSE.txt` |
| `github.com/aws/aws-sdk-go-v2/service/sso` | `v1.30.9` | Apache-2.0 | `github.com/aws/aws-sdk-go-v2/service/sso/LICENSE.txt` |
| `github.com/aws/aws-sdk-go-v2/service/ssooidc` | `v1.35.13` | Apache-2.0 | `github.com/aws/aws-sdk-go-v2/service/ssooidc/LICENSE.txt` |
| `github.com/aws/aws-sdk-go-v2/service/sts` | `v1.41.6` | Apache-2.0 | `github.com/aws/aws-sdk-go-v2/service/sts/LICENSE.txt` |
| `github.com/aws/smithy-go` | `v1.24.0` | Apache-2.0 | `github.com/aws/smithy-go/LICENSE` |
| `github.com/bahlo/generic-list-go` | `v0.2.0` | BSD-like | `github.com/bahlo/generic-list-go/LICENSE` |
| `github.com/buger/jsonparser` | `v1.1.1` | MIT | `github.com/buger/jsonparser/LICENSE` |
| `github.com/davecgh/go-spew` | `v1.1.1` | ISC | `github.com/davecgh/go-spew/LICENSE` |
| `github.com/golang/groupcache` | `v0.0.0-20210331224755-41bb18bfe9da` | Apache-2.0 | `github.com/golang/groupcache/LICENSE` |
| `github.com/google/go-cmp` | `v0.6.0` | BSD-like | `github.com/google/go-cmp/LICENSE` |
| `github.com/google/s2a-go` | `v0.1.8` | Apache-2.0 | `github.com/google/s2a-go/LICENSE.md` |
| `github.com/google/uuid` | `v1.6.0` | BSD-like | `github.com/google/uuid/LICENSE` |
| `github.com/googleapis/enterprise-certificate-proxy` | `v0.3.4` | Apache-2.0 | `github.com/googleapis/enterprise-certificate-proxy/LICENSE` |
| `github.com/gorilla/websocket` | `v1.5.3` | BSD-like | `github.com/gorilla/websocket/LICENSE` |
| `github.com/invopop/jsonschema` | `v0.13.0` | MIT | `github.com/invopop/jsonschema/COPYING` |
| `github.com/joho/godotenv` | `v1.5.1` | MIT | `github.com/joho/godotenv/LICENCE` |
| `github.com/mailru/easyjson` | `v0.7.7` | MIT | `github.com/mailru/easyjson/LICENSE` |
| `github.com/mark3labs/mcp-go` | `v0.44.0` | MIT | `github.com/mark3labs/mcp-go/LICENSE` |
| `github.com/openai/openai-go/v3` | `v3.22.0` | Apache-2.0 | `github.com/openai/openai-go/v3/LICENSE` |
| `github.com/pmezard/go-difflib` | `v1.0.0` | BSD-like | `github.com/pmezard/go-difflib/LICENSE` |
| `github.com/rozoomcool/go-ollama-sdk` | `v0.0.0-20250620220025-710cf9a2c767` | MIT | `github.com/rozoomcool/go-ollama-sdk/LICENSE` |
| `github.com/sirupsen/logrus` | `v1.9.3` | MIT | `github.com/sirupsen/logrus/LICENSE` |
| `github.com/spf13/cast` | `v1.7.1` | MIT | `github.com/spf13/cast/LICENSE` |
| `github.com/stretchr/testify` | `v1.9.0` | MIT | `github.com/stretchr/testify/LICENSE` |
| `github.com/tidwall/gjson` | `v1.18.0` | MIT | `github.com/tidwall/gjson/LICENSE` |
| `github.com/tidwall/match` | `v1.1.1` | MIT | `github.com/tidwall/match/LICENSE` |
| `github.com/tidwall/pretty` | `v1.2.1` | MIT | `github.com/tidwall/pretty/LICENSE` |
| `github.com/tidwall/sjson` | `v1.2.5` | MIT | `github.com/tidwall/sjson/LICENSE` |
| `github.com/wk8/go-ordered-map/v2` | `v2.1.8` | Apache-2.0 | `github.com/wk8/go-ordered-map/v2/LICENSE` |
| `github.com/yosida95/uritemplate/v3` | `v3.0.2` | BSD-like | `github.com/yosida95/uritemplate/v3/LICENSE` |
| `go.opencensus.io` | `v0.24.0` | Apache-2.0 | `go.opencensus.io/LICENSE` |
| `golang.org/x/crypto` | `v0.36.0` | BSD-like | `golang.org/x/crypto/LICENSE` |
| `golang.org/x/net` | `v0.38.0` | BSD-like | `golang.org/x/net/LICENSE` |
| `golang.org/x/sys` | `v0.31.0` | BSD-like | `golang.org/x/sys/LICENSE` |
| `golang.org/x/text` | `v0.23.0` | BSD-like | `golang.org/x/text/LICENSE` |
| `google.golang.org/genai` | `v1.46.0` | Apache-2.0 | `google.golang.org/genai/LICENSE` |
| `google.golang.org/genproto/googleapis/rpc` | `v0.0.0-20240903143218-8af14fe29dc1` | Apache-2.0 | `google.golang.org/genproto/googleapis/rpc/LICENSE` |
| `google.golang.org/grpc` | `v1.66.2` | Apache-2.0 | `google.golang.org/grpc/LICENSE` |
| `google.golang.org/protobuf` | `v1.34.2` | BSD-like | `google.golang.org/protobuf/LICENSE` |
| `gopkg.in/yaml.v3` | `v3.0.1` | MIT + Apache-2.0 | `gopkg.in/yaml.v3/LICENSE` |
