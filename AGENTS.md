# Code Rules & Preferences

Audience: developers and assistants working in this repo.

## Logging & Errors
- Do **not** write to stdout/stderr with `fmt.Print*` or `println`.
- If logging is needed, use `pkg/logger.NewLogger(ctx)` sparingly.
- Wrap errors with `errorutils.WrapIfNotNil`, using the function name as context.

## Testing
- Prefer the `testify` suite pattern with `suite`, `assert`, and `require`.
- Keep tests deterministic (fixed clocks, stable fixtures); avoid network and randomness.

## Files & Style
- Default to ASCII.
- Use existing normalized lab/vital names; prefer the latest rule config when available.
- Run `gofmt` on Go code; avoid noisy output in CI/test runs.

## Data Handling
- Avoid modifying unrelated files; don’t revert user changes.
- Respect missing/stale-data handling before running diagnostic calculations.
- If you see no code or comments, assume its ok, dont change it unless instructed.  
# Spelling and Grammer
 - I am a bad speller, if a type name I give you looks like it has a spelling error, auto correct please

# API Conventions
- Always have the first input be a context.Context and the last return value be an error.
- 