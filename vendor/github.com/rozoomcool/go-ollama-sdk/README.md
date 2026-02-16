# 🧠 Ollama Go SDK

A **Go SDK** for working with the [Ollama](https://ollama.com/) API locally or remotely.

Fully typed, easy to use, streaming-capable and ready for use.

---

## 🚀 Features

- ✅ Generate completions (`/api/generate`)
- ✅ Chat mode (`/api/chat`)
- ✅ Streaming support
- ✅ Model management (`/api/pull`)

---

## 📦 Installation

```bash
go get github.com/yourname/ollama-go-sdk
```

## 🌐 Quick Start

```go
package main

import (
    "fmt"
    "github.com/yourname/ollama-go-sdk"
)

func main() {
    client := ollama.NewClient("http://localhost:11434")

    result, err := client.Generate("llama3", "Tell me a joke.")
    if err != nil {
        panic(err)
    }
    fmt.Println("Response:", result)
}
```

## 🔄 Streaming Usage

```go
client.GenerateStream(
    "llama3",
    "Explain quantum mechanics.",
    func(chunk string) { fmt.Print(chunk) },
    func() { fmt.Println("\n[Done]") },
)
```

## 💬 Chat Mode Usage

```go
messages := []ollama.ChatMessage{
    {Role: "system", Content: "You are a helpful assistant."},
    {Role: "user", Content: "Tell me about black holes."},
}

response, err := client.Chat("llama3", messages)
if err != nil {
    panic(err)
}
fmt.Println(response)
```

## 📥 Model Pull Usage

`Download models dynamically:`

```go
err := client.PullModel("llama3", func(status string) {
    fmt.Println("Pull Status:", status)
})
if err != nil {
    panic(err)
}
```