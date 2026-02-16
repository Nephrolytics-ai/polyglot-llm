

Need to look how the Context is being added into the OPen AI client, looks like we are jsut adding strings now
We need to use the Open AI function to add context for the promnpt and translate the ContextMessageType into the appropriate message type for Open AI.



Add a type def for Factory functions for LLLMs.
Add in an Interface and Factory function for Vector generation

Add someway to return meta data with the responses. (Tokens, in/out/cached,latency etc...)
Define some common meta data and then let each implementation add to it as needed.

I want the generate method to return meta data as a map of string:string .
I want some common meta data to be included in all implementations, such as:
Add someway to return meta data with the responses. (Tokens, in/out/cached,latency etc...)
Define some common meta data and then let each implementation add to it as needed.
Make it a second return value from the Generate method


Review the test settings and standardize on using OPEN_API_TOKEN and GEMINI_KEY for settings.
Make a github workflow that runs the makefile targets to build unit tests and the integration tests, and make sure to set the appropriate secrets for the tests to run in the workflow.
The secrets are the same names.


Now we are going to make the Gemini implementation.  
Since Gemini does not support MCP nativly, use the tools adapter under the hood to implement the MCP functionality.  
Implement both the Generator Interface and the Factory functions for Typed and String generation.
Implement the Vector generation interface and factory function as well, using the Gemini embedding API.
Add a test to the test/* directory to implement an integration test for structure and string generation as well as vector generation, single and batch generation
Use the google.golang.org/genai for the Gemini implementation


Now we are going to make the Ollama implementation. 
Since Ollama API  does not support MCP nativly, use the tools adapter under the hood to implement the MCP functionality.  
Implement both the Generator Interface and the Factory functions for Typed and String generation.
Also implement the Vector generation interface and factory function as well, using the Ollama embedding API.
Add a test to the test/* directory to implement an integration test for structure and string generation and vector generation, single and batch generation
PUT a flag around this keyed of the boolean RUN_OLLAMA_TESTS so that it only runs when we want it to run, since it requires a local Ollama instance to be running with the correct models loaded.  Our integration does not have this yet
Use the https://pkg.go.dev/github.com/rozoomcool/go-ollama-sdk for the Ollama implementation
I dont think there is authentication for Ollama, if there is, let me know.
We wont include Ollama tests in the github workflow.



