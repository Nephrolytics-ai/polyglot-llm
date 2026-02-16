

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

