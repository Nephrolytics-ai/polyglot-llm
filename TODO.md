

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
