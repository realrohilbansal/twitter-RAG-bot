# Twitter-RAG-bot
A twitter bot that fact checks or answers the tweet content of the parent tweet. 
In the backend it uses Langchain, gpt-4-turbo, and airtable for storage.

Steps to use:
1. git clone 
2. The faiss index contains vector embedding database for the data we have used. In case you want to use your own data, change the file name path in `enquiry.py` `line 50: file = "..."` and run the main function using `python enquiry.py` to create a new vector database.
3. Run the  `main.py` file.

! This is a personal project part of my langchain learning process.
