import tweepy
from airtable import Airtable
import os
from datetime import datetime, timedelta
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from enquiry import query

from langchain_openai import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

#import api keys fromm .env file
from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings(api_key=os.getenv("openai_api_key"))

vectordb = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization = True)

# TwitterBot class
class TwitterBot: 
    def __init__(self):
        self.twitter_api = tweepy.Client(bearer_token=os.getenv("twitter_bearer_key"),
                                        consumer_key=os.getenv("twitter_api_key"),
                                        consumer_secret=os.getenv("twitter_api_key_secret"),
                                        access_token=os.getenv("twitter_access_token"), 
                                        access_token_secret=os.getenv("twitter_access_token_secret"))
    
        self.airtable = Airtable(os.getenv("Airtable_base_id"), os.getenv("Airtable_personal_access_token"))

        # Create a llm instance for our twitter bot
        self.llm = ChatOpenAI(temperature=0.6, openai_api_key=os.getenv("openai_api_key"), model="gpt-4-1106-preview")

        self.twitter_mention_id = self.get_me_id()
        self.twitter_response_limit = 10 # number of tweets to respond to at a time 

        # Record logging, for debugging
        self.mentions_found = 0
        self.mentions_responded_to = 0
        self.mentions_error_replies = 0

    def generate_response(llm, mentioned_parent_tweet_text):
        # It would be nice to bring in information about the links, pictures, etc.
        # But that's for later.
        system_template = """
    You are a non-binary bot that is a good teacher on Gender and Sex.
    Your goal is to give a proper knowledgable answer in response to a piece of text from the user.

    % RESPONSE TONE:

    - Your response should be informative and helpful.
    - You should not respond if you feel that you do not have enough information to give a proper response.

    % RESPONSE FORMAT:

    - Respond in around 200 characters.
    - Respond in a paragraph format.
    - Response should not be in a question format.

    % RESPONSE CONTENT:

    - If you aren't sure what to say, or if you don't have enough information, you can say "I don't know" or "I'm not sure", but try to lead the user to a relevant answer.
    - Do not reply with anything racist, sexist, homophobic, transphobic, or otherwise offensive.

    """

        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

        human_template = "{text}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

        #get a chat completion from the formatted messages
        final_prompt = chat_prompt.format_prompt(text=mentioned_parent_tweet_text).to_messages()

        response = query(vectordb, llm, final_prompt)

        return response
    
    def respond_to_mention(self, mention, mentioned_conversation_tweet):
        response_text = self.generate_response(mentioned_conversation_tweet.text)
        
        # Try and create the response to the tweet. If it fails, log it and move on
        try:
            response_tweet = self.twitter_api.create_tweet(text=response_text, in_reply_to_tweet_id=mention.id)
            self.mentions_replied += 1
        except Exception as e:
            print (e)
            self.mentions_replied_errors += 1
            return
        
        # Log the response in airtable if it was successful
        self.airtable.insert({
            'mentioned_conversation_tweet_id': str(mentioned_conversation_tweet.id),
            'mentioned_conversation_tweet_text': mentioned_conversation_tweet.text,
            'tweet_response_id': response_tweet.data['id'],
            'tweet_response_text': response_text,
            'tweet_response_created_at' : datetime.utcnow().isoformat(),
            'mentioned_at' : mention.created_at.isoformat()
        })
        return True
    
    def get_me_id(self):
            return self.twitter_api.get_me()[0].id
        
        # Returns the parent tweet text of a mention if it exists. Otherwise returns None
        # We use this to since we want to respond to the parent tweet, not the mention itself
    def get_mention_conversation_tweet(self, mention):
        # Check to see if mention has a field 'conversation_id' and if it's not null
        if mention.conversation_id is not None:
            conversation_tweet = self.twitter_api.get_tweet(mention.conversation_id).data
            return conversation_tweet
        return None

    # Get mentioned to the user thats authenticated and running the bot.
    # Using a lookback window of 2 hours to avoid parsing over too many tweets
    from datetime import datetime, timedelta  # Import the datetime module

    def get_mentions(self):
        # If doing this in prod make sure to deal with pagination. There could be a lot of mentions!
        # Get current time in UTC
        now = datetime.utcnow()

        # Subtract 2 hours to get the start time
        start_time = now - timedelta(minutes=20)

        # Convert to required string format
        start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        return self.twitter_api.get_users_mentions(id=self.twitter_mention_id,
                                                start_time=start_time_str,
                                                expansions=['referenced_tweets.id'],
                                                tweet_fields=['created_at', 'conversation_id']).data

    # Checking to see if we've already responded to a mention with what's logged in airtable
    def check_already_responded(self, mentioned_conversation_tweet_id):
        records = self.airtable.get_all(view='Grid view')
        for record in records:
            if record['fields'].get('mentioned_conversation_tweet_id') == str(mentioned_conversation_tweet_id):
                return True
        return False

    # Run through all mentioned tweets and generate a response
    def respond_to_mentions(self):
        mentions = self.get_mentions()

        # If no mentions, just return
        if not mentions:
            print("No mentions found")
            return
        
        self.mentions_found = len(mentions)

        for mention in mentions[:self.tweet_response_limit]:
            # Getting the mention's conversation tweet
            mentioned_conversation_tweet = self.get_mention_conversation_tweet(mention)
            
            # If the mention *is* the conversation or you've already responded, skip it and don't respond
            if (mentioned_conversation_tweet.id != mention.id
                and not self.check_already_responded(mentioned_conversation_tweet.id)):

                self.respond_to_mention(mention, mentioned_conversation_tweet)
        return True
    
    # The main entry point for the bot with some logging
    def execute_replies(self):
        print (f"Starting Job: {datetime.utcnow().isoformat()}")
        self.respond_to_mentions()
        print (f"Finished Job: {datetime.utcnow().isoformat()}, Found: {self.mentions_found}, Replied: {self.mentions_replied}, Errors: {self.mentions_replied_errors}")
