#!/usr/bin/env python
import sys
import warnings
from dotenv import load_dotenv
from crew import VideoSummary

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information
load_dotenv()

def run():
    """
    Run the crew.
    """
    inputs = {
        'youtube_url':'https://www.youtube.com/watch?v=JR36oH35Fgg',
        #'youtube_url': 'https://www.youtube.com/watch?v=wjr4iFzlLjo'
    }
    
    VideoSummary().crew().kickoff(inputs=inputs)

if __name__ == "__main__":
    run()
    
