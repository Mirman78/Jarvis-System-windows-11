import feedparser

def get_bbc_news(limit=10):
    """
    Fetches the latest top stories from BBC News (World edition).
    """
    bbc_url = "http://feeds.bbci.co.uk/news/rss.xml"
    
    try:
        feed = feedparser.parse(
            bbc_url,
            agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        
        if not feed.entries:
            if feed.bozo:
                return [f"BBC News feed appears malformed: {feed.bozo_exception}"]
            return ["I couldn't fetch the BBC news at the moment (no entries)."]
        
        headlines = []
        for entry in feed.entries[:limit]:
            headlines.append(entry.title)
        return headlines
    
    except Exception as e:
        return [f"Error fetching BBC News: {str(e)}"]

def get_google_news(limit=10):
    """
    Fetches the latest top stories from Google News (World edition).
    """
    # Google News RSS feed (works reliably)
    google_url = "https://news.google.com/rss"
    
    try:
        feed = feedparser.parse(
            google_url,
            agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        
        if not feed.entries:
            if feed.bozo:
                return [f"Google News feed appears malformed: {feed.bozo_exception}"]
            return ["I couldn't fetch the Google News at the moment (no entries)."]
        
        headlines = []
        for entry in feed.entries[:limit]:
            headlines.append(entry.title)
        return headlines
    
    except Exception as e:
        return [f"Error fetching Google News: {str(e)}"]

if __name__ == "__main__":
    print("Jarvis, fetch the news...\n")
    
    print("--- BBC TOP STORIES ---")
    bbc_stories = get_bbc_news(limit=10)
    for i, story in enumerate(bbc_stories, 1):
        print(f"{i}. {story}")
    
    print("\n--- GOOGLE NEWS TOP STORIES ---")
    google_stories = get_google_news(limit=10)
    for i, story in enumerate(google_stories, 1):
        print(f"{i}. {story}")